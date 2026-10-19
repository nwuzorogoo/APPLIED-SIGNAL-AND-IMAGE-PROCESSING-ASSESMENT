"""Shared state for every task: the recordings, detected events, designed filters and helpers."""


from . import config
from . import filters as flt
from .events import PulseWindowSet, detect_pulses, detect_theta_burst, remove_pulses, window_mask
from .spectral import amplitude_spectrum, band_power, dominant_peak, spectral_peaks
from .timefreq import default_cwt


class StudyBase:
    """Holds the recordings, the quantities derived from them and the designed filters."""

    def __init__(self, records, n_draws=config.N_DRAWS):
        if n_draws < 3:
            raise ValueError("the stress test needs at least three draws per record")
        self.records = records
        self.ids = tuple(records)
        first = records[self.ids[0]]
        self.fs = first.fs
        self.n = first.n_samples
        self.duration = first.duration
        self.time = first.time
        self.nyquist = self.fs / 2
        self.n_draws = n_draws
        self.x = {i: r.data for i, r in records.items()}

        self.peaks = {i: spectral_peaks(self.x[i], self.fs, config.PEAK_NFFT,
                                        config.PEAK_MIN_AMPLITUDE_UV,
                                        config.PEAK_MIN_SPACING_HZ,
                                        config.CLINICAL_CEILING_HZ)
                      for i in self.ids}
        self.main_peaks = {i: [(f, a) for f, a in self.peaks[i] if a > config.DOMINANT_PEAK_UV]
                           for i in self.ids}
        self.pulses = {i: detect_pulses(self.x[i], self.fs) for i in self.ids}
        # pulses are subtracted before the burst is measured, because a broadband pulse adds
        # energy to the theta band; the raw-signal burst is kept to show the difference
        self.x_clean = {i: remove_pulses(self.x[i], self.pulses[i]) for i in self.ids}
        self.bursts = {i: detect_theta_burst(self.x[i], self.fs, pulses=self.pulses[i])
                       for i in self.ids}
        self.bursts_raw = {i: detect_theta_burst(self.x[i], self.fs) for i in self.ids}
        if not self.pulses[config.TRANSIENT_RECORD]:
            raise ValueError(f"no pulse found in record {config.TRANSIENT_RECORD}")
        self.pulse = self.pulses[config.TRANSIENT_RECORD][0]
        self.windows = PulseWindowSet(self.pulse)

        self.sos_butter = flt.butter_bandpass(config.DESIGN_ORDER, self.fs)
        self.sos_ellip = flt.ellip_bandpass(config.DESIGN_ORDER, self.fs)
        self.sos_notch = flt.notch(self.fs)
        self.sos_butter_notch = flt.cascade(self.sos_butter, self.sos_notch)
        self.enhance = flt.Enhancer(self.fs, self.sos_butter, self.sos_notch)
        self.cwt = default_cwt(self.fs)
        self.mc_raw = {}

    # ------------------------------------------------------------------ helpers
    def zp(self, sos, x):
        return flt.zero_phase(sos, x, self.fs)

    def band_power(self, x, low, high):
        return band_power(x, self.fs, low, high)

    def spectrum(self, x, nfft=None):
        return amplitude_spectrum(x, self.fs, nfft=nfft)

    def mask(self, span):
        return window_mask(self.time, *span)

    def probe(self, record_id, band):
        """Frequency of the strongest spectral line of a record in a named band."""
        return dominant_peak(self.peaks[record_id], config.BANDS[band])

    @property
    def high_band(self):
        """Band above clinical EEG, up to Nyquist."""
        return config.CLINICAL_CEILING_HZ, self.nyquist

    def events_summary(self):
        return {i: {"pulses_s": [[p.onset_s, p.offset_s] for p in self.pulses[i]],
                    "theta_burst_s": list(self.bursts[i][:2]),
                    "theta_burst_raw_s": list(self.bursts_raw[i][:2])}
                for i in self.ids}

