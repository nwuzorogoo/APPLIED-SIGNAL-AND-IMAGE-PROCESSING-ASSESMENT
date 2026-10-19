"""Task 3: filter design study, notch and clean-record performance."""
import numpy as np

from . import config
from . import filters as flt
from .spectral import amplitude_near
from .names import FILTER_NAMES


class Task3Filters:
    """Butterworth and elliptic design, parameters, notch and performance. Mixed into Study."""

    # ------------------------------------------------------------------ Task 3 design
    def design_study(self):
        fs = self.fs
        probes = {"at29": self.probe(config.ALIAS_RECORD, "beta"),
                  "at1_84": self.probe(config.TRANSIENT_RECORD, "delta"),
                  "at24": self.probe(config.CUTOFF_PROBE_RECORD, "beta")}
        band = np.linspace(*config.GROUP_DELAY_BAND, 400)

        def common(sos):
            return {"att50": flt.zero_phase_db_at(sos, fs, config.MAINS_HZ),
                    "att45": flt.zero_phase_db_at(sos, fs, config.CLINICAL_CEILING_HZ),
                    "at29": flt.zero_phase_db_at(sos, fs, probes["at29"]),
                    "at40": flt.zero_phase_db_at(sos, fs, config.PASSBAND[1]),
                    "at1": flt.zero_phase_db_at(sos, fs, 1.0)}

        def delay_range(sos):
            return float(np.ptp(flt.group_delay(sos, band, fs)) / fs * 1e3)

        out = {}
        for order in config.BUTTER_ORDERS:
            sos = flt.butter_bandpass(order, fs)
            out[f"butter{order}"] = {**common(sos),
                                     "at1_84": flt.zero_phase_db_at(sos, fs, probes["at1_84"]),
                                     "sections": len(sos), "gd_range_ms_causal": delay_range(sos)}
        stop_per_pass = config.ELLIP_STOP_DB      # the elliptic specification is per pass

        def stopband(sos):
            return {"stop_edge_hz": flt.stopband_edge_hz(sos, fs, stop_per_pass),
                    "zeros_hz": flt.transmission_zeros_hz(sos, fs)}

        for order in config.ELLIP_ORDERS:
            sos = flt.ellip_bandpass(order, fs)
            out[f"ellip{order}"] = {**common(sos), **stopband(sos), "sections": len(sos),
                                    "gd_range_ms_causal": delay_range(sos)}
        for ripple in config.ELLIP_RIPPLES_DB:
            sos = flt.ellip_bandpass(config.DESIGN_ORDER, fs, ripple_db=ripple)
            passband = flt.frequency_response(sos, fs)
            inside = (passband[0] > band[0]) & (passband[0] < band[-1])
            out[f"ellip4_rp{ripple:g}"] = {
                **common(sos), **stopband(sos),
                "ripple_zp_db": float(np.ptp(flt.to_zero_phase_db(passband[1][inside]))),
                "gd_range_ms_causal": delay_range(sos),
                "gd_40_ms_causal": float(flt.group_delay(sos, np.array([40.0]), fs)[0] / fs * 1e3),
                **self.ringing(sos)}
        for cutoff in config.BUTTER_CUTOFFS:
            sos = flt.butter_bandpass(config.DESIGN_ORDER, fs, high=cutoff)
            out[f"butter4_fc{cutoff}"] = {
                "at29": flt.zero_phase_db_at(sos, fs, probes["at29"]),
                "at24": flt.zero_phase_db_at(sos, fs, probes["at24"]),
                "att50": flt.zero_phase_db_at(sos, fs, config.MAINS_HZ)}
        for key, sos in (("butter", self.sos_butter), ("ellip", self.sos_ellip)):
            out[FILTER_NAMES[key] + "_max_pole_radius"] = flt.max_pole_radius(sos)
        w, h_ellip = flt.frequency_response(self.sos_ellip, fs)
        _, h_butter = flt.frequency_response(self.sos_butter, fs)
        passband = (w > band[0]) & (w < band[-1])
        out["ellip_ripple_dB_zp"] = float(np.ptp(flt.to_zero_phase_db(h_ellip[passband])))
        out["butter_ripple_dB_zp"] = float(np.ptp(flt.to_zero_phase_db(h_butter[passband])))
        return out

    # ------------------------------------------------------------------ notch
    def notch_study(self):
        fs = self.fs
        beta = self.probe(config.ALIAS_RECORD, "beta")
        out = {"f0": config.MAINS_HZ, "Q": config.NOTCH_Q,
               "bw_single_pass_Hz": config.MAINS_HZ / config.NOTCH_Q}
        for q in config.NOTCH_Q_STUDY:
            sos = flt.notch(fs, q=q)
            w, h = flt.frequency_response(sos, fs, 2 ** 18)
            gain = flt.to_zero_phase_db(h)
            near = (w > config.MAINS_HZ - 10) & (w < config.MAINS_HZ + 10) & (gain < -3.0)
            radius = flt.max_pole_radius(sos)
            out[f"Q{q}"] = {
                "bw3_zp_Hz": float(w[near].max() - w[near].min()),
                **{key: flt.zero_phase_db_at(sos, fs, f) for key, f in (
                    ("at50_zp", 50.0), ("at49p8_zp", 49.8), ("at49_zp", 49.0),
                    ("at45_zp", 45.0), ("at40_zp", 40.0), ("at29p4_zp", beta))},
                "pole_radius": radius,
                "tau_ms": float(-1 / (fs * np.log(radius)) * 1e3),
            }
        out["detect_clean"] = {i: flt.mains_present(self.x[i], fs) for i in self.ids}
        peak_changes, rms_changes = [], []
        for i in self.ids:
            y_b = self.zp(self.sos_butter, self.x[i])
            y_bn = self.zp(self.sos_butter_notch, self.x[i])
            spec_b = self.spectrum(y_b, config.PEAK_NFFT)
            spec_bn = self.spectrum(y_bn, config.PEAK_NFFT)
            for f, _ in self.main_peaks[i]:
                ratio = amplitude_near(*spec_bn, f, 0.05) / amplitude_near(*spec_b, f, 0.05)
                peak_changes.append(abs(20 * np.log10(ratio)))
            rms_changes.append(float(np.sqrt(np.mean((y_bn - y_b) ** 2))))
        out["clean_max_peak_change_dB"] = float(max(peak_changes))
        out["clean_rms_diff_uV"] = rms_changes
        x3 = self.x[config.TRANSIENT_RECORD]
        reference = flt.ideal_bandpass(x3, fs)
        ring = self.mask(self.windows.ringing)
        for key, sos in (("ring_uV_butter", self.sos_butter),
                         ("ring_uV_butter_notch", self.sos_butter_notch)):
            out[key] = float(np.max(np.abs(self.zp(sos, x3) - reference)[ring]))
        return out

    # ------------------------------------------------------------------ performance
    def ringing(self, sos):
        """Largest deviation from the ideal band-pass just before and after the DS3 pulse."""
        x3 = self.x[config.TRANSIENT_RECORD]
        error = np.abs(self.zp(sos, x3) - flt.ideal_bandpass(x3, self.fs))
        return {"pre_ring_uV_zp": float(np.max(error[self.mask(self.windows.pre_ringing)])),
                "post_ring_uV_zp": float(np.max(error[self.mask(self.windows.post_ringing)]))}

    def performance(self):
        fs = self.fs
        filters = {FILTER_NAMES["butter"]: self.sos_butter, FILTER_NAMES["ellip"]: self.sos_ellip,
                   FILTER_NAMES["butter_notch"]: self.sos_butter_notch}
        x3 = self.x[config.TRANSIENT_RECORD]
        ref3 = flt.ideal_bandpass(x3, fs)
        windows = {"ring_uV_zp": self.windows.ringing, "pre_ring_uV_zp": self.windows.pre_ringing,
                   "post_ring_uV_zp": self.windows.post_ringing}
        delay_band = np.linspace(*config.GROUP_DELAY_BAND, 400)
        out = {}
        for name, sos in filters.items():
            entry = {}
            for i in self.ids:
                x = self.x[i]
                y = self.zp(sos, x)
                ref = flt.ideal_bandpass(x, fs)
                spec_y = self.spectrum(y, config.PEAK_NFFT)
                spec_x = self.spectrum(x, config.PEAK_NFFT)
                entry[i] = {
                    "band_change_dB_zp": {k: float(10 * np.log10(self.band_power(y, *v)
                                                                 / self.band_power(x, *v)))
                                          for k, v in config.BANDS.items()},
                    "peak_change_dB_zp": {
                        f: float(20 * np.log10(amplitude_near(*spec_y, f, 0.05)
                                               / amplitude_near(*spec_x, f, 0.05)))
                        for f, _ in self.main_peaks[i]},
                    "corr_ref": float(np.corrcoef(y, ref)[0, 1]),
                    "nrmse_ref": float(np.sqrt(np.mean((y - ref) ** 2)) / ref.std()),
                    "hf_atten_dB_zp": float(10 * np.log10(self.band_power(y, *self.high_band)
                                                          / self.band_power(x, *self.high_band))),
                    "lf_atten_dB_zp": float(10 * np.log10(
                        self.band_power(y, 0, config.LF_RESIDUE_HZ)
                        / self.band_power(x, 0, config.LF_RESIDUE_HZ))),
                }
            error = np.abs(self.zp(sos, x3) - ref3)
            for key, span in windows.items():
                entry[key] = float(np.max(error[self.mask(span)]))
            entry["sections"] = len(sos)
            padded_length = self.n + 2 * int(config.PAD_SECONDS * fs)
            entry["zero_phase_work_ratio"] = 2 * padded_length / self.n
            entry["mults_per_sample_single_pass"] = int(len(sos) * 5)
            entry["mults_per_second_causal_at_fs"] = int(len(sos) * 5 * fs)
            delay = flt.group_delay(sos, delay_band, fs)
            entry["gd_min_ms_causal"] = float(delay.min() / fs * 1e3)
            entry["gd_max_ms_causal"] = float(delay.max() / fs * 1e3)
            probes = np.array(config.GROUP_DELAY_PROBES_HZ)
            entry["gd_at_ms_causal"] = {
                f"{f:g}": float(d / fs * 1e3)
                for f, d in zip(probes, flt.group_delay(sos, probes, fs))}
            out[name] = entry
        out[FILTER_NAMES["butter_notch"]]["max_pole_radius"] = flt.max_pole_radius(
            self.sos_butter_notch)
        return out

    def stopband_summary(self):
        return {"ellip4_att50_zp": flt.zero_phase_db_at(self.sos_ellip, self.fs, 50),
                "butter4_att50_zp": flt.zero_phase_db_at(self.sos_butter, self.fs, 50),
                "butter4_notch_att50_zp": flt.zero_phase_db_at(self.sos_butter_notch, self.fs,
                                                               50)}

