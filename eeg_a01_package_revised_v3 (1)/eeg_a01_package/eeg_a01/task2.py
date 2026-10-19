"""Task 2: convolution, time-frequency localisation, pulse timing and cost."""
from functools import cached_property

import numpy as np
from scipy import signal

from . import config
from . import convolution as conv
from .events import detect_theta_burst, edge_times, interpolate_pulses, window_mask
from .spectral import half_maximum_span, line_width
from .timefreq import dwt_band_edges, operation_counts, stft_magnitude
from .names import freq_key


class Task2TimeFrequency:
    """Convolution kernels and the FFT, STFT, CWT and DWT comparison. Mixed into Study."""

    # ------------------------------------------------------------------ Task 2a
    def kernels(self):
        sigmas = [conv.gaussian_sigma_for_cutoff(c) for c in config.GAUSS_CUTOFFS_HZ]
        kernels = {}
        for cutoff, sigma in zip(config.GAUSS_CUTOFFS_HZ, sigmas):
            kernels[f"gauss{cutoff:g}"] = (conv.gaussian_kernel(sigma, self.fs,
                                                                config.GAUSS_SUPPORT_SIGMAS),
                                           sigma)
        kernels[f"ma{config.MOVING_AVERAGE_TAPS}"] = (
            conv.moving_average_kernel(config.MOVING_AVERAGE_TAPS), None)
        return kernels

    def pulse_measures(self, z):
        """Peak, steepest slope and rise time of the reference pulse in signal z."""
        start, end = self.windows.rise_samples()
        return {
            "peak": float(z[self.mask(self.windows.peak)].max()),
            "slope": float(np.max(np.diff(z[self.mask(self.windows.slope)])) * self.fs / 1e3),
            "rise": conv.rise_time_ms(z, self.fs, start, end),
        }

    def convolution(self):
        rec = config.TRANSIENT_RECORD
        x1, x3 = self.x[config.ALIAS_RECORD], self.x[rec]
        probes = {"att_29p4": self.probe(config.ALIAS_RECORD, "beta"),
                  "att_18": self.probe(rec, "beta"),
                  "att_10p6": self.probe(rec, "alpha"),
                  "att_50": config.MAINS_HZ}
        kernels = self.kernels()
        g40 = kernels["gauss40"][0]
        check = conv.convolve_same(x1, g40) - np.convolve(x1, g40, "same")
        out = {"conv_check_err": float(np.max(np.abs(check)))}
        raw = self.pulse_measures(x3)
        out.update(raw_rise_ms=raw["rise"], raw_peak=raw["peak"], raw_slope=raw["slope"])
        for name, (h, sigma) in kernels.items():
            w, resp = signal.freqz(h, worN=8192, fs=self.fs)
            mag = np.abs(resp)

            def att(f0):
                return float(20 * np.log10(mag[np.argmin(np.abs(w - f0))] + 1e-12))

            y = conv.convolve_same(x3, h)
            nulls = mag < 1e-3
            measures = self.pulse_measures(y)
            hf_ratio = self.band_power(y, *self.high_band) / self.band_power(x3, *self.high_band)
            out[name] = {
                "taps": len(h),
                "sigma_ms": None if sigma is None else sigma * 1e3,
                **{key: att(f) for key, f in probes.items()},
                "f3dB": float(w[np.argmin(np.abs(mag - 1 / np.sqrt(2)))]),
                "first_null": float(w[np.argmax(nulls)]) if nulls.any() else None,
                "rms_ratio": float(y.std() / x3.std()),
                "hf_power_red_dB": float(10 * np.log10(hf_ratio)),
                "pulse_peak_uV": measures["peak"],
                "max_slope_uV_per_ms": measures["slope"],
                "pulse_rise_ms": measures["rise"],
                "edge_samples": conv.edge_samples(h),
                "edge_ms": conv.edge_samples(h) / self.fs * 1e3,
            }
        return out

    # ------------------------------------------------------------------ Task 2b
    def complexity(self):
        """Predicted operation counts. Measured runtimes live in benchmark.run_benchmarks."""
        units = operation_counts(self.n, self.cwt.support)
        return {"units_N2560": units,
                "units_rel_FFT": {k: v / units["FFT"] for k, v in units.items()}}

    @cached_property
    def cwt_map(self):
        """|CWT| of the transient record and the frequency of each row."""
        return self.cwt.magnitude(self.x[config.TRANSIENT_RECORD])

    def burst_spans(self, x):
        """Theta-burst half-maximum span from each time-frequency method on signal x."""
        theta_lo, theta_hi = config.THETA_BURST_BAND
        spans = {}
        for nper in config.STFT_WINDOWS:
            f, ts, z = stft_magnitude(x, self.fs, nper)
            rows = (f >= theta_lo) & (f <= theta_hi)
            spans[f"stft{nper}_burst"] = half_maximum_span(ts, z[rows].mean(0))
        w, fr = self.cwt.magnitude(x)
        rows = (fr >= theta_lo) & (fr <= theta_hi)
        spans["cwt_burst"] = half_maximum_span(self.time, w[rows].mean(0))
        return spans

    def localisation(self):
        rec = config.TRANSIENT_RECORD
        x3 = self.x[rec]
        t = self.time
        pulses = self.pulses[rec]
        # burst durations are measured with the pulses subtracted, for the reference and for
        # every method alike; the raw-signal and interpolated versions show the pulse effect
        loc = {"ref_burst": self.bursts[rec], "ref_burst_raw": self.bursts_raw[rec],
               "ref_burst_interpolated": detect_theta_burst(interpolate_pulses(x3, pulses),
                                                            self.fs),
               "pulses_overlapping_burst": [
                   [p.onset_s, p.offset_s] for p in pulses
                   if p.offset_s > self.bursts_raw[rec][0] - 0.2
                   and p.onset_s < self.bursts_raw[rec][1] + 0.2]}
        loc.update(self.burst_spans(self.x_clean[rec]))
        loc.update({f"{k}_raw": v for k, v in self.burst_spans(x3).items()})
        for nper in config.STFT_WINDOWS:
            f, ts, z = stft_magnitude(x3, self.fs, nper)
            rows = (f >= config.PULSE_STFT_BAND[0]) & (f <= config.PULSE_STFT_BAND[1])
            cols = window_mask(ts, *self.windows.localisation)
            loc[f"stft{nper}_pulse"] = half_maximum_span(ts[cols], z[rows].mean(0)[cols])
            loc[f"stft{nper}_df"] = self.fs / nper
            loc[f"stft{nper}_dt"] = nper / self.fs
        w, fr = self.cwt_map
        rows = (fr >= config.PULSE_CWT_BAND[0]) & (fr <= config.PULSE_CWT_BAND[1])
        envelope = w[rows].mean(0)
        cols = self.mask(self.windows.localisation)
        loc["cwt_pulse"] = half_maximum_span(t[cols], envelope[cols])
        cols = self.mask(self.windows.falling_edge)
        loc["cwt_pulse_falling_edge"] = half_maximum_span(t[cols], envelope[cols])

        alpha = self.probe(config.TRANSIENT_RECORD, "alpha")
        key = freq_key(alpha)
        f, a = self.spectrum(x3, config.PEAK_NFFT)
        loc[f"fft_linewidth_{key}"] = line_width(f, a, alpha)
        for nper in config.STFT_WINDOWS:
            f, _, z = stft_magnitude(x3, self.fs, nper)
            loc[f"stft{nper}_linewidth_{key}"] = line_width(f, z.mean(1), alpha)
        order = np.argsort(fr)
        loc[f"cwt_linewidth_{key}"] = line_width(fr[order], w.mean(1)[order], alpha)
        burst_freq = float(np.mean(config.THETA_BURST_BAND))
        loc["cwt_coi_s"] = {f"{freq:g}": float(self.cwt.cone_of_influence_s(freq))
                            for freq in (config.CWT_FREQS[0], burst_freq,
                                         *config.CWT_SIGMA_PROBES_HZ)}
        loc["cwt_burst_clear_of_coi"] = self.cwt.inside_cone(burst_freq, *self.bursts[rec][:2],
                                                             self.duration)
        loc["cwt_pulses_clear_of_coi"] = {
            f"DS{i}_{p.onset_s:.3f}s": self.cwt.inside_cone(config.PULSE_CWT_BAND[0], p.onset_s,
                                                           p.offset_s, self.duration)
            for i in self.ids for p in self.pulses[i]}
        loc["cwt_area_inside_coi_frac"] = float(np.mean(
            [(self.time >= c) & (self.time <= self.duration - c)
             for c in self.cwt.cone_of_influence_s(fr)]))
        for freq in config.CWT_SIGMA_PROBES_HZ:
            loc[f"cwt_sigma_t_ms_{freq:g}"] = self.cwt.sigma_t_ms(freq)
            loc[f"cwt_sigma_f_Hz_{freq:g}"] = self.cwt.sigma_f_hz(freq)
        return loc

    def pulse_envelopes(self, record_id):
        """High-frequency band envelope of each time-frequency method, as (time axis, envelope)."""
        x = self.x[record_id]
        envelopes = {}
        for nper in config.STFT_WINDOWS:
            f, ts, z = stft_magnitude(x, self.fs, nper)
            rows = (f >= config.PULSE_STFT_BAND[0]) & (f <= config.PULSE_STFT_BAND[1])
            envelopes[f"stft{nper}"] = (ts, z[rows].mean(0))
        w, fr = (self.cwt_map if record_id == config.TRANSIENT_RECORD
                 else self.cwt.magnitude(x))
        rows = (fr >= config.PULSE_CWT_BAND[0]) & (fr <= config.PULSE_CWT_BAND[1])
        envelopes["cwt"] = (self.time, w[rows].mean(0))
        return envelopes

    def pulse_timing(self):
        """Onset and offset error of each method against the detected pulse edges.

        Each method estimates the edges as the first and last prominent peak of its
        high-frequency envelope near the pulse. A method that shows one blob instead of two
        edges returns the same time for both, and its errors grow to about half the pulse.
        """
        per_pulse = {}
        errors = {}
        for i in self.ids:
            envelopes = self.pulse_envelopes(i)
            for pulse in self.pulses[i]:
                label = f"DS{i}_{pulse.onset_s:.3f}s"
                entry = {"onset_s": pulse.onset_s, "offset_s": pulse.offset_s}
                for method, (axis, env) in envelopes.items():
                    onset, offset, n_peaks = edge_times(axis, env, pulse)
                    entry[method] = {"onset_err_ms": (onset - pulse.onset_s) * 1e3,
                                     "offset_err_ms": (offset - pulse.offset_s) * 1e3,
                                     "edges_resolved": n_peaks >= 2}
                    errors.setdefault(method, []).append(entry[method])
                per_pulse[label] = entry
        summary = {}
        for method, rows in errors.items():
            abs_err = np.abs([[r["onset_err_ms"], r["offset_err_ms"]] for r in rows])
            resolved = np.array([r["edges_resolved"] for r in rows])
            summary[method] = {
                "mean_abs_err_ms": float(abs_err.mean()),
                "max_abs_err_ms": float(abs_err.max()),
                "resolved_mean_abs_err_ms": (float(abs_err[resolved].mean())
                                             if resolved.any() else None),
                "pulses_resolved": int(resolved.sum()), "pulses": len(rows)}
        return {"per_pulse": per_pulse, "summary": summary}

    def dwt(self):
        return dwt_band_edges(self.fs)

