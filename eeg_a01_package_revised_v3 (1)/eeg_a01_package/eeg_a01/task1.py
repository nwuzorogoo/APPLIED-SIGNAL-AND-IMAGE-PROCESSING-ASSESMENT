"""Task 1: FFT check, signal characteristics and aliasing."""
from functools import cached_property

import numpy as np
from scipy import signal

from . import config
from .spectral import alias_frequency, amplitude_near, amplitude_spectrum, fft_radix2


class Task1Signals:
    """Signal representation, resampling and the FFT. Mixed into Study."""

    # ------------------------------------------------------------------ FFT check
    def fft_validation(self):
        test = np.random.default_rng(0).standard_normal(4096)
        fft_error = np.max(np.abs(fft_radix2(test) - np.fft.fft(test)))
        return {"fft_max_abs_error_vs_numpy": float(fft_error)}

    # ------------------------------------------------------------------ Task 1a
    def characteristics(self):
        out = {}
        for i in self.ids:
            x, cal = self.x[i], self.records[i].calibration
            total = self.band_power(x, 0, self.nyquist)
            centred = x - x.mean()
            out[i] = {
                "fs": self.fs, "n": x.size, "dur": x.size / self.fs, "nyq": self.nyquist,
                "df": self.fs / x.size,
                "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
                "std": float(x.std()), "ptp": float(np.ptp(x)),
                "headroom_pct": float(np.abs(x).max() / cal.full_scale * 100),
                "lsb_uV": cal.lsb,
                "sqnr_dB": float(10 * np.log10(x.var() / (cal.lsb ** 2 / 12))),
                "kurtosis": float((centred ** 4).mean() / x.var() ** 2),
                "pow_above_45_pct": float(self.band_power(x, *self.high_band) / total * 100),
                "pow_below_05_pct": float(self.band_power(x, 0, config.SUB_DELTA_HZ)
                                          / total * 100),
                "rel_band": {k: float(self.band_power(x, *v) / total * 100)
                             for k, v in config.BANDS.items()},
            }
        return out

    # ------------------------------------------------------------------ Task 1b
    def decimated(self, factor):
        x = self.x[config.ALIAS_RECORD]
        fs_new = self.fs / factor
        naive = x[::factor]
        filtered = signal.decimate(x, factor, ftype="fir", zero_phase=True)
        return fs_new, naive, filtered

    @cached_property
    def alias_probe(self):
        """The alias record's beta line and where it folds at the demonstration rate."""
        beta = self.probe(config.ALIAS_RECORD, "beta")
        return {"beta": beta,
                "folded": alias_frequency(beta, self.fs / config.ALIAS_DEMO_FACTOR)}

    def aliasing(self):
        x = self.x[config.ALIAS_RECORD]
        beta = self.probe(config.ALIAS_RECORD, "beta")
        delta = self.probe(config.ALIAS_RECORD, "delta")
        alpha = self.probe(config.ALIAS_RECORD, "alpha")
        table = {i: [{"f": f, "M": m, "fs_new": self.fs / m,
                      "alias": round(alias_frequency(f, self.fs / m), 2),
                      "aliased": bool(f > self.fs / m / 2)}
                     for f, _ in self.main_peaks[i] for m in config.ALIAS_FACTORS]
                 for i in self.ids}
        f0, a0 = self.spectrum(x, config.PLOT_NFFT)
        tol = config.ALIAS_TOLERANCE_HZ
        demos = {}
        for factor in (config.ALIAS_DEMO_FACTOR, config.ALIAS_SECOND_FACTOR):
            fs_new, naive, filtered = self.decimated(factor)
            demos[factor] = (fs_new, amplitude_spectrum(naive, fs_new, config.ALIAS_NFFT),
                             amplitude_spectrum(filtered, fs_new, config.ALIAS_NFFT))
        fs8, naive8, aa8 = demos[config.ALIAS_DEMO_FACTOR]
        folded = alias_frequency(beta, fs8)
        demo = {
            "fs_new": fs8, "nyq_new": fs8 / 2,
            "naive_amp_at_2p6": amplitude_near(*naive8, folded, tol),
            "aa_amp_at_2p6": amplitude_near(*aa8, folded, tol),
            "orig_amp_29p4": amplitude_near(f0, a0, beta),
            "orig_amp_2p77": amplitude_near(f0, a0, delta),
            "naive_amp_11p67": amplitude_near(*naive8, alpha),
            "aa_amp_11p67": amplitude_near(*aa8, alpha),
        }
        fs5, naive5, aa5 = demos[config.ALIAS_SECOND_FACTOR]
        folded5 = alias_frequency(beta, fs5)
        demo5 = {"fs_new": fs5, "alias_f": folded5,
                 "naive": amplitude_near(*naive5, folded5), "aa": amplitude_near(*aa5, folded5)}
        return {"alias_table": table, "alias_demo": demo, "alias_demo5": demo5}

