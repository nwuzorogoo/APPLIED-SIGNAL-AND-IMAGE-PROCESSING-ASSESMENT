"""Task 3 stress test: randomised contamination scored against the ideal band-pass.

One loop, score_draws, contaminates each record with the seeded draws and scores every filter
it is given on the same draw. The main Butterworth, elliptic and notch comparison and the
elliptic ripple study both call it, so they are paired on identical contamination by
construction rather than by keeping two loops in step.
"""
import numpy as np

from . import config
from . import evaluation as ev
from . import filters as flt
from .names import STRESS_NAMES
from .spectral import amplitude_near


class StressTest:
    """Monte Carlo stress test and the mains checks. Mixed into Study."""

    def stress_filters(self):
        return {
            "Butterworth": lambda z: self.zp(self.sos_butter, z),
            "Elliptic": lambda z: self.zp(self.sos_ellip, z),
            "Butterworth+notch": lambda z: self.enhance(z)[0],
        }

    def score_draws(self, record_id, filters, observe=None):
        """Score every filter on the same n_draws contaminated copies of one record.

        filters maps a name to a function from contaminated to filtered signal. observe, if
        given, is called once per draw on the contaminated signal (the mains detector uses
        it). Returns ({name: {metric: [values per draw]}}, [observations per draw]).
        """
        fs = self.fs
        edge = config.EDGE_EXCLUSION_S
        inner = (self.time >= edge) & (self.time < self.duration - edge)
        ref = flt.ideal_bandpass(self.x[record_id], fs)
        raw = {name: {k: [] for k in ev.METRIC_NAMES} for name in filters}
        observations = []
        for d in range(self.n_draws):
            xc = ev.contaminate(self.x[record_id], fs,
                                np.random.default_rng(ev.draw_seed(record_id, d)))
            if observe is not None:
                observations.append(observe(xc))
            mains_in = ev.mains_amplitude(xc, fs)          # shared by every filter on this draw
            for name, apply in filters.items():
                metrics = ev.stress_metrics(xc, apply(xc), ref, fs, inner, mains_in)
                for key, value in metrics.items():
                    raw[name][key].append(float(value))
        return raw, observations

    def stress_test(self):
        """Butterworth, elliptic and Butterworth plus notch, paired across filters."""
        filters = self.stress_filters()
        raw, detected = {}, {}
        for i in self.ids:
            raw[i], hits = self.score_draws(i, filters,
                                            observe=lambda xc: flt.mains_present(xc, self.fs)[0])
            detected[i] = int(sum(hits))
        self.mc_raw = raw
        out = {"n_draws": self.n_draws, "notch_detected": detected}
        for i in self.ids:
            gain_b, gain_e, gain_bn = (np.array(raw[i][name]["gain"]) for name in STRESS_NAMES)
            out[i] = {name: {k: ev.summarise(v) for k, v in raw[i][name].items()}
                      for name in filters}
            out[i]["paired_B_minus_E"] = ev.summarise(gain_b - gain_e)
            out[i]["B_wins_frac"] = float(np.mean(gain_b > gain_e))
            out[i]["paired_BN_minus_B"] = ev.summarise(gain_bn - gain_b)
            out[i]["BN_wins_vs_E_frac"] = float(np.mean(gain_bn > gain_e))
            out[i]["BN_wins_vs_B_frac"] = float(np.mean(gain_bn > gain_b))
            for key, diff in (("B_minus_E", gain_b - gain_e), ("BN_minus_B", gain_bn - gain_b),
                              ("BN_minus_E", gain_bn - gain_e)):
                out[i][f"test_{key}"] = ev.paired_test(diff)
        out["mains_offset"] = self.mains_offset_test()
        out["detect_weak_2uV"] = self.weak_mains_detection()
        return out

    def ripple_stress(self):
        """Stress-test SNR gain of the elliptic design at each passband ripple.

        Runs through score_draws, so every ripple is paired with the Butterworth on the same
        contamination as the main stress test.
        """
        designs = {f"{r:g}": flt.ellip_bandpass(config.DESIGN_ORDER, self.fs, ripple_db=r)
                   for r in config.ELLIP_RIPPLES_DB}
        filters = {"butter": lambda z: self.zp(self.sos_butter, z),
                   **{key: (lambda z, sos=sos: self.zp(sos, z)) for key, sos in designs.items()}}
        out = {}
        for i in self.ids:
            raw, _ = self.score_draws(i, filters)
            butter = np.array(raw["butter"]["gain"])
            out[i] = {}
            for key in designs:
                gain = np.array(raw[key]["gain"])
                out[i][key] = {"gain": ev.summarise(gain),
                               "butter_minus_ellip": ev.summarise(butter - gain),
                               "test": ev.paired_test(butter - gain),
                               "butter_wins_frac": float(np.mean(butter > gain))}
        return out

    def mains_offset_test(self):
        """Residual mains after the notch when the line drifts off 50 Hz."""
        x3 = self.x[config.TRANSIENT_RECORD]
        out = {}
        for mains_hz in config.MAINS_OFFSETS_HZ:
            residuals = []
            for d in range(config.OFFSET_DRAWS):
                xc = ev.contaminate(x3, self.fs, np.random.default_rng(d), mains_hz=mains_hz,
                                    mains_amp=config.OFFSET_MAINS_UV)
                y = self.zp(self.sos_butter_notch, xc)
                residuals.append(amplitude_near(*self.spectrum(y, config.PLOT_NFFT), mains_hz, 0.3))
            out[mains_hz] = ev.summarise(residuals)
        return out

    def weak_mains_detection(self):
        """Fraction of draws in which a weak mains line is still detected."""
        out = {}
        for i in self.ids:
            hits = [flt.mains_present(ev.contaminate(self.x[i], self.fs, np.random.default_rng(d),
                                                     mains_amp=config.WEAK_MAINS_UV), self.fs)[0]
                    for d in range(config.OFFSET_DRAWS)]
            out[i] = float(np.mean(hits))
        return out
