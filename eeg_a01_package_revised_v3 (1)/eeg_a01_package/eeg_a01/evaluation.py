"""Stress-test contamination, filter metrics and paired statistics."""
import numpy as np
from scipy import signal, stats

from . import config
from .spectral import amplitude_near, amplitude_spectrum

METRIC_NAMES = ("snr_in", "snr_out", "gain", "gain_inner", "mains", "mains_rel_dB", "corr")


def draw_seed(record_id, draw):
    return config.SEED_STRIDE * record_id + draw


def contaminate(x, fs, rng, mains_hz=config.MAINS_HZ, mains_amp=None,
                spec=config.CONTAMINATION):
    """Add random baseline wander, mains interference and EMG-like noise to x.

    Every contaminant is drawn afresh: drift amplitude, frequency, phase and linear trend;
    mains amplitude (unless mains_amp fixes it) and phase; EMG noise realisation and level.
    The order of random draws is fixed so a given seed always gives the same record.
    """
    x = np.asarray(x, dtype=float)
    time = np.arange(x.size) / fs
    centred_time = time - x.size / fs / 2
    drift_amp = rng.uniform(*spec.drift_amp_uv)
    drift_freq = rng.uniform(*spec.drift_freq_hz)
    drift_phase = rng.uniform(0, 2 * np.pi)
    trend = rng.uniform(*spec.trend_uv_per_s)
    drift = drift_amp * np.sin(2 * np.pi * drift_freq * time + drift_phase) + trend * centred_time
    amp = rng.uniform(*spec.mains_amp_uv) if mains_amp is None else mains_amp
    mains = amp * np.sin(2 * np.pi * mains_hz * time + rng.uniform(0, 2 * np.pi))
    emg_sos = signal.butter(4, list(spec.emg_band_hz), "bandpass", fs=fs, output="sos")
    noise = rng.standard_normal(x.size)
    emg = signal.sosfiltfilt(emg_sos, noise) * rng.uniform(*spec.emg_scale_uv)
    return x + drift + mains + emg


def snr_db(reference, estimate, mask=None):
    """Reference power over error power in dB, optionally on a sub-window."""
    error = estimate - reference
    if mask is not None:
        reference, error = reference[mask], error[mask]
    return 10 * np.log10(np.mean(reference ** 2) / np.mean(error ** 2))


def mains_amplitude(x, fs):
    """Amplitude of the mains line in x."""
    return amplitude_near(*amplitude_spectrum(x, fs, nfft=config.PLOT_NFFT), config.MAINS_HZ, 0.3)


def stress_metrics(contaminated, filtered, reference, fs, inner_mask, mains_in=None):
    """Output quality of one filter on one contaminated draw.

    mains_in, the mains amplitude of the contaminated input, can be passed in when several
    filters are scored on the same draw so its spectrum is computed only once.
    """
    snr_in = snr_db(reference, contaminated)
    snr_out = snr_db(reference, filtered)
    inner_in = np.mean((contaminated - reference)[inner_mask] ** 2)
    inner_out = np.mean((filtered - reference)[inner_mask] ** 2)
    if mains_in is None:
        mains_in = mains_amplitude(contaminated, fs)
    mains_out = mains_amplitude(filtered, fs)
    return {
        "snr_in": snr_in,
        "snr_out": snr_out,
        "gain": snr_out - snr_in,
        "gain_inner": 10 * np.log10(inner_in / inner_out),
        "mains": mains_out,
        "mains_rel_dB": 20 * np.log10(mains_out / mains_in),
        "corr": np.corrcoef(filtered, reference)[0, 1],
    }


def summarise(values):
    """Mean, sample SD, minimum and maximum."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        raise ValueError("need at least two values to summarise")
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max())}


def paired_test(differences, confidence=0.95):
    """Mean paired difference with its t confidence interval, t and Wilcoxon tests and dz."""
    d = np.asarray(differences, dtype=float)
    n = d.size
    if n < 3:
        raise ValueError("a paired test needs at least three pairs")
    se = d.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.5 + confidence / 2, n - 1)
    t_test = stats.ttest_rel(d, np.zeros(n))
    wilcoxon = stats.wilcoxon(d)
    return {
        "mean": float(d.mean()),
        "ci95": [float(d.mean() - t_crit * se), float(d.mean() + t_crit * se)],
        "t": float(t_test.statistic),
        "p_t": float(t_test.pvalue),
        "p_wilcoxon": float(wilcoxon.pvalue),
        "dz": float(d.mean() / d.std(ddof=1)),
        "n": int(n),
    }
