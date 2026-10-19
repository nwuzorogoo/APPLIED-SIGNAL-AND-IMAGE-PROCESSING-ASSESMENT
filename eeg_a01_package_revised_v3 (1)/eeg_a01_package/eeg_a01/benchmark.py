"""Wall-clock timing. Kept out of results.json because runtimes change from run to run."""
import os
import platform
import time

import numpy as np
import pywt
from scipy import signal

from . import config
from .names import FILTER_NAMES
from .spectral import fft_radix2
from .timefreq import stft_magnitude


def time_call(fn, calls=30, blocks=7):
    """Median over `blocks` blocks of `calls` calls, in ms per call (damps scheduler jitter)."""
    if calls < 1 or blocks < 1:
        raise ValueError("calls and blocks must be positive")
    per_call = []
    for _ in range(blocks):
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        per_call.append((time.perf_counter() - start) / calls * 1e3)
    return float(np.median(per_call))


def time_once(fn):
    """Duration of a single call in ms, for operations too slow to repeat."""
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1e3


def scaling_exponent(lengths, runtimes):
    """Slope of log runtime against log length, the empirical exponent p in O(N^p)."""
    return float(np.polyfit(np.log(lengths), np.log(runtimes), 1)[0])


def machine_info():
    """Where the timings were taken, since they only compare within one machine."""
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count()}


def run_benchmarks(study):
    """Every wall-clock measurement, kept apart from the deterministic results.

    The keys mirror results.json so the two files can be merged by path: timing_ms, the
    runtime parts of complexity, and the runtime fields of each perf entry.
    """
    fs, n = study.fs, study.n
    x1 = study.x[study.ids[0]]
    x3 = study.x[config.TRANSIENT_RECORD]
    n_fft = 1 << (n - 1).bit_length()
    padded = np.r_[x1, np.zeros(n_fft - n)]
    sample = np.arange(n)
    dft_bins = 400
    out = {
        "machine": machine_info(),
        "fft_custom_ms": time_call(lambda: fft_radix2(padded), 200, 1),
        "fft_numpy_ms": time_call(lambda: np.fft.fft(x1, n_fft), 200, 1),
        "dft_direct_ms_400bins": time_once(
            lambda: np.exp(-2j * np.pi * np.outer(sample[:dft_bins], sample) / n) @ x1),
    }
    short, long_ = config.STFT_WINDOWS
    timing = {
        "FFT": time_call(lambda: np.fft.rfft(x3)),
        "STFT_0.5s": time_call(lambda: stft_magnitude(x3, fs, short)),
        "STFT_2s": time_call(lambda: stft_magnitude(x3, fs, long_)),
        "CWT": time_call(lambda: study.cwt.magnitude(x3), 10),
        "CWT_direct_conv": time_once(lambda: study.cwt.magnitude(x3, method="conv")),
        "DWT": time_call(lambda: pywt.wavedec(x3, config.DWT_WAVELET, level=config.DWT_LEVEL)),
    }
    out["timing_ms"] = timing

    lengths = [n * 2 ** k for k in range(config.SCALING_LENGTHS)]
    runs = {k: [] for k in ("FFT", f"STFT_{short}", f"STFT_{long_}", "CWT_fft", "DWT")}
    rng = np.random.default_rng(7)
    for length in lengths:
        z = rng.standard_normal(length)
        runs["FFT"].append(time_call(lambda: np.fft.rfft(z), 50))
        runs[f"STFT_{short}"].append(time_call(lambda: stft_magnitude(z, fs, short), 10))
        runs[f"STFT_{long_}"].append(time_call(lambda: stft_magnitude(z, fs, long_), 5))
        runs["CWT_fft"].append(time_call(lambda: study.cwt.magnitude(z), 3))
        runs["DWT"].append(time_call(
            lambda: pywt.wavedec(z, config.DWT_WAVELET, level=config.DWT_LEVEL), 50))
    batched = study.cwt.batched(n)
    batched_ms = time_call(lambda: batched(x3), 10)
    out["complexity"] = {
        "runtime_rel_FFT": {
            "FFT": 1.0,
            f"STFT_{short}": timing["STFT_0.5s"] / timing["FFT"],
            f"STFT_{long_}": timing["STFT_2s"] / timing["FFT"],
            "CWT_fft": timing["CWT"] / timing["FFT"],
            "CWT_direct": timing["CWT_direct_conv"] / timing["FFT"]},
        "scaling_N": lengths,
        "scaling_ms": runs,
        "scaling_exponent": {k: scaling_exponent(lengths, v) for k, v in runs.items()},
        "CWT_batched_ms": batched_ms,
        "CWT_pywt_ms": timing["CWT"],
        "CWT_batched_rel_FFT": batched_ms / timing["FFT"],
    }

    filters = {FILTER_NAMES["butter"]: study.sos_butter, FILTER_NAMES["ellip"]: study.sos_ellip,
               FILTER_NAMES["butter_notch"]: study.sos_butter_notch}
    pad = int(config.PAD_SECONDS * fs)
    x_padded = np.r_[np.zeros(pad), x1, np.zeros(pad)]
    out["perf"] = {}
    for name, sos in filters.items():
        out["perf"][name] = {
            "runtime_ms_zero_phase": time_call(lambda: study.zp(sos, x1), 200, 15),
            "runtime_ms_causal_single_pass": time_call(lambda: signal.sosfilt(sos, x1), 200, 15),
            # the bare arithmetic of zero-phase filtering, two passes over the padded record,
            # separates the cost of padding from sosfiltfilt's fixed set-up overhead
            "runtime_ms_two_padded_passes": time_call(
                lambda: signal.sosfilt(sos, signal.sosfilt(sos, x_padded)[::-1]), 200, 15),
        }
    return out


def merge(results, timings):
    """Overlay timings onto a copy of results by key path, for reports that need both."""
    merged = dict(results)
    for key, value in timings.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged
