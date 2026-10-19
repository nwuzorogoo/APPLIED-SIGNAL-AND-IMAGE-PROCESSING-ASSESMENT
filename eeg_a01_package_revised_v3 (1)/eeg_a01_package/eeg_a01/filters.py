"""IIR band-pass and notch design, zero-phase application and response measures.

Reporting convention: every IIR filter is applied forward and backward (sosfiltfilt), so every
magnitude in dB is zero-phase, |H_zp(f)| = |H(f)|^2, twice the single-pass figure. Group delay
is the exception, since zero-phase delay is identically zero; it is reported for causal
single-pass use and labelled as such.
"""
import numpy as np
from scipy import signal

from . import config
from .spectral import amplitude_spectrum

ZP = config.ZERO_PHASE_FACTOR


def butter_bandpass(order, fs, low=config.PASSBAND[0], high=config.PASSBAND[1]):
    return signal.butter(order, [low, high], "bandpass", fs=fs, output="sos")


def ellip_bandpass(order, fs, low=config.PASSBAND[0], high=config.PASSBAND[1],
                   ripple_db=config.ELLIP_RIPPLE_DB, stop_db=config.ELLIP_STOP_DB):
    return signal.ellip(order, ripple_db, stop_db, [low, high], "bandpass", fs=fs, output="sos")


def notch(fs, f0=config.MAINS_HZ, q=config.NOTCH_Q):
    """Second-order IIR notch at f0 with quality factor q, as one SOS section."""
    b, a = signal.iirnotch(f0, q, fs=fs)
    return signal.tf2sos(b, a)


def cascade(*sections):
    """Join SOS filters into a single cascade."""
    return np.vstack(sections)


def zero_phase(sos, x, fs):
    """Forward-backward filtering with one odd-padding rule for every call."""
    return signal.sosfiltfilt(sos, x, padtype="odd", padlen=int(config.PAD_SECONDS * fs))


def frequency_response(sos, fs, n_points=2 ** 15):
    return signal.sosfreqz(sos, worN=n_points, fs=fs)


def to_zero_phase_db(h):
    """Zero-phase magnitude in dB from a single-pass complex response."""
    return ZP * 20 * np.log10(np.abs(h) + 1e-15)


def zero_phase_db_at(sos, fs, f0):
    """Zero-phase magnitude in dB at one frequency."""
    _, h = signal.sosfreqz(sos, worN=[f0], fs=fs)
    return float(to_zero_phase_db(h[0]))


def group_delay(sos, freqs, fs):
    """Single-pass (causal) group delay in samples, summed over sections."""
    delay = np.zeros_like(freqs, dtype=float)
    for section in sos:
        _, section_delay = signal.group_delay((section[:3], section[3:]), w=freqs, fs=fs)
        delay += section_delay
    return delay


def stopband_edge_hz(sos, fs, stop_db, start_hz=config.PASSBAND[1], n_points=2 ** 15):
    """First frequency above start_hz where the single-pass attenuation reaches stop_db."""
    freqs = np.linspace(start_hz, fs / 2, n_points)
    _, h = signal.sosfreqz(sos, worN=freqs, fs=fs)
    reached = 20 * np.log10(np.abs(h) + 1e-15) <= -stop_db
    return float(freqs[np.argmax(reached)]) if reached.any() else None


def transmission_zeros_hz(sos, fs, above_hz=config.PASSBAND[1]):
    """Frequencies of the zeros on the unit circle above above_hz (elliptic and Chebyshev II)."""
    zeros = signal.sos2zpk(sos)[0]
    on_circle = np.abs(np.abs(zeros) - 1) < 1e-6
    freqs = np.abs(np.angle(zeros[on_circle])) * fs / (2 * np.pi)
    return sorted({round(float(f), 2) for f in freqs if above_hz < f < fs / 2})


def max_pole_radius(sos):
    return float(np.max(np.abs(signal.sos2zpk(sos)[1])))


def minimum_orders(fs):
    """Minimum low-pass order of each family for the zero-phase specification.

    Each pass carries half the zero-phase dB, so the per-pass targets are halved.
    """
    args = (config.SPEC_PASS_HZ, config.SPEC_STOP_HZ, config.SPEC_PASS_DB / ZP,
            config.SPEC_STOP_DB / ZP)
    return {
        "butter": signal.buttord(*args, fs=fs)[0],
        "cheby1": signal.cheb1ord(*args, fs=fs)[0],
        "cheby2": signal.cheb2ord(*args, fs=fs)[0],
        "ellip": signal.ellipord(*args, fs=fs)[0],
    }


def minimum_order_lowpass(family, order, fs):
    """Low-pass designs matching minimum_orders, used for the family comparison figure."""
    ripple, stop = config.SPEC_PASS_DB / ZP, config.SPEC_STOP_DB / ZP
    edge = config.SPEC_PASS_HZ
    designs = {
        "butter": lambda: signal.butter(order, edge, fs=fs, output="sos"),
        "cheby1": lambda: signal.cheby1(order, ripple, edge, fs=fs, output="sos"),
        "cheby2": lambda: signal.cheby2(order, stop, config.SPEC_STOP_HZ, fs=fs, output="sos"),
        "ellip": lambda: signal.ellip(order, ripple, stop, edge, fs=fs, output="sos"),
    }
    if family not in designs:
        raise ValueError(f"unknown filter family {family!r}")
    return designs[family]()


def ideal_bandpass(x, fs, low=config.PASSBAND[0], high=config.PASSBAND[1]):
    """Brick-wall band-pass in the DFT domain, the reference output for evaluation."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, len(x))


def mains_line_ratio_db(x, fs, f0=config.MAINS_HZ, nfft=config.PLOT_NFFT):
    """Mains line amplitude over the median of the nearby floor, in dB."""
    freqs, amp = amplitude_spectrum(x, fs, nfft=nfft)
    half = config.DETECTOR_LINE_HALF_WIDTH_HZ
    outer, inner = config.DETECTOR_FLOOR_HZ
    line = amp[(freqs > f0 - half) & (freqs < f0 + half)].max()
    below = (freqs > f0 - outer) & (freqs < f0 - inner)
    above = (freqs > f0 + inner) & (freqs < f0 + outer)
    floor = np.median(amp[below | above])
    return float(20 * np.log10(line / floor))


def mains_present(x, fs, threshold_db=config.DETECTOR_THRESHOLD_DB):
    """Detector: (is the mains line above threshold, line-to-floor ratio in dB)."""
    ratio = mains_line_ratio_db(x, fs)
    return bool(ratio > threshold_db), ratio


class Enhancer:
    """Recommended pipeline: zero-phase band-pass, plus the notch only when mains is detected."""

    def __init__(self, fs, bandpass_sos, notch_sos):
        self.fs = fs
        self.bandpass = bandpass_sos
        self.with_notch = cascade(bandpass_sos, notch_sos)

    def __call__(self, x):
        notch_on, _ = mains_present(x, self.fs)
        sos = self.with_notch if notch_on else self.bandpass
        return zero_phase(sos, x, self.fs), notch_on
