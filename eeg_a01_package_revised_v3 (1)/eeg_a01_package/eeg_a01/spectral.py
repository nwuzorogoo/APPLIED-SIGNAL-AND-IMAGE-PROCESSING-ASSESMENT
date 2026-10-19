"""Custom FFT and the spectral measurements built on it."""
import numpy as np
from scipy import signal


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def fft_radix2(x):
    """Iterative radix-2 decimation-in-time Cooley-Tukey FFT.

    The input length must be a power of two. Samples are put in bit-reversed order, then
    log2(n) butterfly stages combine transforms of length 2, 4, ..., n.
    """
    x = np.asarray(x, dtype=complex)
    n = x.size
    if x.ndim != 1 or not is_power_of_two(n):
        raise ValueError(f"fft_radix2 needs a 1-D input whose length is a power of two, got {n}")
    stages = n.bit_length() - 1
    index = np.arange(n)
    reversed_index = np.zeros(n, dtype=int)
    for bit in range(stages):
        reversed_index |= ((index >> bit) & 1) << (stages - 1 - bit)
    spectrum = x[reversed_index].copy()
    size = 2
    while size <= n:
        half = size // 2
        twiddle = np.exp(-2j * np.pi * np.arange(half) / size)
        blocks = spectrum.reshape(-1, size)
        top = blocks[:, :half].copy()
        bottom = blocks[:, half:] * twiddle
        blocks[:, :half] = top + bottom
        blocks[:, half:] = top - bottom
        spectrum = blocks.reshape(-1)
        size *= 2
    return spectrum


def amplitude_spectrum(x, fs, nfft=None, window=True):
    """Single-sided amplitude spectrum, mean removed, Hann-windowed and zero-padded.

    Amplitudes are scaled by the window sum so a sinusoid of amplitude A reads as A.
    Returns (frequencies in Hz, amplitudes in the units of x).
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("amplitude_spectrum needs at least two samples")
    if nfft is None:
        nfft = 1 << (x.size - 1).bit_length()
    if nfft < x.size:
        raise ValueError(f"nfft={nfft} is shorter than the signal ({x.size} samples)")
    if not is_power_of_two(nfft):
        raise ValueError(f"nfft={nfft} must be a power of two for the radix-2 FFT")
    taper = np.hanning(x.size) if window else np.ones(x.size)
    padded = np.zeros(nfft)
    padded[:x.size] = (x - x.mean()) * taper
    half_spectrum = fft_radix2(padded)[:nfft // 2 + 1]
    amplitude = 2 * np.abs(half_spectrum) / taper.sum()
    freqs = np.arange(nfft // 2 + 1) * fs / nfft
    return freqs, amplitude


def amplitude_near(freqs, amplitude, f0, tolerance=0.3):
    """Largest amplitude within +-tolerance Hz of f0."""
    mask = (freqs > f0 - tolerance) & (freqs < f0 + tolerance)
    if not mask.any():
        raise ValueError(f"no spectral bins within {tolerance} Hz of {f0} Hz")
    return float(amplitude[mask].max())


def band_power(x, fs, low, high):
    """Power of x in [low, high) Hz from a Hann periodogram."""
    freqs, psd = signal.periodogram(x - np.mean(x), fs, window="hann")
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        raise ValueError(f"band {low}-{high} Hz contains no periodogram bins")
    return np.trapezoid(psd[mask], freqs[mask])


def spectral_peaks(x, fs, nfft, min_amplitude, min_spacing_hz, max_freq):
    """Local maxima of the amplitude spectrum below max_freq as (Hz, amplitude) pairs."""
    freqs, amplitude = amplitude_spectrum(x, fs, nfft=nfft)
    spacing = int(min_spacing_hz / freqs[1])
    indices, _ = signal.find_peaks(amplitude, height=min_amplitude, distance=spacing)
    return [(round(float(freqs[k]), 2), round(float(amplitude[k]), 1))
            for k in indices if freqs[k] < max_freq]


def dominant_peak(peaks, band):
    """Frequency of the highest-amplitude peak inside band = (low, high)."""
    low, high = band
    in_band = [(amp, freq) for freq, amp in peaks if low <= freq < high]
    if not in_band:
        raise ValueError(f"no spectral peak in {low}-{high} Hz")
    return max(in_band)[1]


def alias_frequency(f0, fs_new):
    """Apparent frequency of a tone at f0 after sampling at fs_new: |f0 - k fs_new|."""
    if fs_new <= 0:
        raise ValueError("sampling rate must be positive")
    return abs(f0 - fs_new * round(f0 / fs_new))


def half_maximum_span(axis, envelope):
    """Start, end and width of the contiguous region around the maximum where the envelope
    stays at or above half its peak (full width at half maximum)."""
    envelope = np.asarray(envelope, dtype=float)
    if envelope.size == 0:
        raise ValueError("empty envelope")
    level = envelope / envelope.max()
    left = right = int(np.argmax(level))
    while left > 0 and level[left - 1] >= 0.5:
        left -= 1
    while right < level.size - 1 and level[right + 1] >= 0.5:
        right += 1
    return float(axis[left]), float(axis[right]), float(axis[right] - axis[left])


def line_width(freqs, spectrum, f0):
    """-3 dB width of the spectral line nearest f0."""
    k = int(np.argmin(np.abs(freqs - f0)))
    level = spectrum / spectrum[k]
    left = right = k
    while left > 0 and level[left - 1] >= 1 / np.sqrt(2):
        left -= 1
    while right < level.size - 1 and level[right + 1] >= 1 / np.sqrt(2):
        right += 1
    return float(freqs[right] - freqs[left])
