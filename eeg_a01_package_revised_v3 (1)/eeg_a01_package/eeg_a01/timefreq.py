"""STFT, continuous and discrete wavelet transforms, and their operation counts."""
import numpy as np
import pywt
from scipy import signal
from scipy.fft import next_fast_len

from . import config


def stft_magnitude(x, fs, window_len, hop=config.STFT_HOP):
    """Hann-window STFT magnitude with even boundary extension. Returns (f, t, |Z|)."""
    if not 0 < hop <= window_len:
        raise ValueError("hop must lie between 1 and the window length")
    freqs, times, z = signal.stft(x, fs, window="hann", nperseg=window_len,
                                  noverlap=window_len - hop, boundary="even", padded=True)
    return freqs, times, np.abs(z)


class MorletCWT:
    """Complex Morlet CWT on a fixed frequency grid, computed with pywt (FFT method).

    The wavelet support at each scale (16 s + 1 samples, as pywt uses) is kept so operation
    counts and the batched implementation use the same lengths.
    """

    def __init__(self, fs, freqs, wavelet=config.CWT_WAVELET,
                 bandwidth=config.CWT_BANDWIDTH):
        self.fs = fs
        self.freqs = np.asarray(freqs, dtype=float)
        self.wavelet = wavelet
        self.bandwidth = bandwidth
        self.scales = pywt.central_frequency(wavelet) * fs / self.freqs
        self.support = (self.scales * 16 + 1).astype(int)

    def magnitude(self, x, method="fft"):
        """|W(f, t)| and the frequency of each row."""
        coeffs, freqs = pywt.cwt(x, self.scales, self.wavelet,
                                 sampling_period=1 / self.fs, method=method)
        return np.abs(coeffs), freqs

    def batched(self, n_samples):
        """Return a function computing the same CWT arithmetic in one vectorised call.

        Wavelet spectra do not depend on the data, so they are computed once here. Comparing
        this with pywt isolates the cost of pywt's per-scale Python loop.
        """
        n_fft = next_fast_len(n_samples + self.support.max() - 1)
        bank = np.empty((self.freqs.size, n_fft), dtype=complex)
        for row, (freq, length) in enumerate(zip(self.freqs, self.support)):
            lag = (np.arange(length) - (length - 1) / 2) / self.fs
            envelope = np.exp(-(lag * freq) ** 2 / self.bandwidth)
            bank[row] = np.fft.fft(envelope * np.exp(2j * np.pi * freq * lag), n_fft)

        def transform(x):
            return np.abs(np.fft.ifft(bank * np.fft.fft(x, n_fft)[None, :], axis=1))

        return transform

    def sigma_t_ms(self, freq):
        """Time spread of the Morlet envelope exp(-t^2 / B) at frequency freq."""
        return float(np.sqrt(self.bandwidth / 2) * config.CWT_CENTRE / freq * 1e3)

    def cone_of_influence_s(self, freq):
        """Edge e-folding time of the CWT at freq, in seconds (Torrence and Compo, 1998).

        Their Morlet has |psi|^2 ~ exp(-t^2 / s^2) and a cone of sqrt(2) s. The pywt complex
        Morlet exp(-t^2 / B) at scale s = C / f has |psi|^2 ~ exp(-2 t^2 / (B s^2)), so the
        same rule gives sqrt(B) C / f. Values closer than this to either end of the record
        are affected by the zero extension and should not be interpreted.
        """
        freq = np.asarray(freq, dtype=float)
        return np.sqrt(self.bandwidth) * config.CWT_CENTRE / freq

    def inside_cone(self, freq, start_s, end_s, duration_s):
        """True when the span [start_s, end_s] lies clear of the cone of influence at freq."""
        edge = float(self.cone_of_influence_s(freq))
        return bool(start_s >= edge and end_s <= duration_s - edge)

    def sigma_f_hz(self, freq):
        return float(freq / (2 * np.pi * np.sqrt(self.bandwidth / 2) * config.CWT_CENTRE))


def default_cwt(fs):
    start, stop, count = config.CWT_FREQS
    return MorletCWT(fs, np.linspace(start, stop, count))


def operation_counts(n_samples, cwt_support, stft_windows=config.STFT_WINDOWS,
                     hop=config.STFT_HOP, dwt_taps=8):
    """Theoretical cost in n log2 n butterfly units (about one complex multiply-add each).

    FFT          O(N log N)                one length-N transform
    STFT         O((N/H) L log L)          about N/H frames of one length-L FFT
    CWT (FFT)    O(J M log M)              forward and inverse FFT per scale, data FFT reused
    CWT (direct) O(J N L)                  time-domain convolution per scale
    DWT          O(N)                      two filters, length halves at each level
    """
    counts = {"FFT": n_samples * np.log2(n_samples)}
    for length in stft_windows:
        frames = int(np.ceil((n_samples + length) / hop))
        counts[f"STFT_{length}"] = frames * length * np.log2(length)
    fft_lengths = np.array([next_fast_len(n_samples + s - 1) for s in cwt_support])
    counts["CWT_fft"] = float(np.sum(2 * fft_lengths * np.log2(fft_lengths))
                              + fft_lengths.max() * np.log2(fft_lengths.max()))
    counts["CWT_direct"] = float(np.sum(n_samples * cwt_support))
    counts["DWT"] = float(2 * dwt_taps * n_samples)
    return counts


def dwt_band_edges(fs, level=config.DWT_LEVEL):
    """Nominal frequency band of each DWT output, approximation first."""
    bands = {f"A{level}": (0.0, fs / 2 ** (level + 1))}
    for detail in range(level, 0, -1):
        bands[f"D{detail}"] = (fs / 2 ** (detail + 1), fs / 2 ** detail)
    return bands


def dwt_single_level_reconstructions(x, wavelet=config.DWT_WAVELET, level=config.DWT_LEVEL):
    """Time-aligned reconstruction from each DWT output alone, approximation first."""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    parts = []
    for keep in range(len(coeffs)):
        masked = [c if j == keep else np.zeros_like(c) for j, c in enumerate(coeffs)]
        parts.append(pywt.waverec(masked, wavelet)[:len(x)])
    return parts
