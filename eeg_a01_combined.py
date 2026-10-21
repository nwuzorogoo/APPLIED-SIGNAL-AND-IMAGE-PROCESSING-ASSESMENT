"""eeg_a01, single file edition. CMP9780 Assessment 1, EEG signal analysis and digital filter design.

Every module of the eeg_a01 package is combined here in dependency order. Run it with

    python eeg_a01_combined.py --data-dir dataset --out-dir output

Options are the same as `python -m eeg_a01`: --draws N, --no-figures, --no-benchmarks.
"""
__version__ = "1.0.0"

import argparse
import json
import os
import platform
import time
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pywt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from scipy import signal, stats  # noqa: E402
from scipy.fft import next_fast_len  # noqa: E402


# ============================================================================================
# config.py
# Analysis settings in one place.
#
# Everything that depends on the recordings themselves (sampling rate, record length, amplitude
# range, spectral peak frequencies, pulse and burst timings) is read or detected at run time. The
# values below are design choices: band definitions, filter specifications, analysis windows
# expressed relative to detected events, and Monte Carlo settings.
# ============================================================================================

RECORD_IDS = (1, 2, 3)
EDF_PATTERN = "eeg_dataset_0{}.edf"

# Conventional EEG bands in Hz, as [low, high)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
CLINICAL_CEILING_HZ = 45.0      # upper edge of clinically used EEG
SUB_DELTA_HZ = 0.5              # lower edge of delta, baseline wander lives below it
LF_RESIDUE_HZ = 0.4             # band used to measure residual drift after filtering

# Spectral peak picking (Task 1c)
PEAK_NFFT = 2 ** 16             # zero-padded length used to interpolate peak positions
PLOT_NFFT = 2 ** 14
PEAK_MIN_AMPLITUDE_UV = 5.0
PEAK_MIN_SPACING_HZ = 0.3
DOMINANT_PEAK_UV = 7.0          # peaks above this are treated as the record's main lines

# Aliasing demonstration (Task 1b)
ALIAS_RECORD = 1
ALIAS_FACTORS = (4, 5, 8)
ALIAS_DEMO_FACTOR = 8
ALIAS_SECOND_FACTOR = 5
ALIAS_NFFT = 2 ** 12
ALIAS_TOLERANCE_HZ = 0.25

# Convolution kernels (Task 2a)
GAUSS_CUTOFFS_HZ = (40.0, 15.0)  # -3 dB points of the two Gaussian kernels
GAUSS_SUPPORT_SIGMAS = 4.0
MOVING_AVERAGE_TAPS = 9

# Time-frequency analysis (Task 2b)
STFT_WINDOWS = (128, 512)       # 0.5 s and 2 s at 256 Hz
STFT_HOP = 8
CWT_WAVELET = "cmor1.5-1.0"
CWT_BANDWIDTH = 1.5
CWT_CENTRE = 1.0
CWT_FREQS = (1.0, 45.0, 120)    # start, stop, count
DWT_WAVELET = "db4"
DWT_LEVEL = 5
THETA_BURST_BAND = (5.5, 7.5)
PULSE_STFT_BAND = (30.0, 60.0)
PULSE_CWT_BAND = (30.0, 45.0)
CWT_SIGMA_PROBES_HZ = (6.5, 10.6, 35.0)
SCALING_LENGTHS = 5             # record lengths N, 2N, ..., 16N for the empirical big-O fit

# Band-pass design (Task 3)
PASSBAND = (0.5, 40.0)
DESIGN_ORDER = 4
ELLIP_RIPPLE_DB = 0.5
ELLIP_STOP_DB = 40.0
ZERO_PHASE_FACTOR = 2.0         # sosfiltfilt squares |H|, so zero-phase dB = 2 x single-pass dB
SPEC_PASS_HZ, SPEC_STOP_HZ = 40.0, 50.0
SPEC_PASS_DB, SPEC_STOP_DB = 0.5, 40.0   # zero-phase targets, halved per pass
GROUP_DELAY_BAND = (1.0, 39.0)
GROUP_DELAY_PROBES_HZ = (1.0, 5.0, 20.0, 40.0)
PAD_SECONDS = 3.0               # odd padding before every zero-phase call
BUTTER_ORDERS = (2, 4, 6, 8)
ELLIP_ORDERS = (2, 3, 4, 5)
ELLIP_RIPPLES_DB = (0.05, 0.1, 0.25, 0.5, 1.0)   # passband ripple study at N = 4
BUTTER_CUTOFFS = (30, 35, 40, 45)
CUTOFF_PROBE_RECORD = 2        # its beta line sits closest to the cut-off under study

# Mains notch and detector
MAINS_HZ = 50.0
NOTCH_Q = 30.0
NOTCH_Q_STUDY = (10, 30, 50)
DETECTOR_THRESHOLD_DB = 20.0
DETECTOR_LINE_HALF_WIDTH_HZ = 0.5
DETECTOR_FLOOR_HZ = (5.0, 1.0)  # floor taken 1 to 5 Hz either side of the line

# Pulse detection: a pulse is a large rising edge followed by a large falling edge
PULSE_EDGE_SIGMAS = 6.0         # edge threshold in robust SDs of the first difference
PULSE_DURATION_S = (0.05, 0.15)
EDGE_SEARCH_HALF_WIDTH_S = 0.4   # envelope peaks are sought this far either side of a pulse
EDGE_PEAK_PROMINENCE = 0.25     # fraction of the local envelope maximum
TRANSIENT_RECORD = 3            # record whose first pulse is used for the transient tests
PULSE_FIT_SAMPLES = 5           # samples each side of an edge used to measure the step height


@dataclass(frozen=True)
class PulseWindows:
    """Analysis windows around a detected pulse, in seconds relative to its onset or offset.

    Onset is the first sample on the plateau and offset the first sample after it, so a window
    such as (onset - 0.010, offset + 0.005) spans the whole pulse plus a short margin.
    """
    peak_before_onset: float = 0.010
    peak_after_offset: float = 0.005
    slope_before_onset: float = 0.018
    slope_after_onset: float = 0.010
    rise_before_onset: float = 0.012
    rise_after_onset: float = 0.015
    ring_before_onset: float = 0.088
    ring_after_onset: float = 0.310
    locate_before_onset: float = 0.486
    locate_after_onset: float = 0.510
    fall_before_offset: float = 0.037
    fall_after_offset: float = 0.115


PULSE_WINDOWS = PulseWindows()

# Monte Carlo stress test
N_DRAWS = 200
SEED_STRIDE = 10_000            # draw d of record i uses seed SEED_STRIDE * i + d
OFFSET_DRAWS = 50
MAINS_OFFSETS_HZ = (49.8, 50.0, 50.2)
OFFSET_MAINS_UV = 15.0
WEAK_MAINS_UV = 2.0
EDGE_EXCLUSION_S = 1.0          # inner window drops this much at each end


@dataclass(frozen=True)
class Contamination:
    """Ranges of the random contaminants added in each stress-test draw."""
    drift_amp_uv: tuple = (15.0, 35.0)
    drift_freq_hz: tuple = (0.1, 0.3)
    trend_uv_per_s: tuple = (-4.0, 4.0)
    mains_amp_uv: tuple = (5.0, 25.0)
    emg_band_hz: tuple = (60.0, 120.0)
    emg_scale_uv: tuple = (5.0, 11.0)


CONTAMINATION = Contamination()


# ============================================================================================
# names.py
# Labels shared by the analysis, the benchmarks and the plots.
# ============================================================================================

FILTER_NAMES = {"butter": "Butterworth (N=4)", "ellip": "Elliptic (N=4)",
                "butter_notch": "Butterworth + notch"}
STRESS_NAMES = ("Butterworth", "Elliptic", "Butterworth+notch")


def freq_key(freq):
    """Short key for a frequency, such as 29.39 -> '29p4'."""
    return f"{freq:.1f}".replace(".", "p")


# ============================================================================================
# spectral.py
# Custom FFT and the spectral measurements built on it.
# ============================================================================================

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


# ============================================================================================
# convolution.py
# Direct convolution and the smoothing kernels used in Task 2a.
# ============================================================================================

def convolve_same(x, h):
    """Direct-form linear convolution y[n] = sum_k h[k] x[n - k], centred to len(x).

    Samples within (len(h) - 1) // 2 of either end are computed against implicit zeros.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if x.ndim != 1 or h.ndim != 1 or x.size == 0 or h.size == 0:
        raise ValueError("convolve_same needs two non-empty 1-D arrays")
    full = np.zeros(x.size + h.size - 1)
    for k, weight in enumerate(h):
        full[k:k + x.size] += weight * x
    shift = (h.size - 1) // 2
    return full[shift:shift + x.size]


def gaussian_sigma_for_cutoff(cutoff_hz):
    """Gaussian width whose response exp(-2 pi^2 sigma^2 f^2) is -3 dB at cutoff_hz."""
    if cutoff_hz <= 0:
        raise ValueError("cutoff must be positive")
    return np.sqrt(np.log(2)) / (2 * np.pi * cutoff_hz)


def gaussian_kernel(sigma_s, fs, support_sigmas=4.0):
    """Sampled Gaussian of width sigma_s seconds, truncated at +-support_sigmas, unit DC gain."""
    if sigma_s <= 0:
        raise ValueError("sigma must be positive")
    half = int(np.ceil(support_sigmas * sigma_s * fs))
    lags = np.arange(-half, half + 1) / fs
    kernel = np.exp(-lags ** 2 / (2 * sigma_s ** 2))
    return kernel / kernel.sum()


def moving_average_kernel(taps):
    if taps < 1:
        raise ValueError("a moving average needs at least one tap")
    return np.ones(taps) / taps


def edge_samples(kernel):
    """Number of output samples at each end that see the implicit zero padding."""
    return (len(kernel) - 1) // 2


def rise_time_ms(z, fs, start, end, points=2000):
    """10 to 90 % rise time of an edge between sample indices start and end.

    The pre-step level is z[start] and the plateau z[end]. The segment, end sample included, is
    linearly interpolated and the 90 % crossing is searched only after the 10 % crossing.
    """
    if not 0 <= start < end < len(z):
        raise ValueError(f"invalid edge indices {start}, {end}")
    segment = np.asarray(z[start:end + 1], dtype=float)
    low, high = float(z[start]), float(z[end])
    if high <= low:
        raise ValueError("edge does not rise between the given samples")
    seg_time = np.arange(segment.size) / fs
    fine_time = np.linspace(0, seg_time[-1], points)
    fine = np.interp(fine_time, seg_time, segment)
    i10 = int(np.argmax(fine > low + 0.1 * (high - low)))
    after = fine[i10:] >= low + 0.9 * (high - low)
    i90 = i10 + int(np.argmax(after))
    return float((fine_time[i90] - fine_time[i10]) * 1e3)


# ============================================================================================
# timefreq.py
# STFT, continuous and discrete wavelet transforms, and their operation counts.
# ============================================================================================

def stft_magnitude(x, fs, window_len, hop=STFT_HOP):
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

    def __init__(self, fs, freqs, wavelet=CWT_WAVELET,
                 bandwidth=CWT_BANDWIDTH):
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
        return float(np.sqrt(self.bandwidth / 2) * CWT_CENTRE / freq * 1e3)

    def cone_of_influence_s(self, freq):
        """Edge e-folding time of the CWT at freq, in seconds (Torrence and Compo, 1998).

        Their Morlet has |psi|^2 ~ exp(-t^2 / s^2) and a cone of sqrt(2) s. The pywt complex
        Morlet exp(-t^2 / B) at scale s = C / f has |psi|^2 ~ exp(-2 t^2 / (B s^2)), so the
        same rule gives sqrt(B) C / f. Values closer than this to either end of the record
        are affected by the zero extension and should not be interpreted.
        """
        freq = np.asarray(freq, dtype=float)
        return np.sqrt(self.bandwidth) * CWT_CENTRE / freq

    def inside_cone(self, freq, start_s, end_s, duration_s):
        """True when the span [start_s, end_s] lies clear of the cone of influence at freq."""
        edge = float(self.cone_of_influence_s(freq))
        return bool(start_s >= edge and end_s <= duration_s - edge)

    def sigma_f_hz(self, freq):
        return float(freq / (2 * np.pi * np.sqrt(self.bandwidth / 2) * CWT_CENTRE))


def default_cwt(fs):
    start, stop, count = CWT_FREQS
    return MorletCWT(fs, np.linspace(start, stop, count))


def operation_counts(n_samples, cwt_support, stft_windows=STFT_WINDOWS,
                     hop=STFT_HOP, dwt_taps=8):
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


def dwt_band_edges(fs, level=DWT_LEVEL):
    """Nominal frequency band of each DWT output, approximation first."""
    bands = {f"A{level}": (0.0, fs / 2 ** (level + 1))}
    for detail in range(level, 0, -1):
        bands[f"D{detail}"] = (fs / 2 ** (detail + 1), fs / 2 ** detail)
    return bands


def dwt_single_level_reconstructions(x, wavelet=DWT_WAVELET, level=DWT_LEVEL):
    """Time-aligned reconstruction from each DWT output alone, approximation first."""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    parts = []
    for keep in range(len(coeffs)):
        masked = [c if j == keep else np.zeros_like(c) for j, c in enumerate(coeffs)]
        parts.append(pywt.waverec(masked, wavelet)[:len(x)])
    return parts


# ============================================================================================
# edf_io.py
# Loading the EDF recordings and reading their header calibration.
# ============================================================================================

_FIXED_HEADER_BYTES = 256
# Per-signal header fields in file order, with their width in bytes (EDF specification)
_SIGNAL_FIELDS = (
    ("label", 16), ("transducer", 80), ("physical_dimension", 8),
    ("physical_min", 8), ("physical_max", 8), ("digital_min", 8), ("digital_max", 8),
    ("prefiltering", 80), ("samples_per_record", 8), ("reserved", 32),
)


@dataclass(frozen=True)
class ChannelCalibration:
    """Physical and digital range of one EDF channel."""
    label: str
    unit: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int

    @property
    def lsb(self):
        """Quantisation step in physical units per digital count."""
        return (self.physical_max - self.physical_min) / (self.digital_max - self.digital_min)

    @property
    def full_scale(self):
        """Largest representable magnitude in physical units."""
        return max(abs(self.physical_min), abs(self.physical_max))


@dataclass(frozen=True)
class Recording:
    """One single-channel EEG record in microvolts."""
    record_id: int
    data: np.ndarray
    fs: float
    calibration: ChannelCalibration

    @property
    def n_samples(self):
        return self.data.size

    @property
    def duration(self):
        return self.data.size / self.fs

    @property
    def time(self):
        return np.arange(self.data.size) / self.fs


def read_calibration(path, channel=0):
    """Read the physical and digital range of one channel straight from the EDF header."""
    raw = Path(path).read_bytes()
    if len(raw) < _FIXED_HEADER_BYTES:
        raise ValueError(f"{path} is too short to be an EDF file")
    n_signals = int(raw[252:256].decode("ascii").strip())
    if not 0 <= channel < n_signals:
        raise ValueError(f"channel {channel} not in file with {n_signals} signals")
    fields, offset = {}, _FIXED_HEADER_BYTES
    for name, width in _SIGNAL_FIELDS:
        start = offset + channel * width
        fields[name] = raw[start:start + width].decode("ascii").strip()
        offset += n_signals * width
    return ChannelCalibration(
        label=fields["label"], unit=fields["physical_dimension"],
        physical_min=float(fields["physical_min"]), physical_max=float(fields["physical_max"]),
        digital_min=int(fields["digital_min"]), digital_max=int(fields["digital_max"]),
    )


def load_recording(path, record_id):
    """Load the first EEG channel of an EDF file, converted from volts to microvolts."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    calibration = read_calibration(path)
    if calibration.unit not in ("uV", "µV"):
        raise ValueError(f"{path}: expected a microvolt channel, header says {calibration.unit!r}")
    data = raw.get_data()[0] * 1e6
    return Recording(record_id, data, float(raw.info["sfreq"]), calibration)


def load_recordings(data_dir, record_ids=RECORD_IDS, pattern=EDF_PATTERN):
    """Load all records and check they share one sampling rate and length."""
    data_dir = Path(data_dir)
    records = {}
    for record_id in record_ids:
        path = data_dir / pattern.format(record_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing recording {path}")
        records[record_id] = load_recording(path, record_id)
    rates = {r.fs for r in records.values()}
    lengths = {r.n_samples for r in records.values()}
    if len(rates) != 1 or len(lengths) != 1:
        raise ValueError(f"records differ in sampling rate {rates} or length {lengths}")
    return records


# ============================================================================================
# filters.py
# IIR band-pass and notch design, zero-phase application and response measures.
#
# Reporting convention: every IIR filter is applied forward and backward (sosfiltfilt), so every
# magnitude in dB is zero-phase, |H_zp(f)| = |H(f)|^2, twice the single-pass figure. Group delay
# is the exception, since zero-phase delay is identically zero; it is reported for causal
# single-pass use and labelled as such.
# ============================================================================================

ZP = ZERO_PHASE_FACTOR


def butter_bandpass(order, fs, low=PASSBAND[0], high=PASSBAND[1]):
    return signal.butter(order, [low, high], "bandpass", fs=fs, output="sos")


def ellip_bandpass(order, fs, low=PASSBAND[0], high=PASSBAND[1],
                   ripple_db=ELLIP_RIPPLE_DB, stop_db=ELLIP_STOP_DB):
    return signal.ellip(order, ripple_db, stop_db, [low, high], "bandpass", fs=fs, output="sos")


def notch(fs, f0=MAINS_HZ, q=NOTCH_Q):
    """Second-order IIR notch at f0 with quality factor q, as one SOS section."""
    b, a = signal.iirnotch(f0, q, fs=fs)
    return signal.tf2sos(b, a)


def cascade(*sections):
    """Join SOS filters into a single cascade."""
    return np.vstack(sections)


def zero_phase(sos, x, fs):
    """Forward-backward filtering with one odd-padding rule for every call."""
    return signal.sosfiltfilt(sos, x, padtype="odd", padlen=int(PAD_SECONDS * fs))


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


def stopband_edge_hz(sos, fs, stop_db, start_hz=PASSBAND[1], n_points=2 ** 15):
    """First frequency above start_hz where the single-pass attenuation reaches stop_db."""
    freqs = np.linspace(start_hz, fs / 2, n_points)
    _, h = signal.sosfreqz(sos, worN=freqs, fs=fs)
    reached = 20 * np.log10(np.abs(h) + 1e-15) <= -stop_db
    return float(freqs[np.argmax(reached)]) if reached.any() else None


def transmission_zeros_hz(sos, fs, above_hz=PASSBAND[1]):
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
    args = (SPEC_PASS_HZ, SPEC_STOP_HZ, SPEC_PASS_DB / ZP,
            SPEC_STOP_DB / ZP)
    return {
        "butter": signal.buttord(*args, fs=fs)[0],
        "cheby1": signal.cheb1ord(*args, fs=fs)[0],
        "cheby2": signal.cheb2ord(*args, fs=fs)[0],
        "ellip": signal.ellipord(*args, fs=fs)[0],
    }


def minimum_order_lowpass(family, order, fs):
    """Low-pass designs matching minimum_orders, used for the family comparison figure."""
    ripple, stop = SPEC_PASS_DB / ZP, SPEC_STOP_DB / ZP
    edge = SPEC_PASS_HZ
    designs = {
        "butter": lambda: signal.butter(order, edge, fs=fs, output="sos"),
        "cheby1": lambda: signal.cheby1(order, ripple, edge, fs=fs, output="sos"),
        "cheby2": lambda: signal.cheby2(order, stop, SPEC_STOP_HZ, fs=fs, output="sos"),
        "ellip": lambda: signal.ellip(order, ripple, stop, edge, fs=fs, output="sos"),
    }
    if family not in designs:
        raise ValueError(f"unknown filter family {family!r}")
    return designs[family]()


def ideal_bandpass(x, fs, low=PASSBAND[0], high=PASSBAND[1]):
    """Brick-wall band-pass in the DFT domain, the reference output for evaluation."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, len(x))


def mains_line_ratio_db(x, fs, f0=MAINS_HZ, nfft=PLOT_NFFT):
    """Mains line amplitude over the median of the nearby floor, in dB."""
    freqs, amp = amplitude_spectrum(x, fs, nfft=nfft)
    half = DETECTOR_LINE_HALF_WIDTH_HZ
    outer, inner = DETECTOR_FLOOR_HZ
    line = amp[(freqs > f0 - half) & (freqs < f0 + half)].max()
    below = (freqs > f0 - outer) & (freqs < f0 - inner)
    above = (freqs > f0 + inner) & (freqs < f0 + outer)
    floor = np.median(amp[below | above])
    return float(20 * np.log10(line / floor))


def mains_present(x, fs, threshold_db=DETECTOR_THRESHOLD_DB):
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


# ============================================================================================
# events.py
# Detection of the transient events used in the analysis, so no event time is typed in.
# ============================================================================================

@dataclass(frozen=True)
class Pulse:
    """A rectangular pulse artefact. onset is the first plateau sample, offset the first after."""
    onset: int
    offset: int
    fs: float

    @property
    def onset_s(self):
        return self.onset / self.fs

    @property
    def offset_s(self):
        return self.offset / self.fs

    @property
    def centre_s(self):
        return (self.onset_s + self.offset_s) / 2


def robust_sd(x):
    """Standard deviation estimated from the median absolute deviation."""
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def detect_pulses(x, fs, edge_sigmas=PULSE_EDGE_SIGMAS,
                  duration=PULSE_DURATION_S):
    """Find pulses as a large rising edge followed by a large falling edge.

    Edges are first differences beyond edge_sigmas robust SDs. Each falling edge is paired with
    the largest rising edge that precedes it by a plausible pulse duration, so a smaller step
    on the plateau or a lone edge from muscle activity is not taken as a pulse.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        raise ValueError("signal too short for edge detection")
    step = np.diff(x)
    threshold = edge_sigmas * robust_sd(step)
    rises = np.flatnonzero(step > threshold)
    falls = np.flatnonzero(step < -threshold)
    shortest, longest = (int(round(d * fs)) for d in duration)
    pulses, used = [], set()
    for fall in falls:
        candidates = [r for r in rises if shortest <= fall - r <= longest and r not in used]
        if not candidates:
            continue
        rise = max(candidates, key=lambda r: step[r])
        used.add(rise)
        pulses.append(Pulse(onset=int(rise) + 1, offset=int(fall) + 1, fs=fs))
    return pulses


def edge_times(axis, envelope, pulse, half_width=EDGE_SEARCH_HALF_WIDTH_S,
               prominence=EDGE_PEAK_PROMINENCE):
    """Estimate a pulse's onset and offset times from a time-frequency envelope.

    Prominent envelope peaks are found within half_width of the pulse centre. The first is
    taken as the onset and the last as the offset, so a method that merges both edges into one
    blob reports the same time for each. Returns (onset_s, offset_s, number of peaks).
    """
    axis = np.asarray(axis, dtype=float)
    envelope = np.asarray(envelope, dtype=float)
    if axis.shape != envelope.shape:
        raise ValueError("axis and envelope must have the same shape")
    mask = window_mask(axis, pulse.centre_s - half_width, pulse.centre_s + half_width)
    if not mask.any():
        raise ValueError("no envelope samples near the pulse")
    local_axis, local = axis[mask], envelope[mask]
    peaks, _ = signal.find_peaks(local, prominence=prominence * local.max())
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(local))])
    return float(local_axis[peaks[0]]), float(local_axis[peaks[-1]]), int(peaks.size)


def band_envelope(x, fs, band, order=4):
    """Hilbert envelope of a zero-phase Butterworth band-pass of x."""
    sos = signal.butter(order, list(band), "bandpass", fs=fs, output="sos")
    return np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))


def step_height(x, k, width=PULSE_FIT_SAMPLES):
    """Height of the step between x[k-1] and x[k].

    A quadratic is fitted to width samples on each side and both are extrapolated to the
    midpoint between the two samples, so the local slope of the EEG is not counted as step.
    """
    x = np.asarray(x, dtype=float)
    if k - width < 0 or k + width > x.size:
        raise ValueError("not enough samples around the edge to measure its height")
    before = np.polyfit(np.arange(-width, 0), x[k - width:k], 2)
    after = np.polyfit(np.arange(width), x[k:k + width], 2)
    return float(np.polyval(after, -0.5) - np.polyval(before, -0.5))


def measurable(pulse, n_samples, width=PULSE_FIT_SAMPLES):
    """True when both edges of the pulse have width samples on each side to fit."""
    return pulse.onset - width >= 0 and pulse.offset + width <= n_samples


def remove_pulses(x, pulses, width=PULSE_FIT_SAMPLES):
    """Subtract each pulse as an additive step, keeping the EEG that rides on its plateau.

    The rising and falling step heights are measured separately and a straight ramp between
    them is subtracted over the plateau, which allows for a slightly sloping pulse top. A pulse
    too close to either end of the record for its step heights to be measured is left in place
    with a warning, so one awkward pulse cannot stop the whole analysis.
    """
    y = np.asarray(x, dtype=float).copy()
    for p in pulses:
        if not measurable(p, y.size, width):
            warnings.warn(f"pulse at samples {p.onset} to {p.offset} is within {width} samples "
                          "of the record edge and was left in place", RuntimeWarning, stacklevel=2)
            continue
        rise = step_height(y, p.onset, width)
        fall = -step_height(y, p.offset, width)
        y[p.onset:p.offset] -= np.linspace(rise, fall, p.offset - p.onset)
    return y


def interpolate_pulses(x, pulses):
    """Replace each pulse by a straight line between its neighbours (a cruder alternative)."""
    y = np.asarray(x, dtype=float).copy()
    for p in pulses:
        start, end = p.onset - 1, p.offset
        y[start:end + 1] = np.linspace(y[start], y[end], end - start + 1)
    return y


def detect_theta_burst(x, fs, band=THETA_BURST_BAND, pulses=None):
    """Start, end and width (s) of the strongest burst, as the half-maximum span of its envelope.

    A pulse is broadband, so it adds energy to the theta band and can stretch or shift the
    burst. When pulses are given they are subtracted first (remove_pulses).
    """
    if pulses:
        x = remove_pulses(x, pulses)
    envelope = band_envelope(x, fs, band)
    return half_maximum_span(np.arange(x.size) / fs, envelope)


def window_mask(time, start, end):
    """Boolean mask of start < time < end."""
    return (time > start) & (time < end)


class PulseWindowSet:
    """Analysis windows anchored on one detected pulse (see PulseWindows)."""

    def __init__(self, pulse, margins=PULSE_WINDOWS):
        self.pulse = pulse
        self.margins = margins

    def _span(self, before, after, anchor_before="onset", anchor_after="onset"):
        anchors = {"onset": self.pulse.onset_s, "offset": self.pulse.offset_s}
        return anchors[anchor_before] - before, anchors[anchor_after] + after

    @property
    def peak(self):
        m = self.margins
        return self._span(m.peak_before_onset, m.peak_after_offset, anchor_after="offset")

    @property
    def slope(self):
        m = self.margins
        return self._span(m.slope_before_onset, m.slope_after_onset)

    @property
    def ringing(self):
        m = self.margins
        return self._span(m.ring_before_onset, m.ring_after_onset)

    @property
    def pre_ringing(self):
        m = self.margins
        return self._span(m.ring_before_onset, -m.peak_before_onset)

    @property
    def post_ringing(self):
        m = self.margins
        return self.pulse.offset_s + m.peak_after_offset, self.ringing[1]

    @property
    def localisation(self):
        m = self.margins
        return self._span(m.locate_before_onset, m.locate_after_onset)

    @property
    def falling_edge(self):
        m = self.margins
        return self._span(m.fall_before_offset, m.fall_after_offset, "offset", "offset")

    def rise_samples(self):
        """Sample indices bracketing the leading edge, for the rise-time measurement."""
        m, fs = self.margins, self.pulse.fs
        start = int(round((self.pulse.onset_s - m.rise_before_onset) * fs))
        end = int(round((self.pulse.onset_s + m.rise_after_onset) * fs))
        return start, end


# ============================================================================================
# evaluation.py
# Stress-test contamination, filter metrics and paired statistics.
# ============================================================================================

METRIC_NAMES = ("snr_in", "snr_out", "gain", "gain_inner", "mains", "mains_rel_dB", "corr")


def draw_seed(record_id, draw):
    return SEED_STRIDE * record_id + draw


def contaminate(x, fs, rng, mains_hz=MAINS_HZ, mains_amp=None,
                spec=CONTAMINATION):
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
    return amplitude_near(*amplitude_spectrum(x, fs, nfft=PLOT_NFFT), MAINS_HZ, 0.3)


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


# ============================================================================================
# study_base.py
# Shared state for every task: the recordings, detected events, designed filters and helpers.
# ============================================================================================

class StudyBase:
    """Holds the recordings, the quantities derived from them and the designed filters."""

    def __init__(self, records, n_draws=N_DRAWS):
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

        self.peaks = {i: spectral_peaks(self.x[i], self.fs, PEAK_NFFT,
                                        PEAK_MIN_AMPLITUDE_UV,
                                        PEAK_MIN_SPACING_HZ,
                                        CLINICAL_CEILING_HZ)
                      for i in self.ids}
        self.main_peaks = {i: [(f, a) for f, a in self.peaks[i] if a > DOMINANT_PEAK_UV]
                           for i in self.ids}
        self.pulses = {i: detect_pulses(self.x[i], self.fs) for i in self.ids}
        # pulses are subtracted before the burst is measured, because a broadband pulse adds
        # energy to the theta band; the raw-signal burst is kept to show the difference
        self.x_clean = {i: remove_pulses(self.x[i], self.pulses[i]) for i in self.ids}
        self.bursts = {i: detect_theta_burst(self.x[i], self.fs, pulses=self.pulses[i])
                       for i in self.ids}
        self.bursts_raw = {i: detect_theta_burst(self.x[i], self.fs) for i in self.ids}
        if not self.pulses[TRANSIENT_RECORD]:
            raise ValueError(f"no pulse found in record {TRANSIENT_RECORD}")
        self.pulse = self.pulses[TRANSIENT_RECORD][0]
        self.windows = PulseWindowSet(self.pulse)

        self.sos_butter = butter_bandpass(DESIGN_ORDER, self.fs)
        self.sos_ellip = ellip_bandpass(DESIGN_ORDER, self.fs)
        self.sos_notch = notch(self.fs)
        self.sos_butter_notch = cascade(self.sos_butter, self.sos_notch)
        self.enhance = Enhancer(self.fs, self.sos_butter, self.sos_notch)
        self.cwt = default_cwt(self.fs)
        self.mc_raw = {}

    # ------------------------------------------------------------------ helpers
    def zp(self, sos, x):
        return zero_phase(sos, x, self.fs)

    def band_power(self, x, low, high):
        return band_power(x, self.fs, low, high)

    def spectrum(self, x, nfft=None):
        return amplitude_spectrum(x, self.fs, nfft=nfft)

    def mask(self, span):
        return window_mask(self.time, *span)

    def probe(self, record_id, band):
        """Frequency of the strongest spectral line of a record in a named band."""
        return dominant_peak(self.peaks[record_id], BANDS[band])

    @property
    def high_band(self):
        """Band above clinical EEG, up to Nyquist."""
        return CLINICAL_CEILING_HZ, self.nyquist

    def events_summary(self):
        return {i: {"pulses_s": [[p.onset_s, p.offset_s] for p in self.pulses[i]],
                    "theta_burst_s": list(self.bursts[i][:2]),
                    "theta_burst_raw_s": list(self.bursts_raw[i][:2])}
                for i in self.ids}


# ============================================================================================
# task1.py
# Task 1: FFT check, signal characteristics and aliasing.
# ============================================================================================

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
                "pow_below_05_pct": float(self.band_power(x, 0, SUB_DELTA_HZ)
                                          / total * 100),
                "rel_band": {k: float(self.band_power(x, *v) / total * 100)
                             for k, v in BANDS.items()},
            }
        return out

    # ------------------------------------------------------------------ Task 1b
    def decimated(self, factor):
        x = self.x[ALIAS_RECORD]
        fs_new = self.fs / factor
        naive = x[::factor]
        filtered = signal.decimate(x, factor, ftype="fir", zero_phase=True)
        return fs_new, naive, filtered

    @cached_property
    def alias_probe(self):
        """The alias record's beta line and where it folds at the demonstration rate."""
        beta = self.probe(ALIAS_RECORD, "beta")
        return {"beta": beta,
                "folded": alias_frequency(beta, self.fs / ALIAS_DEMO_FACTOR)}

    def aliasing(self):
        x = self.x[ALIAS_RECORD]
        beta = self.probe(ALIAS_RECORD, "beta")
        delta = self.probe(ALIAS_RECORD, "delta")
        alpha = self.probe(ALIAS_RECORD, "alpha")
        table = {i: [{"f": f, "M": m, "fs_new": self.fs / m,
                      "alias": round(alias_frequency(f, self.fs / m), 2),
                      "aliased": bool(f > self.fs / m / 2)}
                     for f, _ in self.main_peaks[i] for m in ALIAS_FACTORS]
                 for i in self.ids}
        f0, a0 = self.spectrum(x, PLOT_NFFT)
        tol = ALIAS_TOLERANCE_HZ
        demos = {}
        for factor in (ALIAS_DEMO_FACTOR, ALIAS_SECOND_FACTOR):
            fs_new, naive, filtered = self.decimated(factor)
            demos[factor] = (fs_new, amplitude_spectrum(naive, fs_new, ALIAS_NFFT),
                             amplitude_spectrum(filtered, fs_new, ALIAS_NFFT))
        fs8, naive8, aa8 = demos[ALIAS_DEMO_FACTOR]
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
        fs5, naive5, aa5 = demos[ALIAS_SECOND_FACTOR]
        folded5 = alias_frequency(beta, fs5)
        demo5 = {"fs_new": fs5, "alias_f": folded5,
                 "naive": amplitude_near(*naive5, folded5), "aa": amplitude_near(*aa5, folded5)}
        return {"alias_table": table, "alias_demo": demo, "alias_demo5": demo5}


# ============================================================================================
# task2.py
# Task 2: convolution, time-frequency localisation, pulse timing and cost.
# ============================================================================================

class Task2TimeFrequency:
    """Convolution kernels and the FFT, STFT, CWT and DWT comparison. Mixed into Study."""

    # ------------------------------------------------------------------ Task 2a
    def kernels(self):
        sigmas = [gaussian_sigma_for_cutoff(c) for c in GAUSS_CUTOFFS_HZ]
        kernels = {}
        for cutoff, sigma in zip(GAUSS_CUTOFFS_HZ, sigmas):
            kernels[f"gauss{cutoff:g}"] = (gaussian_kernel(sigma, self.fs,
                                                                GAUSS_SUPPORT_SIGMAS),
                                           sigma)
        kernels[f"ma{MOVING_AVERAGE_TAPS}"] = (
            moving_average_kernel(MOVING_AVERAGE_TAPS), None)
        return kernels

    def pulse_measures(self, z):
        """Peak, steepest slope and rise time of the reference pulse in signal z."""
        start, end = self.windows.rise_samples()
        return {
            "peak": float(z[self.mask(self.windows.peak)].max()),
            "slope": float(np.max(np.diff(z[self.mask(self.windows.slope)])) * self.fs / 1e3),
            "rise": rise_time_ms(z, self.fs, start, end),
        }

    def convolution(self):
        rec = TRANSIENT_RECORD
        x1, x3 = self.x[ALIAS_RECORD], self.x[rec]
        probes = {"att_29p4": self.probe(ALIAS_RECORD, "beta"),
                  "att_18": self.probe(rec, "beta"),
                  "att_10p6": self.probe(rec, "alpha"),
                  "att_50": MAINS_HZ}
        kernels = self.kernels()
        g40 = kernels["gauss40"][0]
        check = convolve_same(x1, g40) - np.convolve(x1, g40, "same")
        out = {"conv_check_err": float(np.max(np.abs(check)))}
        raw = self.pulse_measures(x3)
        out.update(raw_rise_ms=raw["rise"], raw_peak=raw["peak"], raw_slope=raw["slope"])
        for name, (h, sigma) in kernels.items():
            w, resp = signal.freqz(h, worN=8192, fs=self.fs)
            mag = np.abs(resp)

            def att(f0):
                return float(20 * np.log10(mag[np.argmin(np.abs(w - f0))] + 1e-12))

            y = convolve_same(x3, h)
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
                "edge_samples": edge_samples(h),
                "edge_ms": edge_samples(h) / self.fs * 1e3,
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
        return self.cwt.magnitude(self.x[TRANSIENT_RECORD])

    def burst_spans(self, x):
        """Theta-burst half-maximum span from each time-frequency method on signal x."""
        theta_lo, theta_hi = THETA_BURST_BAND
        spans = {}
        for nper in STFT_WINDOWS:
            f, ts, z = stft_magnitude(x, self.fs, nper)
            rows = (f >= theta_lo) & (f <= theta_hi)
            spans[f"stft{nper}_burst"] = half_maximum_span(ts, z[rows].mean(0))
        w, fr = self.cwt.magnitude(x)
        rows = (fr >= theta_lo) & (fr <= theta_hi)
        spans["cwt_burst"] = half_maximum_span(self.time, w[rows].mean(0))
        return spans

    def localisation(self):
        rec = TRANSIENT_RECORD
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
        for nper in STFT_WINDOWS:
            f, ts, z = stft_magnitude(x3, self.fs, nper)
            rows = (f >= PULSE_STFT_BAND[0]) & (f <= PULSE_STFT_BAND[1])
            cols = window_mask(ts, *self.windows.localisation)
            loc[f"stft{nper}_pulse"] = half_maximum_span(ts[cols], z[rows].mean(0)[cols])
            loc[f"stft{nper}_df"] = self.fs / nper
            loc[f"stft{nper}_dt"] = nper / self.fs
        w, fr = self.cwt_map
        rows = (fr >= PULSE_CWT_BAND[0]) & (fr <= PULSE_CWT_BAND[1])
        envelope = w[rows].mean(0)
        cols = self.mask(self.windows.localisation)
        loc["cwt_pulse"] = half_maximum_span(t[cols], envelope[cols])
        cols = self.mask(self.windows.falling_edge)
        loc["cwt_pulse_falling_edge"] = half_maximum_span(t[cols], envelope[cols])

        alpha = self.probe(TRANSIENT_RECORD, "alpha")
        key = freq_key(alpha)
        f, a = self.spectrum(x3, PEAK_NFFT)
        loc[f"fft_linewidth_{key}"] = line_width(f, a, alpha)
        for nper in STFT_WINDOWS:
            f, _, z = stft_magnitude(x3, self.fs, nper)
            loc[f"stft{nper}_linewidth_{key}"] = line_width(f, z.mean(1), alpha)
        order = np.argsort(fr)
        loc[f"cwt_linewidth_{key}"] = line_width(fr[order], w.mean(1)[order], alpha)
        burst_freq = float(np.mean(THETA_BURST_BAND))
        loc["cwt_coi_s"] = {f"{freq:g}": float(self.cwt.cone_of_influence_s(freq))
                            for freq in (CWT_FREQS[0], burst_freq,
                                         *CWT_SIGMA_PROBES_HZ)}
        loc["cwt_burst_clear_of_coi"] = self.cwt.inside_cone(burst_freq, *self.bursts[rec][:2],
                                                             self.duration)
        loc["cwt_pulses_clear_of_coi"] = {
            f"DS{i}_{p.onset_s:.3f}s": self.cwt.inside_cone(PULSE_CWT_BAND[0], p.onset_s,
                                                           p.offset_s, self.duration)
            for i in self.ids for p in self.pulses[i]}
        loc["cwt_area_inside_coi_frac"] = float(np.mean(
            [(self.time >= c) & (self.time <= self.duration - c)
             for c in self.cwt.cone_of_influence_s(fr)]))
        for freq in CWT_SIGMA_PROBES_HZ:
            loc[f"cwt_sigma_t_ms_{freq:g}"] = self.cwt.sigma_t_ms(freq)
            loc[f"cwt_sigma_f_Hz_{freq:g}"] = self.cwt.sigma_f_hz(freq)
        return loc

    def pulse_envelopes(self, record_id):
        """High-frequency band envelope of each time-frequency method, as (time axis, envelope)."""
        x = self.x[record_id]
        envelopes = {}
        for nper in STFT_WINDOWS:
            f, ts, z = stft_magnitude(x, self.fs, nper)
            rows = (f >= PULSE_STFT_BAND[0]) & (f <= PULSE_STFT_BAND[1])
            envelopes[f"stft{nper}"] = (ts, z[rows].mean(0))
        w, fr = (self.cwt_map if record_id == TRANSIENT_RECORD
                 else self.cwt.magnitude(x))
        rows = (fr >= PULSE_CWT_BAND[0]) & (fr <= PULSE_CWT_BAND[1])
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


# ============================================================================================
# task3.py
# Task 3: filter design study, notch and clean-record performance.
# ============================================================================================

class Task3Filters:
    """Butterworth and elliptic design, parameters, notch and performance. Mixed into Study."""

    # ------------------------------------------------------------------ Task 3 design
    def design_study(self):
        fs = self.fs
        probes = {"at29": self.probe(ALIAS_RECORD, "beta"),
                  "at1_84": self.probe(TRANSIENT_RECORD, "delta"),
                  "at24": self.probe(CUTOFF_PROBE_RECORD, "beta")}
        band = np.linspace(*GROUP_DELAY_BAND, 400)

        def common(sos):
            return {"att50": zero_phase_db_at(sos, fs, MAINS_HZ),
                    "att45": zero_phase_db_at(sos, fs, CLINICAL_CEILING_HZ),
                    "at29": zero_phase_db_at(sos, fs, probes["at29"]),
                    "at40": zero_phase_db_at(sos, fs, PASSBAND[1]),
                    "at1": zero_phase_db_at(sos, fs, 1.0)}

        def delay_range(sos):
            return float(np.ptp(group_delay(sos, band, fs)) / fs * 1e3)

        out = {}
        for order in BUTTER_ORDERS:
            sos = butter_bandpass(order, fs)
            out[f"butter{order}"] = {**common(sos),
                                     "at1_84": zero_phase_db_at(sos, fs, probes["at1_84"]),
                                     "sections": len(sos), "gd_range_ms_causal": delay_range(sos)}
        stop_per_pass = ELLIP_STOP_DB      # the elliptic specification is per pass

        def stopband(sos):
            return {"stop_edge_hz": stopband_edge_hz(sos, fs, stop_per_pass),
                    "zeros_hz": transmission_zeros_hz(sos, fs)}

        for order in ELLIP_ORDERS:
            sos = ellip_bandpass(order, fs)
            out[f"ellip{order}"] = {**common(sos), **stopband(sos), "sections": len(sos),
                                    "gd_range_ms_causal": delay_range(sos)}
        for ripple in ELLIP_RIPPLES_DB:
            sos = ellip_bandpass(DESIGN_ORDER, fs, ripple_db=ripple)
            passband = frequency_response(sos, fs)
            inside = (passband[0] > band[0]) & (passband[0] < band[-1])
            out[f"ellip4_rp{ripple:g}"] = {
                **common(sos), **stopband(sos),
                "ripple_zp_db": float(np.ptp(to_zero_phase_db(passband[1][inside]))),
                "gd_range_ms_causal": delay_range(sos),
                "gd_40_ms_causal": float(group_delay(sos, np.array([40.0]), fs)[0] / fs * 1e3),
                **self.ringing(sos)}
        for cutoff in BUTTER_CUTOFFS:
            sos = butter_bandpass(DESIGN_ORDER, fs, high=cutoff)
            out[f"butter4_fc{cutoff}"] = {
                "at29": zero_phase_db_at(sos, fs, probes["at29"]),
                "at24": zero_phase_db_at(sos, fs, probes["at24"]),
                "att50": zero_phase_db_at(sos, fs, MAINS_HZ)}
        for key, sos in (("butter", self.sos_butter), ("ellip", self.sos_ellip)):
            out[FILTER_NAMES[key] + "_max_pole_radius"] = max_pole_radius(sos)
        w, h_ellip = frequency_response(self.sos_ellip, fs)
        _, h_butter = frequency_response(self.sos_butter, fs)
        passband = (w > band[0]) & (w < band[-1])
        out["ellip_ripple_dB_zp"] = float(np.ptp(to_zero_phase_db(h_ellip[passband])))
        out["butter_ripple_dB_zp"] = float(np.ptp(to_zero_phase_db(h_butter[passband])))
        return out

    # ------------------------------------------------------------------ notch
    def notch_study(self):
        fs = self.fs
        beta = self.probe(ALIAS_RECORD, "beta")
        out = {"f0": MAINS_HZ, "Q": NOTCH_Q,
               "bw_single_pass_Hz": MAINS_HZ / NOTCH_Q}
        for q in NOTCH_Q_STUDY:
            sos = notch(fs, q=q)
            w, h = frequency_response(sos, fs, 2 ** 18)
            gain = to_zero_phase_db(h)
            near = (w > MAINS_HZ - 10) & (w < MAINS_HZ + 10) & (gain < -3.0)
            radius = max_pole_radius(sos)
            out[f"Q{q}"] = {
                "bw3_zp_Hz": float(w[near].max() - w[near].min()),
                **{key: zero_phase_db_at(sos, fs, f) for key, f in (
                    ("at50_zp", 50.0), ("at49p8_zp", 49.8), ("at49_zp", 49.0),
                    ("at45_zp", 45.0), ("at40_zp", 40.0), ("at29p4_zp", beta))},
                "pole_radius": radius,
                "tau_ms": float(-1 / (fs * np.log(radius)) * 1e3),
            }
        out["detect_clean"] = {i: mains_present(self.x[i], fs) for i in self.ids}
        peak_changes, rms_changes = [], []
        for i in self.ids:
            y_b = self.zp(self.sos_butter, self.x[i])
            y_bn = self.zp(self.sos_butter_notch, self.x[i])
            spec_b = self.spectrum(y_b, PEAK_NFFT)
            spec_bn = self.spectrum(y_bn, PEAK_NFFT)
            for f, _ in self.main_peaks[i]:
                ratio = amplitude_near(*spec_bn, f, 0.05) / amplitude_near(*spec_b, f, 0.05)
                peak_changes.append(abs(20 * np.log10(ratio)))
            rms_changes.append(float(np.sqrt(np.mean((y_bn - y_b) ** 2))))
        out["clean_max_peak_change_dB"] = float(max(peak_changes))
        out["clean_rms_diff_uV"] = rms_changes
        x3 = self.x[TRANSIENT_RECORD]
        reference = ideal_bandpass(x3, fs)
        ring = self.mask(self.windows.ringing)
        for key, sos in (("ring_uV_butter", self.sos_butter),
                         ("ring_uV_butter_notch", self.sos_butter_notch)):
            out[key] = float(np.max(np.abs(self.zp(sos, x3) - reference)[ring]))
        return out

    # ------------------------------------------------------------------ performance
    def ringing(self, sos):
        """Largest deviation from the ideal band-pass just before and after the DS3 pulse."""
        x3 = self.x[TRANSIENT_RECORD]
        error = np.abs(self.zp(sos, x3) - ideal_bandpass(x3, self.fs))
        return {"pre_ring_uV_zp": float(np.max(error[self.mask(self.windows.pre_ringing)])),
                "post_ring_uV_zp": float(np.max(error[self.mask(self.windows.post_ringing)]))}

    def performance(self):
        fs = self.fs
        filters = {FILTER_NAMES["butter"]: self.sos_butter, FILTER_NAMES["ellip"]: self.sos_ellip,
                   FILTER_NAMES["butter_notch"]: self.sos_butter_notch}
        x3 = self.x[TRANSIENT_RECORD]
        ref3 = ideal_bandpass(x3, fs)
        windows = {"ring_uV_zp": self.windows.ringing, "pre_ring_uV_zp": self.windows.pre_ringing,
                   "post_ring_uV_zp": self.windows.post_ringing}
        delay_band = np.linspace(*GROUP_DELAY_BAND, 400)
        out = {}
        for name, sos in filters.items():
            entry = {}
            for i in self.ids:
                x = self.x[i]
                y = self.zp(sos, x)
                ref = ideal_bandpass(x, fs)
                spec_y = self.spectrum(y, PEAK_NFFT)
                spec_x = self.spectrum(x, PEAK_NFFT)
                entry[i] = {
                    "band_change_dB_zp": {k: float(10 * np.log10(self.band_power(y, *v)
                                                                 / self.band_power(x, *v)))
                                          for k, v in BANDS.items()},
                    "peak_change_dB_zp": {
                        f: float(20 * np.log10(amplitude_near(*spec_y, f, 0.05)
                                               / amplitude_near(*spec_x, f, 0.05)))
                        for f, _ in self.main_peaks[i]},
                    "corr_ref": float(np.corrcoef(y, ref)[0, 1]),
                    "nrmse_ref": float(np.sqrt(np.mean((y - ref) ** 2)) / ref.std()),
                    "hf_atten_dB_zp": float(10 * np.log10(self.band_power(y, *self.high_band)
                                                          / self.band_power(x, *self.high_band))),
                    "lf_atten_dB_zp": float(10 * np.log10(
                        self.band_power(y, 0, LF_RESIDUE_HZ)
                        / self.band_power(x, 0, LF_RESIDUE_HZ))),
                }
            error = np.abs(self.zp(sos, x3) - ref3)
            for key, span in windows.items():
                entry[key] = float(np.max(error[self.mask(span)]))
            entry["sections"] = len(sos)
            padded_length = self.n + 2 * int(PAD_SECONDS * fs)
            entry["zero_phase_work_ratio"] = 2 * padded_length / self.n
            entry["mults_per_sample_single_pass"] = int(len(sos) * 5)
            entry["mults_per_second_causal_at_fs"] = int(len(sos) * 5 * fs)
            delay = group_delay(sos, delay_band, fs)
            entry["gd_min_ms_causal"] = float(delay.min() / fs * 1e3)
            entry["gd_max_ms_causal"] = float(delay.max() / fs * 1e3)
            probes = np.array(GROUP_DELAY_PROBES_HZ)
            entry["gd_at_ms_causal"] = {
                f"{f:g}": float(d / fs * 1e3)
                for f, d in zip(probes, group_delay(sos, probes, fs))}
            out[name] = entry
        out[FILTER_NAMES["butter_notch"]]["max_pole_radius"] = max_pole_radius(
            self.sos_butter_notch)
        return out

    def stopband_summary(self):
        return {"ellip4_att50_zp": zero_phase_db_at(self.sos_ellip, self.fs, 50),
                "butter4_att50_zp": zero_phase_db_at(self.sos_butter, self.fs, 50),
                "butter4_notch_att50_zp": zero_phase_db_at(self.sos_butter_notch, self.fs,
                                                               50)}


# ============================================================================================
# stress.py
# Task 3 stress test: randomised contamination scored against the ideal band-pass.
#
# One loop, score_draws, contaminates each record with the seeded draws and scores every filter
# it is given on the same draw. The main Butterworth, elliptic and notch comparison and the
# elliptic ripple study both call it, so they are paired on identical contamination by
# construction rather than by keeping two loops in step.
# ============================================================================================

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
        edge = EDGE_EXCLUSION_S
        inner = (self.time >= edge) & (self.time < self.duration - edge)
        ref = ideal_bandpass(self.x[record_id], fs)
        raw = {name: {k: [] for k in METRIC_NAMES} for name in filters}
        observations = []
        for d in range(self.n_draws):
            xc = contaminate(self.x[record_id], fs,
                                np.random.default_rng(draw_seed(record_id, d)))
            if observe is not None:
                observations.append(observe(xc))
            mains_in = mains_amplitude(xc, fs)          # shared by every filter on this draw
            for name, apply in filters.items():
                metrics = stress_metrics(xc, apply(xc), ref, fs, inner, mains_in)
                for key, value in metrics.items():
                    raw[name][key].append(float(value))
        return raw, observations

    def stress_test(self):
        """Butterworth, elliptic and Butterworth plus notch, paired across filters."""
        filters = self.stress_filters()
        raw, detected = {}, {}
        for i in self.ids:
            raw[i], hits = self.score_draws(i, filters,
                                            observe=lambda xc: mains_present(xc, self.fs)[0])
            detected[i] = int(sum(hits))
        self.mc_raw = raw
        out = {"n_draws": self.n_draws, "notch_detected": detected}
        for i in self.ids:
            gain_b, gain_e, gain_bn = (np.array(raw[i][name]["gain"]) for name in STRESS_NAMES)
            out[i] = {name: {k: summarise(v) for k, v in raw[i][name].items()}
                      for name in filters}
            out[i]["paired_B_minus_E"] = summarise(gain_b - gain_e)
            out[i]["B_wins_frac"] = float(np.mean(gain_b > gain_e))
            out[i]["paired_BN_minus_B"] = summarise(gain_bn - gain_b)
            out[i]["BN_wins_vs_E_frac"] = float(np.mean(gain_bn > gain_e))
            out[i]["BN_wins_vs_B_frac"] = float(np.mean(gain_bn > gain_b))
            for key, diff in (("B_minus_E", gain_b - gain_e), ("BN_minus_B", gain_bn - gain_b),
                              ("BN_minus_E", gain_bn - gain_e)):
                out[i][f"test_{key}"] = paired_test(diff)
        out["mains_offset"] = self.mains_offset_test()
        out["detect_weak_2uV"] = self.weak_mains_detection()
        return out

    def ripple_stress(self):
        """Stress-test SNR gain of the elliptic design at each passband ripple.

        Runs through score_draws, so every ripple is paired with the Butterworth on the same
        contamination as the main stress test.
        """
        designs = {f"{r:g}": ellip_bandpass(DESIGN_ORDER, self.fs, ripple_db=r)
                   for r in ELLIP_RIPPLES_DB}
        filters = {"butter": lambda z: self.zp(self.sos_butter, z),
                   **{key: (lambda z, sos=sos: self.zp(sos, z)) for key, sos in designs.items()}}
        out = {}
        for i in self.ids:
            raw, _ = self.score_draws(i, filters)
            butter = np.array(raw["butter"]["gain"])
            out[i] = {}
            for key in designs:
                gain = np.array(raw[key]["gain"])
                out[i][key] = {"gain": summarise(gain),
                               "butter_minus_ellip": summarise(butter - gain),
                               "test": paired_test(butter - gain),
                               "butter_wins_frac": float(np.mean(butter > gain))}
        return out

    def mains_offset_test(self):
        """Residual mains after the notch when the line drifts off 50 Hz."""
        x3 = self.x[TRANSIENT_RECORD]
        out = {}
        for mains_hz in MAINS_OFFSETS_HZ:
            residuals = []
            for d in range(OFFSET_DRAWS):
                xc = contaminate(x3, self.fs, np.random.default_rng(d), mains_hz=mains_hz,
                                    mains_amp=OFFSET_MAINS_UV)
                y = self.zp(self.sos_butter_notch, xc)
                residuals.append(amplitude_near(*self.spectrum(y, PLOT_NFFT), mains_hz, 0.3))
            out[mains_hz] = summarise(residuals)
        return out

    def weak_mains_detection(self):
        """Fraction of draws in which a weak mains line is still detected."""
        out = {}
        for i in self.ids:
            hits = [mains_present(contaminate(self.x[i], self.fs, np.random.default_rng(d),
                                                     mains_amp=WEAK_MAINS_UV), self.fs)[0]
                    for d in range(OFFSET_DRAWS)]
            out[i] = float(np.mean(hits))
        return out


# ============================================================================================
# analysis.py
# The analysis for each task, returning plain dictionaries that are saved as results.json.
#
# Study is assembled from one class per part of the brief, each in its own module:
#
#   study_base.py  recordings, detected events, designed filters and shared helpers
#   task1.py       FFT check, signal characteristics and aliasing (Task 1)
#   task2.py       convolution, time-frequency localisation, pulse timing and cost (Task 2)
#   task3.py       filter design study, notch and clean-record performance (Task 3)
#   stress.py      the Monte Carlo stress test and ripple study (Task 3), sharing one loop
#
# Values that plots need but the report does not (raw Monte Carlo draws, detected events) are
# kept on the Study object.
# ============================================================================================

class Study(Task1Signals, Task2TimeFrequency, Task3Filters, StressTest, StudyBase):
    """Every analysis in the report, run in order by run()."""

    def run(self):
        """Run every analysis in order and return the results dictionary."""
        results = self.fft_validation()
        results["chars"] = self.characteristics()
        results["peaks"] = self.peaks
        results["events"] = self.events_summary()
        results.update(self.aliasing())
        results["conv"] = self.convolution()
        results["complexity"] = self.complexity()
        results["loc"] = self.localisation()
        results["pulse_timing"] = self.pulse_timing()
        results["dwt_bands"] = self.dwt()
        results["min_orders_40_50_zero_phase"] = minimum_orders(self.fs)
        results["param"] = self.design_study()
        results["perf"] = self.performance()
        results.update(self.stopband_summary())
        results["notch"] = self.notch_study()
        results["stress_mc"] = self.stress_test()
        results["ripple_stress"] = self.ripple_stress()
        return results


# ============================================================================================
# benchmark.py
# Wall-clock timing. Kept out of results.json because runtimes change from run to run.
# ============================================================================================

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
    x3 = study.x[TRANSIENT_RECORD]
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
    short, long_ = STFT_WINDOWS
    timing = {
        "FFT": time_call(lambda: np.fft.rfft(x3)),
        "STFT_0.5s": time_call(lambda: stft_magnitude(x3, fs, short)),
        "STFT_2s": time_call(lambda: stft_magnitude(x3, fs, long_)),
        "CWT": time_call(lambda: study.cwt.magnitude(x3), 10),
        "CWT_direct_conv": time_once(lambda: study.cwt.magnitude(x3, method="conv")),
        "DWT": time_call(lambda: pywt.wavedec(x3, DWT_WAVELET, level=DWT_LEVEL)),
    }
    out["timing_ms"] = timing

    lengths = [n * 2 ** k for k in range(SCALING_LENGTHS)]
    runs = {k: [] for k in ("FFT", f"STFT_{short}", f"STFT_{long_}", "CWT_fft", "DWT")}
    rng = np.random.default_rng(7)
    for length in lengths:
        z = rng.standard_normal(length)
        runs["FFT"].append(time_call(lambda: np.fft.rfft(z), 50))
        runs[f"STFT_{short}"].append(time_call(lambda: stft_magnitude(z, fs, short), 10))
        runs[f"STFT_{long_}"].append(time_call(lambda: stft_magnitude(z, fs, long_), 5))
        runs["CWT_fft"].append(time_call(lambda: study.cwt.magnitude(z), 3))
        runs["DWT"].append(time_call(
            lambda: pywt.wavedec(z, DWT_WAVELET, level=DWT_LEVEL), 50))
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
    pad = int(PAD_SECONDS * fs)
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


# ============================================================================================
# plotting.py
# Figures for the report. Each function draws one figure from a finished Study.
# ============================================================================================

BLUE, ORANGE, GREEN, GREY, INK, PURPLE = (
    "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#4a3aa7")
SERIES = (GREEN, BLUE, ORANGE, PURPLE)
BAND_SYMBOLS = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
STYLE = {
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3",
    "grid.linewidth": 0.5, "lines.linewidth": 0.9, "font.family": "DejaVu Sans",
}
LEGEND_SMALL = dict(frameon=False, loc="lower left", bbox_to_anchor=(0.08, 0.0), fontsize=6,
                    title_fontsize=6, handlelength=1.0, labelspacing=0.25, borderaxespad=0.2)


def _save(fig, out_dir, name, dpi=300):
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=dpi)
    plt.close(fig)


def _zp_db(sos, fs, n_points=2 ** 15):
    w, h = frequency_response(sos, fs, n_points)
    return w, to_zero_phase_db(h)


def time_and_spectra(study, out_dir):
    """Figure 1: raw records with detected events shaded, and custom-FFT spectra."""
    fig, ax = plt.subplots(3, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios": [1.35, 1]})
    for row, i in enumerate(study.ids):
        a = ax[row, 0]
        a.plot(study.time, study.x[i], color=BLUE, lw=0.5)
        spans = [(p.onset_s, p.offset_s) for p in study.pulses[i]] + [study.bursts[i][:2]]
        for start, end in spans:
            a.axvspan(start, end, color=ORANGE, alpha=0.15, lw=0)
        a.set_ylabel(f"DS{i} (µV)")
        a.set_xlim(0, study.duration)
        a.set_ylim(-110, 150)
        f, amp = study.spectrum(study.x[i], PLOT_NFFT)
        b = ax[row, 1]
        for k, (lo, hi) in enumerate(BANDS.values()):
            b.axvspan(lo, hi, color=GREY, alpha=0.08 if k % 2 else 0.16, lw=0)
        b.plot(f, amp, color=BLUE, lw=0.7)
        b.set_xlim(0, CLINICAL_CEILING_HZ)
        b.set_ylim(0, 27)
        for fp, ap in study.main_peaks[i]:
            b.annotate(f"{fp:.1f}", (fp, ap), xytext=(2, 1), textcoords="offset points",
                       fontsize=6.5, color=INK)
        b.set_ylabel("Amplitude (µV)")
    ax[0, 1].set_title("Amplitude spectrum (custom radix-2 FFT)")
    ax[0, 0].set_title("Raw EEG, shaded = detected pulses and theta burst")
    for name, (lo, hi) in BANDS.items():
        ax[0, 1].text((lo + hi) / 2, 25.0, BAND_SYMBOLS[name], ha="center", fontsize=6.5,
                      color=GREY)
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 1].set_xlabel("Frequency (Hz)")
    _save(fig, out_dir, "fig1_time_freq.png")


def aliasing(study, results, out_dir):
    """Figure 2: naive against anti-aliased decimation of the alias record."""
    factor = ALIAS_DEMO_FACTOR
    fs_new, naive, filtered = study.decimated(factor)
    x = study.x[ALIAS_RECORD]
    demo = results["alias_demo"]
    beta, folded = study.alias_probe["beta"], study.alias_probe["folded"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    seg = (study.time >= 1) & (study.time < 1.5)
    t_new = np.arange(naive.size) / fs_new
    seg_new = (t_new >= 1) & (t_new < 1.5)
    ax[0].plot(study.time[seg], x[seg], color=BLUE, lw=0.8, label=f"{study.fs:g} Hz original")
    ax[0].plot(t_new[seg_new], naive[seg_new], "o-", color=ORANGE, ms=2.5, lw=0.8,
               label=f"{fs_new:g} Hz naive")
    ax[0].plot(t_new[seg_new], filtered[seg_new], "s-", color=GREEN, ms=2.5, lw=0.8,
               label=f"{fs_new:g} Hz anti-aliased")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("µV")
    ax[0].legend(loc="lower center", ncol=3, frameon=False, fontsize=5.6, handlelength=1.2,
                 columnspacing=0.8)
    ax[0].set_ylim(-80, 50)
    ax[0].set_title(f"(a) DS{ALIAS_RECORD} segment")
    f0, a0 = study.spectrum(x, PLOT_NFFT)
    ax[1].plot(f0, a0, color=BLUE, lw=0.7)
    ax[1].axvline(fs_new / 2, color=GREY, ls="--", lw=0.8)
    ax[1].text(fs_new / 2 + 0.5, 17, f"new Nyquist\n{fs_new / 2:g} Hz", fontsize=6.5, color=GREY)
    ax[1].set_xlim(0, 35)
    ax[1].set_ylim(0, 23)
    ax[1].set_title("(b) Original spectrum")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("µV")
    f_n, a_n = amplitude_spectrum(naive, fs_new, ALIAS_NFFT)
    f_a, a_a = amplitude_spectrum(filtered, fs_new, ALIAS_NFFT)
    ax[2].plot(f_n, a_n, color=ORANGE, lw=0.8, label="naive")
    ax[2].plot(f_a, a_a, color=GREEN, lw=0.8, label="anti-aliased")
    ax[2].annotate(f"{beta:.1f} Hz\nfolded to {folded:.1f}", (folded, demo["naive_amp_at_2p6"]),
                   xytext=(folded + 0.3, 19.3), fontsize=6.5, ha="center", va="bottom", color=INK)
    ax[2].set_xlim(0, fs_new / 2)
    ax[2].set_ylim(0, 25)
    ax[2].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=6.5)
    ax[2].set_title(f"(c) After ↓{factor}, fs = {fs_new:g} Hz")
    ax[2].set_xlabel("Frequency (Hz)")
    _save(fig, out_dir, "fig2_aliasing.png")


def convolution(study, out_dir):
    """Figure 3: kernels, their responses and their effect on the reference pulse."""
    kernels = study.kernels()
    x3 = study.x[TRANSIENT_RECORD]
    beta = study.probe(ALIAS_RECORD, "beta")
    labels = {name: (f"Gaussian σ={sigma * 1e3:.1f} ms" if sigma else f"Moving avg. {len(h)} taps")
              for name, (h, sigma) in kernels.items()}
    colours = dict(zip(kernels, (BLUE, GREEN, ORANGE)))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), gridspec_kw={"width_ratios": [0.8, 1, 1.25]})
    for name, (h, _) in kernels.items():
        lag_ms = (np.arange(len(h)) - (len(h) - 1) / 2) / study.fs * 1e3
        ax[0].plot(lag_ms, h, "o-", ms=1.8, color=colours[name], lw=0.8, label=labels[name])
        w, resp = signal.freqz(h, worN=4096, fs=study.fs)
        ax[1].plot(w, 20 * np.log10(np.abs(resp) + 1e-9), color=colours[name], lw=0.9)
    ax[0].set_xlabel("Lag (ms)")
    ax[0].set_title("(a) Kernels h[k]")
    ax[0].legend(frameon=False, fontsize=6, loc="upper right")
    ax[0].set_xlim(-40, 40)
    ax[1].axvline(beta, color=GREY, ls=":", lw=0.8)
    ax[1].text(beta + 0.6, -55, f"{beta:.1f} Hz β", fontsize=6.5, color=GREY)
    ax[1].set_ylim(-60, 3)
    ax[1].set_xlim(0, study.nyquist)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("|H| (dB)")
    ax[1].set_title("(b) Kernel frequency response")
    onset = study.pulse.onset_s
    zoom = study.mask((onset - 0.14, onset + 0.26))
    t = study.time[zoom]
    ax[2].plot(t, x3[zoom], color=GREY, lw=1.1, label="raw")
    styles = {"gauss40": ("Gauss 40 Hz", "-"), "gauss15": ("Gauss 15 Hz", "-"),
              f"ma{MOVING_AVERAGE_TAPS}": (f"MA-{MOVING_AVERAGE_TAPS}", "--")}
    for name, (h, _) in kernels.items():
        label, ls = styles[name]
        ax[2].plot(t, convolve_same(x3, h)[zoom], color=colours[name], lw=0.9, ls=ls,
                   label=label)
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("µV")
    ax[2].legend(frameon=False, ncol=2, fontsize=6, loc="lower right")
    ax[2].set_title(f"(c) DS{TRANSIENT_RECORD} pulse at {onset:.1f} s after convolution")
    _save(fig, out_dir, "fig3_convolution.png")


def time_frequency(study, out_dir):
    """Figure 4: two STFT windows against the Morlet CWT, detected pulses dotted."""
    x3 = study.x[TRANSIENT_RECORD]
    w, fr = study.cwt_map
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25), sharey=True)
    meshes = []
    for a, nper, tag in zip(ax[:2], STFT_WINDOWS, "ab"):
        f, ts, z = stft_magnitude(x3, study.fs, nper)
        meshes.append(a.pcolormesh(ts, f, 20 * np.log10(z + 1e-3), shading="gouraud",
                                   cmap="Blues", vmin=-15, vmax=25, rasterized=True))
        a.set_title(f"({tag}) STFT, {nper / study.fs:g} s (Δf = {study.fs / nper:g} Hz)")
        a.set_xlabel("Time (s)")
    cwt_mesh = ax[2].pcolormesh(study.time, fr, 20 * np.log10(w + 1e-3), shading="gouraud",
                                cmap="Blues", vmin=0, vmax=40, rasterized=True)
    # cone of influence: values within one edge e-folding time of either end are unreliable
    coi = study.cwt.cone_of_influence_s(fr)
    duration = study.duration
    ax[2].fill_betweenx(fr, 0, np.minimum(coi, duration / 2), color="white", alpha=0.55, lw=0)
    ax[2].fill_betweenx(fr, np.maximum(duration - coi, duration / 2), duration, color="white",
                        alpha=0.55, lw=0)
    ax[2].plot(np.minimum(coi, duration / 2), fr, color=INK, lw=0.7, ls="--")
    ax[2].plot(np.maximum(duration - coi, duration / 2), fr, color=INK, lw=0.7, ls="--")
    ax[2].set_xlim(0, duration)
    ax[2].set_title("(c) CWT, complex Morlet")
    ax[2].set_xlabel("Time (s)")
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_ylim(0, CLINICAL_CEILING_HZ)
    for a in ax:
        a.grid(False)
        for pulse in study.pulses[TRANSIENT_RECORD]:
            a.axvline(pulse.centre_s, color=ORANGE, lw=0.5, ls=":")
    for a, mesh, label in ((ax[1], meshes[1], "STFT (dB)"), (ax[2], cwt_mesh, "CWT (dB)")):
        cax = make_axes_locatable(a).append_axes("right", size="4%", pad=0.06)
        bar = fig.colorbar(mesh, cax=cax)
        bar.set_label(label, fontsize=7.5)
        bar.ax.tick_params(labelsize=7)
    _save(fig, out_dir, "fig4_tf.png")


def dwt(study, results, out_dir):
    """Appendix figure: single-level DWT reconstructions of the transient record."""
    x3 = study.x[TRANSIENT_RECORD]
    parts = dwt_single_level_reconstructions(x3)
    fig, ax = plt.subplots(len(parts) + 1, 1, figsize=(7.2, 5.2), sharex=True)
    ax[0].plot(study.time, x3, color=INK, lw=0.5)
    ax[0].set_ylabel(f"DS{TRANSIENT_RECORD}", fontsize=6.5)
    for a, (name, (lo, hi)), part in zip(ax[1:], results["dwt_bands"].items(), parts):
        a.plot(study.time, part, color=BLUE, lw=0.5)
        a.set_ylabel(f"{name} {lo:g}–{hi:g} Hz", fontsize=6)
    ax[-1].set_xlabel("Time (s)")
    _save(fig, out_dir, "figA_dwt.png", dpi=250)


def filter_parameters(study, out_dir):
    """Figure 5: effect of order and cut-off, and the passband detail of the two designs."""
    fs = study.fs
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1))
    sweeps = (
        ("(a) Butterworth, order N", "N", BUTTER_ORDERS,
         lambda o: butter_bandpass(o, fs)),
        (f"(b) Butterworth N={DESIGN_ORDER}, fc", "fc (Hz)", BUTTER_CUTOFFS,
         lambda c: butter_bandpass(DESIGN_ORDER, fs, high=c)),
        ("(c) Elliptic, order N", "N", ELLIP_ORDERS, lambda o: ellip_bandpass(o, fs)),
    )
    for a, (title, legend_title, values, design) in zip(ax, sweeps):
        for colour, value in zip(SERIES, values):
            w, db = _zp_db(design(value), fs)
            a.plot(w, db, color=colour, label=f"{value}")
        a.set_title(title)
        a.legend(title=legend_title, **LEGEND_SMALL)
        a.set_xlim(0, 100)
        a.set_ylim(-120, 5)
        a.set_xlabel("Frequency (Hz)")
        a.axvline(MAINS_HZ, color=GREY, ls=":", lw=0.7)
    ax[0].set_ylabel("Zero-phase |H|² (dB)")
    for sos, colour, label in ((study.sos_butter, BLUE, "Butterworth"),
                               (study.sos_ellip, ORANGE, "Elliptic")):
        w, db = _zp_db(sos, fs)
        ax[3].plot(w, db, color=colour, label=label)
    ax[3].set_xlim(0, CLINICAL_CEILING_HZ)
    ax[3].set_ylim(-6.5, 0.5)
    ax[3].set_title("(d) Passband detail")
    ax[3].set_xlabel("Frequency (Hz)")
    ax[3].legend(frameon=False, loc="lower center", fontsize=6, handlelength=1.0)
    _save(fig, out_dir, "fig5_filter_params.png")


def filter_performance(study, out_dir):
    """Figure 6: one stress-test draw, its PSD, pulse ringing and single-pass group delay."""
    fs, rec = study.fs, TRANSIENT_RECORD
    x3 = study.x[rec]
    xc = contaminate(x3, fs, np.random.default_rng(draw_seed(rec, 0)))
    y_b, y_e = study.zp(study.sos_butter, xc), study.zp(study.sos_ellip, xc)
    y_bn = study.zp(study.sos_butter_notch, xc)
    fig = plt.figure(figsize=(7.2, 3.3))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    a = fig.add_subplot(grid[0, :2])
    a.plot(study.time, xc, color=GREY, lw=0.4, label=f"DS{rec} + drift + 50 Hz + EMG")
    a.plot(study.time, y_b, color=BLUE, lw=0.6, label=f"Butterworth N={DESIGN_ORDER}")
    a.plot(study.time, y_e, color=ORANGE, lw=0.6, label=f"Elliptic N={DESIGN_ORDER}",
           alpha=0.9)
    a.set_xlim(0, study.duration)
    a.set_ylabel("µV")
    a.legend(frameon=False, ncol=3, loc="upper left", fontsize=6.5)
    a.set_title(f"(a) Stress test on contaminated DS{rec} (one draw, zero-phase)")
    a.set_ylim(-120, 190)
    b = fig.add_subplot(grid[0, 2])
    for z, colour, label, ls in ((xc, GREY, "contaminated", "-"), (y_b, BLUE, "Butterworth", "-"),
                                 (y_e, ORANGE, "Elliptic", "-"),
                                 (y_bn, GREEN, "Butterworth + notch", "--")):
        f, psd = signal.welch(z, fs, nperseg=512)
        b.semilogy(f, psd, color=colour, lw=0.7, ls=ls, label=label)
    b.set_xlim(0, study.nyquist)
    b.set_ylim(1e-6, 1e6)
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("PSD (µV²/Hz)")
    b.set_title("(b) Welch PSD")
    b.legend(frameon=False, fontsize=5.3, loc="upper right", ncol=2, columnspacing=0.6,
             handlelength=1.2)
    c = fig.add_subplot(grid[1, :2])
    onset = study.pulse.onset_s
    zoom = study.mask((onset - 0.19, onset + 0.36))
    c.plot(study.time[zoom], x3[zoom], color=GREY, lw=1.2, label=f"raw DS{rec}")
    c.plot(study.time[zoom], study.zp(study.sos_butter, x3)[zoom], color=BLUE, lw=0.9,
           label="Butterworth")
    c.plot(study.time[zoom], study.zp(study.sos_ellip, x3)[zoom], color=ORANGE, lw=0.9,
           label="Elliptic")
    c.set_xlabel("Time (s)")
    c.set_ylabel("µV")
    c.set_title(f"(c) Pulse artefact at {onset:.1f} s: ringing and pre-ringing (zero-phase)")
    c.legend(frameon=False, fontsize=6.5, loc="upper right")
    d = fig.add_subplot(grid[1, 2])
    band = np.linspace(0.6, CLINICAL_CEILING_HZ, 600)
    for sos, colour, label in ((study.sos_butter, BLUE, "Butterworth"),
                               (study.sos_ellip, ORANGE, "Elliptic")):
        d.plot(band, group_delay(sos, band, fs) / fs * 1e3, color=colour, label=label)
    d.set_yscale("log")
    d.set_xlabel("Frequency (Hz)")
    d.set_ylabel("Group delay (ms)")
    d.set_title("(d) Group delay, single pass")
    d.legend(frameon=False, fontsize=6)
    _save(fig, out_dir, "fig6_filter_perf.png")


def notch_and_stress(study, out_dir):
    """Figure 7: notch responses and the spread of stress-test SNR gain."""
    fs = study.fs
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [1, 1.4]})
    for q, colour in zip(NOTCH_Q_STUDY, (GREEN, BLUE, ORANGE)):
        w, db = _zp_db(notch(fs, q=q), fs, 2 ** 18)
        ax[0].plot(w, db, color=colour, label=f"Q = {q:g}")
    w, db = _zp_db(study.sos_butter_notch, fs, 2 ** 18)
    ax[0].plot(w, db, color=INK, lw=0.7, ls="--",
               label=f"Butterworth + notch (Q {NOTCH_Q:g})")
    ax[0].set_xlim(MAINS_HZ - 10, MAINS_HZ + 10)
    ax[0].set_ylim(-80, 3)
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("Zero-phase |H|² (dB)")
    ax[0].set_title(f"(a) {MAINS_HZ:g} Hz notch, zero-phase")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    jitter = np.random.default_rng(1)
    colours = (BLUE, ORANGE, GREEN)
    for j, i in enumerate(study.ids):
        for k, (name, colour) in enumerate(zip(STRESS_NAMES, colours)):
            gain = np.array(study.mc_raw[i][name]["gain"])
            x0 = j * 4 + k
            ax[1].scatter(x0 + jitter.uniform(-0.28, 0.28, gain.size), gain, s=1.5, color=colour,
                          alpha=0.35, lw=0)
            ax[1].errorbar(x0 + 0.42, gain.mean(), yerr=gain.std(ddof=1), fmt="o", ms=2.5,
                           color=INK, elinewidth=0.8, capsize=2)
    ax[1].set_xticks([j * 4 + 1 for j in range(len(study.ids))])
    ax[1].set_xticklabels([f"DS{i}" for i in study.ids])
    ax[1].set_ylabel("SNR gain (dB)")
    ax[1].set_title(f"(b) Stress test SNR gain, {study.n_draws} noise draws per record")
    handles = [Patch(color=c, alpha=0.7, label=n) for n, c in zip(STRESS_NAMES, colours)]
    handles.append(plt.Line2D([], [], color=INK, marker="o", ms=2.5, lw=0.8, label="mean ± SD"))
    ax[1].legend(handles=handles, frameon=False, fontsize=6.5, loc="upper center", ncol=4,
                 columnspacing=0.8, handlelength=1.2)
    ax[1].set_ylim(5, 36)
    _save(fig, out_dir, "fig7_notch_mc.png")


def filter_families(study, results, out_dir):
    """Appendix figure: each family at its minimum order for the zero-phase specification."""
    orders = results["min_orders_40_50_zero_phase"]
    names = {"butter": "Butterworth", "cheby1": "Chebyshev I", "cheby2": "Chebyshev II",
             "ellip": "Elliptic"}
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.4))
    for (family, label), colour in zip(names.items(), SERIES):
        sos = minimum_order_lowpass(family, orders[family], study.fs)
        w, db = _zp_db(sos, study.fs)
        ax.plot(w, db, color=colour, label=f"{label} (N={orders[family]})")
    ax.set_xlim(0, 100)
    ax.set_ylim(-100, 5)
    ax.axvspan(SPEC_PASS_HZ, SPEC_STOP_HZ, color=GREY, alpha=0.15, lw=0)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Zero-phase |H|² (dB)")
    ax.set_title(f"Minimum-order low-pass, zero-phase: ≤{SPEC_PASS_DB:g} dB at "
                 f"{SPEC_PASS_HZ:g} Hz, ≥{SPEC_STOP_DB:g} dB at "
                 f"{SPEC_STOP_HZ:g} Hz")
    _save(fig, out_dir, "figA_families.png", dpi=250)


def make_all(study, results, out_dir):
    """Draw every figure into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    time_and_spectra(study, out_dir)
    aliasing(study, results, out_dir)
    convolution(study, out_dir)
    time_frequency(study, out_dir)
    dwt(study, results, out_dir)
    filter_parameters(study, out_dir)
    filter_performance(study, out_dir)
    notch_and_stress(study, out_dir)
    filter_families(study, results, out_dir)


# ============================================================================================
# cli.py
# Command-line entry point: python -m eeg_a01 --data-dir dataset --out-dir output
#
# results.json holds every deterministic value. timings.json holds the wall-clock measurements,
# which vary between runs and machines, so the two never need to be compared together.
# ============================================================================================

def to_json(value):
    """Convert NumPy scalars for json.dump."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=to_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="eeg_a01", description="EEG analysis and filter design for CMP9780 A01.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"),
                        help="folder holding eeg_dataset_01.edf to eeg_dataset_03.edf")
    parser.add_argument("--out-dir", type=Path, default=Path("output"),
                        help="folder for results.json and figs/")
    parser.add_argument("--draws", type=int, default=N_DRAWS,
                        help="Monte Carlo draws per record (default %(default)s)")
    parser.add_argument("--no-figures", action="store_true", help="skip drawing figures")
    parser.add_argument("--no-benchmarks", action="store_true",
                        help="skip the wall-clock timings (timings.json is not written)")
    args = parser.parse_args(argv)
    if args.draws < 3:
        parser.error("--draws must be at least 3")
    if not args.data_dir.is_dir():
        parser.error(f"data folder {args.data_dir} does not exist")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    records = load_recordings(args.data_dir)
    study = Study(records, n_draws=args.draws)
    results = study.run()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "results.json", results)
    if not args.no_benchmarks:
        write_json(args.out_dir / "timings.json", run_benchmarks(study))
    if not args.no_figures:
        make_all(study, results, args.out_dir / "figs")
    print(f"done in {time.perf_counter() - started:.0f} s, results in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
