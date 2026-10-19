"""Direct convolution and the smoothing kernels used in Task 2a."""
import numpy as np


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
