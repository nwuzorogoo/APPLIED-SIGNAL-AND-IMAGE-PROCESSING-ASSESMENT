"""Tests for the STFT, CWT and DWT routines and the operation counts."""
import numpy as np
import pytest

from eeg_a01 import config
from eeg_a01.events import Pulse, edge_times
from eeg_a01.timefreq import (MorletCWT, default_cwt, dwt_band_edges,
                              dwt_single_level_reconstructions, operation_counts,
                              stft_magnitude)

FS = 256.0
N = 2560
TIME = np.arange(N) / FS


def tone(freq, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * TIME)


# ---------------------------------------------------------------- STFT
@pytest.mark.parametrize("window", config.STFT_WINDOWS)
def test_stft_axes_and_peak(window):
    freqs, times, z = stft_magnitude(tone(10.0, 5.0), FS, window)
    assert freqs[1] - freqs[0] == pytest.approx(FS / window)
    assert times[1] - times[0] == pytest.approx(config.STFT_HOP / FS)
    assert freqs[np.argmax(z.mean(axis=1))] == pytest.approx(10.0, abs=FS / window)


def test_stft_short_window_localises_a_burst_better():
    burst = tone(10.0, 5.0) * ((TIME > 4.0) & (TIME < 4.5))
    widths = {}
    for window in config.STFT_WINDOWS:
        freqs, times, z = stft_magnitude(burst, FS, window)
        envelope = z[np.argmin(np.abs(freqs - 10.0))]
        widths[window] = np.ptp(times[envelope > envelope.max() / 2])
    short, long_ = config.STFT_WINDOWS
    assert widths[short] < widths[long_]


@pytest.mark.parametrize("hop", [0, 600])
def test_stft_rejects_bad_hop(hop):
    with pytest.raises(ValueError):
        stft_magnitude(tone(10.0), FS, 512, hop=hop)


# ---------------------------------------------------------------- CWT
@pytest.fixture(scope="module")
def cwt():
    return default_cwt(FS)


def test_cwt_grid(cwt):
    start, stop, count = config.CWT_FREQS
    assert cwt.freqs[0] == start and cwt.freqs[-1] == stop and cwt.freqs.size == count
    assert np.all(np.diff(cwt.scales) < 0)          # higher frequency, smaller scale
    assert np.all(cwt.support == (cwt.scales * 16 + 1).astype(int))


@pytest.mark.parametrize("freq", [6.5, 20.0])
def test_cwt_peaks_at_the_tone_frequency(cwt, freq):
    magnitude, freqs = cwt.magnitude(tone(freq, 5.0))
    centre = magnitude[:, N // 4:3 * N // 4].mean(axis=1)
    assert freqs[np.argmax(centre)] == pytest.approx(freq, rel=0.1)


def test_batched_cwt_finds_the_same_line(cwt):
    transform = cwt.batched(N)
    magnitude = transform(tone(12.0, 5.0))
    assert magnitude.shape[0] == cwt.freqs.size
    centre = magnitude[:, N // 4:3 * N // 4].mean(axis=1)
    assert cwt.freqs[np.argmax(centre)] == pytest.approx(12.0, rel=0.1)


def test_cwt_time_spread_shrinks_with_frequency(cwt):
    assert cwt.sigma_t_ms(35.0) < cwt.sigma_t_ms(6.5)
    for freq in config.CWT_SIGMA_PROBES_HZ:
        # Morlet sits at a fixed time-frequency product, 1 / (2 pi) for this definition
        product = cwt.sigma_t_ms(freq) / 1e3 * cwt.sigma_f_hz(freq)
        assert product == pytest.approx(1 / (2 * np.pi))


def test_cwt_resolves_both_pulse_edges(cwt):
    x = np.zeros(N)
    x[768:793] = 90.0
    magnitude, freqs = cwt.magnitude(x)
    envelope = magnitude[(freqs >= 30) & (freqs <= 45)].mean(0)
    pulse = Pulse(768, 793, FS)
    onset, offset, n_peaks = edge_times(TIME, envelope, pulse)
    assert n_peaks == 2
    assert abs(onset - pulse.onset_s) < 0.01
    assert abs(offset - pulse.offset_s) < 0.01


def test_custom_wavelet_bandwidth_changes_spread():
    narrow = MorletCWT(FS, [10.0], bandwidth=0.5)
    wide = MorletCWT(FS, [10.0], bandwidth=3.0)
    assert narrow.sigma_t_ms(10.0) < wide.sigma_t_ms(10.0)


# ---------------------------------------------------------------- edge times
def test_edge_times_merged_blob_gives_one_peak():
    axis = np.linspace(0, 2, 401)
    envelope = np.exp(-((axis - 1.0) / 0.05) ** 2)
    onset, offset, n_peaks = edge_times(axis, envelope, Pulse(240, 272, FS))
    assert n_peaks == 1 and onset == offset


def test_edge_times_needs_matching_shapes():
    with pytest.raises(ValueError):
        edge_times(np.arange(10.0), np.arange(9.0), Pulse(2, 4, FS))


# ---------------------------------------------------------------- DWT and counts
def test_dwt_bands_tile_zero_to_nyquist():
    bands = list(dwt_band_edges(FS).values())
    assert bands[0][0] == 0.0 and bands[-1][1] == FS / 2
    for (_, high), (low, _) in zip(bands, bands[1:]):
        assert high == low


def test_dwt_parts_sum_to_the_signal():
    x = np.random.default_rng(4).standard_normal(N)
    parts = dwt_single_level_reconstructions(x)
    assert len(parts) == config.DWT_LEVEL + 1
    np.testing.assert_allclose(np.sum(parts, axis=0), x, atol=1e-9)


def test_operation_counts_order(cwt):
    counts = operation_counts(N, cwt.support)
    assert counts["FFT"] == pytest.approx(N * np.log2(N))
    assert counts["FFT"] < counts["STFT_128"] < counts["STFT_512"]
    # the DWT is linear in N, so doubling N exactly doubles its count
    assert operation_counts(2 * N, cwt.support)["DWT"] == pytest.approx(2 * counts["DWT"])
    assert counts["STFT_512"] < counts["CWT_fft"] < counts["CWT_direct"]
