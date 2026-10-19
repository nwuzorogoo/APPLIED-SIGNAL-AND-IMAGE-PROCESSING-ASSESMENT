"""Unit tests for the building blocks, using synthetic signals with known answers."""
import numpy as np
import pytest

from eeg_a01 import config
from eeg_a01 import convolution as conv
from eeg_a01 import evaluation as ev
from eeg_a01 import filters as flt
from eeg_a01.events import PulseWindowSet, detect_pulses, detect_theta_burst
from eeg_a01.spectral import (alias_frequency, amplitude_near, amplitude_spectrum, band_power,
                              dominant_peak, fft_radix2, half_maximum_span)

FS = 256.0


def tone(freq, amp=1.0, n=2560, fs=FS, phase=0.0):
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / fs + phase)


# ---------------------------------------------------------------- FFT and spectrum
@pytest.mark.parametrize("n", [1, 2, 8, 1024, 4096])
def test_fft_matches_numpy(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    np.testing.assert_allclose(fft_radix2(x), np.fft.fft(x), atol=1e-9)


@pytest.mark.parametrize("n", [0, 3, 2560])
def test_fft_rejects_non_power_of_two(n):
    with pytest.raises(ValueError):
        fft_radix2(np.ones(n))


def test_spectrum_reads_sine_amplitude():
    freqs, amp = amplitude_spectrum(tone(10.0, amp=20.0), FS, nfft=2 ** 16)
    assert freqs[np.argmax(amp)] == pytest.approx(10.0, abs=0.01)
    assert amp.max() == pytest.approx(20.0, rel=0.01)


def test_spectrum_rejects_short_or_odd_nfft():
    with pytest.raises(ValueError):
        amplitude_spectrum(np.ones(2560), FS, nfft=2048)
    with pytest.raises(ValueError):
        amplitude_spectrum(np.ones(2560), FS, nfft=3000)


def test_amplitude_near_empty_window_raises():
    freqs, amp = amplitude_spectrum(tone(10.0), FS)
    with pytest.raises(ValueError):
        amplitude_near(freqs, amp, 500.0)


def test_band_power_concentrates_on_the_tone():
    x = tone(10.0, amp=5.0)
    assert band_power(x, FS, 8, 13) / band_power(x, FS, 0, FS / 2) > 0.99


def test_dominant_peak_picks_largest_in_band():
    peaks = [(2.0, 9.0), (20.0, 4.0), (25.0, 12.0)]
    assert dominant_peak(peaks, (13, 30)) == 25.0
    with pytest.raises(ValueError):
        dominant_peak(peaks, (30, 45))


@pytest.mark.parametrize("f0, fs_new, expected", [(29.39, 32.0, 2.61), (10.0, 32.0, 10.0),
                                                  (29.39, 51.2, 21.81), (16.0, 32.0, 16.0)])
def test_alias_frequency(f0, fs_new, expected):
    assert alias_frequency(f0, fs_new) == pytest.approx(expected)


def test_alias_frequency_matches_sampling():
    """A tone sampled below Nyquist shows up at the predicted alias."""
    fs_new = 32.0
    x = tone(29.39, n=2560 // 8, fs=fs_new)
    freqs, amp = amplitude_spectrum(x, fs_new, nfft=2 ** 14)
    assert freqs[np.argmax(amp)] == pytest.approx(alias_frequency(29.39, fs_new), abs=0.02)


def test_half_maximum_span_of_gaussian():
    axis = np.linspace(-1, 1, 2001)
    sigma = 0.1
    start, end, width = half_maximum_span(axis, np.exp(-axis ** 2 / (2 * sigma ** 2)))
    assert width == pytest.approx(2 * np.sqrt(2 * np.log(2)) * sigma, abs=2e-3)
    assert start == pytest.approx(-end)


# ---------------------------------------------------------------- convolution
def test_convolve_same_matches_numpy():
    rng = np.random.default_rng(3)
    x, h = rng.standard_normal(500), rng.standard_normal(15)
    np.testing.assert_allclose(conv.convolve_same(x, h), np.convolve(x, h, "same"), atol=1e-12)


def test_convolve_same_rejects_empty():
    with pytest.raises(ValueError):
        conv.convolve_same(np.array([]), np.ones(3))


@pytest.mark.parametrize("cutoff", [15.0, 40.0])
def test_gaussian_kernel_is_3db_at_cutoff(cutoff):
    h = conv.gaussian_kernel(conv.gaussian_sigma_for_cutoff(cutoff), FS)
    assert h.sum() == pytest.approx(1.0)
    response = abs(np.sum(h * np.exp(-2j * np.pi * cutoff * np.arange(h.size) / FS)))
    assert 20 * np.log10(response) == pytest.approx(-3.01, abs=0.05)


def test_rise_time_of_linear_ramp():
    ramp = np.r_[np.zeros(10), np.linspace(0, 1, 11), np.ones(10)]
    # the ramp takes 10 samples, so 10 to 90 % takes 8 samples
    assert conv.rise_time_ms(ramp, FS, 10, 20) == pytest.approx(8 / FS * 1e3, rel=0.01)


# ---------------------------------------------------------------- filters
def test_butterworth_zero_phase_is_6db_at_cutoff():
    sos = flt.butter_bandpass(4, FS)
    assert flt.zero_phase_db_at(sos, FS, config.PASSBAND[1]) == pytest.approx(-6.02, abs=0.05)
    assert flt.zero_phase_db_at(sos, FS, 10.0) == pytest.approx(0.0, abs=0.01)


def test_zero_phase_filtering_does_not_shift_a_tone():
    x = tone(10.0, amp=10.0)
    y = flt.zero_phase(flt.butter_bandpass(4, FS), x, FS)
    inner = slice(512, -512)
    np.testing.assert_allclose(y[inner], x[inner], atol=0.2)
    lags = np.correlate(y[inner], x[inner], "full")
    assert np.argmax(lags) == x[inner].size - 1          # best match at zero lag


def test_notch_nulls_mains_and_spares_eeg():
    sos = flt.notch(FS)
    assert flt.zero_phase_db_at(sos, FS, 50.0) < -100
    assert flt.zero_phase_db_at(sos, FS, 30.0) > -0.05
    y = flt.zero_phase(sos, tone(50.0, amp=20.0) + tone(10.0, amp=5.0), FS)
    freqs, amp = amplitude_spectrum(y, FS, nfft=2 ** 14)
    assert amplitude_near(freqs, amp, 50.0) < 0.1
    assert amplitude_near(freqs, amp, 10.0) == pytest.approx(5.0, rel=0.02)


def test_mains_detector():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(2560) * 10
    assert not flt.mains_present(clean, FS)[0]
    assert flt.mains_present(clean + tone(50.0, amp=5.0), FS)[0]


def test_filters_are_stable():
    for sos in (flt.butter_bandpass(8, FS), flt.ellip_bandpass(5, FS), flt.notch(FS)):
        assert flt.max_pole_radius(sos) < 1.0


# ---------------------------------------------------------------- events
def synthetic_pulse_record():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(2560) * 5
    x[1000:1025] += 90.0
    return x


def test_detect_pulses_finds_onset_and_offset():
    pulses = detect_pulses(synthetic_pulse_record(), FS)
    assert len(pulses) == 1
    assert (pulses[0].onset, pulses[0].offset) == (1000, 1025)


def test_detect_pulses_ignores_lone_edges():
    x = synthetic_pulse_record()
    x[2000:] -= 80.0          # a single step, no matching rise
    assert len(detect_pulses(x, FS)) == 1


def test_pulse_windows_are_anchored_on_the_pulse():
    pulse = detect_pulses(synthetic_pulse_record(), FS)[0]
    windows = PulseWindowSet(pulse)
    start, end = windows.peak
    assert start < pulse.onset_s < pulse.offset_s < end
    rise_start, rise_end = windows.rise_samples()
    assert rise_start < pulse.onset < rise_end


def test_detect_theta_burst():
    t = np.arange(2560) / FS
    x = np.random.default_rng(1).standard_normal(2560)
    x += 30 * np.sin(2 * np.pi * 6.5 * t) * ((t > 4) & (t < 6))
    start, end, _ = detect_theta_burst(x, FS)
    assert start == pytest.approx(4.0, abs=0.15)
    assert end == pytest.approx(6.0, abs=0.15)


# ---------------------------------------------------------------- evaluation
def test_contaminate_is_reproducible():
    x = tone(10.0)
    a = ev.contaminate(x, FS, np.random.default_rng(42))
    b = ev.contaminate(x, FS, np.random.default_rng(42))
    c = ev.contaminate(x, FS, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_contaminate_fixed_mains_amplitude():
    x = np.zeros(2560)
    y = ev.contaminate(x, FS, np.random.default_rng(0), mains_amp=15.0)
    freqs, amp = amplitude_spectrum(y, FS, nfft=2 ** 14)
    assert amplitude_near(freqs, amp, 50.0) == pytest.approx(15.0, rel=0.05)


def test_paired_test_interval_and_effect_size():
    d = np.random.default_rng(2).normal(1.0, 0.5, 400)
    result = ev.paired_test(d)
    low, high = result["ci95"]
    assert low < 1.0 < high
    assert result["dz"] == pytest.approx(2.0, rel=0.15)
    assert result["p_t"] < 1e-10


def test_paired_test_needs_pairs():
    with pytest.raises(ValueError):
        ev.paired_test([1.0, 2.0])
