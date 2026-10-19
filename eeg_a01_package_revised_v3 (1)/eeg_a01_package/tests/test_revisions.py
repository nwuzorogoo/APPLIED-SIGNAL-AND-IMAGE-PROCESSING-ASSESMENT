"""Tests for the pulse removal, the cone of influence and the elliptic stopband measures."""
import numpy as np
import pytest

from eeg_a01 import config
from eeg_a01 import filters as flt
from eeg_a01.events import (Pulse, detect_pulses, detect_theta_burst, interpolate_pulses,
                            remove_pulses, step_height)
from eeg_a01.timefreq import default_cwt

FS = 256.0
N = 2560
T = np.arange(N) / FS


def burst_signal():
    """Steady 10.6 Hz line plus a 6.5 Hz burst from 4.4 to 5.4 s."""
    return 15 * np.sin(2 * np.pi * 10.6 * T) + 30 * np.sin(2 * np.pi * 6.5 * T) * ((T > 4.4) & (T < 5.4))


def with_pulse(x, onset, offset, height=90.0):
    y = x.copy()
    y[onset:offset] += height
    return y


# ---------------------------------------------------------------- pulse removal
def test_step_height_ignores_the_local_slope():
    x = 0.8 * np.arange(40.0)
    x[20:] += 50.0
    assert step_height(x, 20) == pytest.approx(50.0, abs=1e-9)


def test_remove_pulses_recovers_the_eeg_under_the_plateau():
    clean = burst_signal()
    onset, offset = int(4.35 * FS), int(4.45 * FS)       # pulse straddles the burst onset
    pulsed = with_pulse(clean, onset, offset)
    pulses = detect_pulses(pulsed, FS)
    assert [(p.onset, p.offset) for p in pulses] == [(onset, offset)]
    repaired = remove_pulses(pulsed, pulses)
    assert np.max(np.abs(repaired - clean)) < 3.0       # within a few µV of the true EEG


def test_interpolate_pulses_draws_a_straight_line():
    x = np.zeros(100)
    x[40:50] = 80.0
    y = interpolate_pulses(x, [Pulse(onset=40, offset=50, fs=FS)])
    assert np.allclose(y, 0.0)


def test_burst_measured_after_removing_an_overlapping_pulse():
    clean = burst_signal()
    truth = detect_theta_burst(clean, FS)
    pulsed = with_pulse(clean, int(4.35 * FS), int(4.45 * FS))
    pulses = detect_pulses(pulsed, FS)
    repaired = detect_theta_burst(pulsed, FS, pulses=pulses)
    assert repaired[2] == pytest.approx(truth[2], abs=0.02)
    assert repaired[0] == pytest.approx(truth[0], abs=0.02)


def test_pulse_near_the_record_edge_is_skipped_not_fatal():
    clean = burst_signal()
    edge_pulse = Pulse(onset=2, offset=27, fs=FS)               # only 2 samples before the rise
    late_pulse = Pulse(onset=N - 26, offset=N - 1, fs=FS)       # only 1 sample after the fall
    inner = Pulse(onset=int(4.35 * FS), offset=int(4.45 * FS), fs=FS)
    pulsed = clean.copy()
    for p in (edge_pulse, late_pulse, inner):
        pulsed[p.onset:p.offset] += 90.0
    with pytest.warns(RuntimeWarning, match="record edge"):
        repaired = remove_pulses(pulsed, [edge_pulse, late_pulse, inner])
    assert np.array_equal(repaired[:30], pulsed[:30])           # edge pulses left untouched
    assert np.array_equal(repaired[-30:], pulsed[-30:])
    middle = slice(inner.onset - 10, inner.offset + 10)
    assert np.max(np.abs(repaired[middle] - clean[middle])) < 3.0   # inner pulse still removed
    with pytest.warns(RuntimeWarning):
        span = detect_theta_burst(pulsed, FS, pulses=[edge_pulse, inner])
    assert span[2] > 0


def test_pulse_removal_is_a_no_op_without_pulses():
    x = burst_signal()
    assert np.array_equal(remove_pulses(x, []), x)
    assert detect_theta_burst(x, FS, pulses=[]) == detect_theta_burst(x, FS)


# ---------------------------------------------------------------- cone of influence
def test_cone_of_influence_follows_sqrt_bandwidth_over_frequency():
    cwt = default_cwt(FS)
    expected = np.sqrt(config.CWT_BANDWIDTH) * config.CWT_CENTRE / np.array([1.0, 6.5, 35.0])
    assert np.allclose(cwt.cone_of_influence_s([1.0, 6.5, 35.0]), expected)
    assert cwt.cone_of_influence_s(6.5) == pytest.approx(0.188, abs=1e-3)


def test_inside_cone():
    cwt = default_cwt(FS)
    assert cwt.inside_cone(6.5, 1.0, 9.0, 10.0)
    assert not cwt.inside_cone(6.5, 0.1, 2.0, 10.0)      # starts within 0.19 s of the edge
    assert not cwt.inside_cone(1.0, 1.0, 9.0, 10.0)      # 1.22 s cone at 1 Hz


# ---------------------------------------------------------------- elliptic stopband
def test_elliptic_reaches_its_stop_level_at_the_reported_edge():
    sos = flt.ellip_bandpass(4, FS)
    edge = flt.stopband_edge_hz(sos, FS, config.ELLIP_STOP_DB)
    assert 55.0 < edge < 61.0                            # 50 Hz is still in the transition band
    single_pass_50 = flt.zero_phase_db_at(sos, FS, 50.0) / config.ZERO_PHASE_FACTOR
    assert -config.ELLIP_STOP_DB < single_pass_50 < -10.0
    zeros = flt.transmission_zeros_hz(sos, FS)
    assert zeros and zeros[0] > edge                     # first null lies just above the edge


def test_butterworth_has_no_finite_transmission_zeros_in_the_stopband():
    assert flt.transmission_zeros_hz(flt.butter_bandpass(4, FS), FS) == []


def test_more_ripple_buys_more_attenuation_at_50_hz():
    att = [flt.zero_phase_db_at(flt.ellip_bandpass(4, FS, ripple_db=r), FS, 50.0)
           for r in config.ELLIP_RIPPLES_DB]
    assert all(a > b for a, b in zip(att, att[1:]))
