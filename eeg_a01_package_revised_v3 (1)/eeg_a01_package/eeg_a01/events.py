"""Detection of the transient events used in the analysis, so no event time is typed in."""
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import config
from .spectral import half_maximum_span


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


def detect_pulses(x, fs, edge_sigmas=config.PULSE_EDGE_SIGMAS,
                  duration=config.PULSE_DURATION_S):
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


def edge_times(axis, envelope, pulse, half_width=config.EDGE_SEARCH_HALF_WIDTH_S,
               prominence=config.EDGE_PEAK_PROMINENCE):
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


def step_height(x, k, width=config.PULSE_FIT_SAMPLES):
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


def measurable(pulse, n_samples, width=config.PULSE_FIT_SAMPLES):
    """True when both edges of the pulse have width samples on each side to fit."""
    return pulse.onset - width >= 0 and pulse.offset + width <= n_samples


def remove_pulses(x, pulses, width=config.PULSE_FIT_SAMPLES):
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


def detect_theta_burst(x, fs, band=config.THETA_BURST_BAND, pulses=None):
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
    """Analysis windows anchored on one detected pulse (see config.PulseWindows)."""

    def __init__(self, pulse, margins=config.PULSE_WINDOWS):
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
