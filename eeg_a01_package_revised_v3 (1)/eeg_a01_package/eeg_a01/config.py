"""Analysis settings in one place.

Everything that depends on the recordings themselves (sampling rate, record length, amplitude
range, spectral peak frequencies, pulse and burst timings) is read or detected at run time. The
values below are design choices: band definitions, filter specifications, analysis windows
expressed relative to detected events, and Monte Carlo settings.
"""
from dataclasses import dataclass

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
