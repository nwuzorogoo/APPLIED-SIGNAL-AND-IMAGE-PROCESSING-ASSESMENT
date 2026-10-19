"""Shared fixtures: a synthetic three-record dataset that needs no EDF files."""
import numpy as np
import pytest

from eeg_a01.edf_io import ChannelCalibration, Recording

FS = 256.0
N = 2560
CALIBRATION = ChannelCalibration("EEG1", "uV", -200.0, 200.0, -32768, 32767)

# one line in each of delta, theta, alpha and beta, as in the supplied records
LINES_HZ = {1: (2.8, 6.7, 11.7, 29.4), 2: (2.6, 7.1, 9.4, 24.0), 3: (1.8, 5.1, 10.6, 18.1)}
PULSES = {1: [(400, 425)], 2: [(2240, 2265)], 3: [(765, 790), (2156, 2181)]}


def synthetic_record(record_id, seed=0):
    rng = np.random.default_rng(seed + record_id)
    t = np.arange(N) / FS
    x = rng.standard_normal(N) * 2.0
    for freq in LINES_HZ[record_id]:
        x += 15.0 * np.sin(2 * np.pi * freq * t)
    x += 30.0 * np.sin(2 * np.pi * 6.5 * t) * ((t > 4.4) & (t < 5.4))
    for onset, offset in PULSES[record_id]:
        x[onset:offset] += 90.0
    return Recording(record_id, x, FS, CALIBRATION)


@pytest.fixture(scope="session")
def synthetic_records():
    return {i: synthetic_record(i) for i in (1, 2, 3)}


@pytest.fixture(scope="session")
def synthetic_study(synthetic_records):
    from eeg_a01.analysis import Study
    study = Study(synthetic_records, n_draws=3)
    study.results = study.run()
    return study
