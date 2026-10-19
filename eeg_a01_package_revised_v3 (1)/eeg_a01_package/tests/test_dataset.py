"""Checks on the supplied recordings. Skipped when the dataset folder is not present.

Set EEG_DATA_DIR to point at the folder holding the EDF files (default: ./dataset).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from eeg_a01.edf_io import load_recordings, read_calibration
from eeg_a01.events import detect_pulses

DATA_DIR = Path(os.environ.get("EEG_DATA_DIR", "dataset"))
pytestmark = pytest.mark.skipif(not (DATA_DIR / "eeg_dataset_01.edf").is_file(),
                                reason="dataset not available")


@pytest.fixture(scope="module")
def records():
    return load_recordings(DATA_DIR)


def test_header_calibration_is_read_from_file():
    cal = read_calibration(DATA_DIR / "eeg_dataset_01.edf")
    assert cal.unit == "uV"
    assert cal.full_scale == pytest.approx(200.0)
    assert cal.lsb == pytest.approx(400 / 65535)


def test_records_share_rate_and_length(records):
    assert {r.fs for r in records.values()} == {256.0}
    assert {r.n_samples for r in records.values()} == {2560}


def test_samples_lie_on_the_quantisation_grid(records):
    for r in records.values():
        counts = (r.data - r.calibration.physical_min) / r.calibration.lsb
        assert np.max(np.abs(counts - np.round(counts))) < 1e-3


def test_pulses_detected_in_every_record(records):
    found = {i: len(detect_pulses(r.data, r.fs)) for i, r in records.items()}
    assert found == {1: 1, 2: 1, 3: 3}


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recordings(tmp_path)
