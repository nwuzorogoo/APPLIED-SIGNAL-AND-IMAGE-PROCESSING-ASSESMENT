"""Regression test: the supplied recordings must still give the numbers quoted in the report.

expected_results.json holds every deterministic section of results.json from the reference
run except the Monte Carlo summary, which this test shortens to three draws for speed.
Skipped when the dataset is not available (set EEG_DATA_DIR to its folder).
"""
import json
import math
import os
from pathlib import Path

import pytest

from eeg_a01.analysis import Study
from eeg_a01.cli import to_json
from eeg_a01.edf_io import load_recordings

DATA_DIR = Path(os.environ.get("EEG_DATA_DIR", "dataset"))
EXPECTED = Path(__file__).parent / "data" / "expected_results.json"
pytestmark = pytest.mark.skipif(not (DATA_DIR / "eeg_dataset_01.edf").is_file(),
                                reason="dataset not available")


def differences(expected, actual, path=""):
    """List every path where actual departs from expected beyond floating-point noise."""
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return [f"{path}: keys {sorted(set(expected) ^ set(actual))}"]
        return [d for k in expected for d in differences(expected[k], actual[k], f"{path}/{k}")]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        return [d for k, (e, a) in enumerate(zip(expected, actual))
                for d in differences(e, a, f"{path}[{k}]")]
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        ok = math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12)
        return [] if ok else [f"{path}: {expected} != {actual}"]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


@pytest.fixture(scope="module")
def results():
    study = Study(load_recordings(DATA_DIR), n_draws=3)
    # round-trip through JSON so keys and types match the stored file
    return json.loads(json.dumps(study.run(), default=to_json))


def test_report_numbers_are_reproduced(results):
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    problems = differences(expected, {k: results[k] for k in expected})
    assert not problems, "\n".join(problems[:20])
