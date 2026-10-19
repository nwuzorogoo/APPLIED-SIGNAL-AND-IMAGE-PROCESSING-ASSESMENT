"""The shared stress-test loop and the split of the analysis into task modules."""
import numpy as np

from eeg_a01 import analysis
from eeg_a01.stress import StressTest
from eeg_a01.study_base import StudyBase
from eeg_a01.task1 import Task1Signals
from eeg_a01.task2 import Task2TimeFrequency
from eeg_a01.task3 import Task3Filters


def test_study_is_assembled_from_the_task_modules():
    for part in (Task1Signals, Task2TimeFrequency, Task3Filters, StressTest, StudyBase):
        assert issubclass(analysis.Study, part)
    # the old import path still works for code that used it
    assert analysis.STRESS_NAMES == ("Butterworth", "Elliptic", "Butterworth+notch")


def test_score_draws_gives_every_filter_the_same_contamination(synthetic_study):
    study = synthetic_study
    same = lambda z: study.zp(study.sos_butter, z)                      # noqa: E731
    raw, seen = study.score_draws(1, {"a": same, "b": same}, observe=lambda xc: float(xc[0]))
    assert raw["a"] == raw["b"]                     # identical draws, identical scores
    assert len(seen) == study.n_draws and len(set(seen)) == study.n_draws


def test_ripple_study_is_paired_with_the_main_stress_test(synthetic_study):
    """The ripple study's Butterworth gains are the main test's, draw for draw."""
    study = synthetic_study
    main = np.array(study.mc_raw[1]["Butterworth"]["gain"])
    raw, _ = study.score_draws(1, {"butter": lambda z: study.zp(study.sos_butter, z)})
    assert np.array_equal(main, np.array(raw["butter"]["gain"]))
    ripple = study.results["ripple_stress"][1]
    for entry in ripple.values():
        assert np.isclose(entry["butter_minus_ellip"]["mean"],
                          main.mean() - entry["gain"]["mean"])
