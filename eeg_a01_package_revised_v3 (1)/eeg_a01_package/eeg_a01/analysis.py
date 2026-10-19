"""The analysis for each task, returning plain dictionaries that are saved as results.json.

Study is assembled from one class per part of the brief, each in its own module:

  study_base.py  recordings, detected events, designed filters and shared helpers
  task1.py       FFT check, signal characteristics and aliasing (Task 1)
  task2.py       convolution, time-frequency localisation, pulse timing and cost (Task 2)
  task3.py       filter design study, notch and clean-record performance (Task 3)
  stress.py      the Monte Carlo stress test and ripple study (Task 3), sharing one loop

Values that plots need but the report does not (raw Monte Carlo draws, detected events) are
kept on the Study object.
"""
from . import filters as flt
from .names import FILTER_NAMES, STRESS_NAMES, freq_key
from .stress import StressTest
from .study_base import StudyBase
from .task1 import Task1Signals
from .task2 import Task2TimeFrequency
from .task3 import Task3Filters

__all__ = ["Study", "FILTER_NAMES", "STRESS_NAMES", "freq_key"]


class Study(Task1Signals, Task2TimeFrequency, Task3Filters, StressTest, StudyBase):
    """Every analysis in the report, run in order by run()."""

    def run(self):
        """Run every analysis in order and return the results dictionary."""
        results = self.fft_validation()
        results["chars"] = self.characteristics()
        results["peaks"] = self.peaks
        results["events"] = self.events_summary()
        results.update(self.aliasing())
        results["conv"] = self.convolution()
        results["complexity"] = self.complexity()
        results["loc"] = self.localisation()
        results["pulse_timing"] = self.pulse_timing()
        results["dwt_bands"] = self.dwt()
        results["min_orders_40_50_zero_phase"] = flt.minimum_orders(self.fs)
        results["param"] = self.design_study()
        results["perf"] = self.performance()
        results.update(self.stopband_summary())
        results["notch"] = self.notch_study()
        results["stress_mc"] = self.stress_test()
        results["ripple_stress"] = self.ripple_stress()
        return results
