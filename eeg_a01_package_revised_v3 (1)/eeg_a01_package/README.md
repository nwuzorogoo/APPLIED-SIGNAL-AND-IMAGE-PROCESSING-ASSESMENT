# eeg_a01

Code for CMP9780 Assessment 1, EEG signal analysis and digital filter design. One command
regenerates every figure and number in the report.

## Setup

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.13 was used. The versions in `requirements.txt` are the ones that produced the report.

## Run

```
python -m eeg_a01 --data-dir dataset --out-dir output
```

| Option | Meaning |
| --- | --- |
| `--data-dir` | folder with `eeg_dataset_01.edf` to `eeg_dataset_03.edf` (default `dataset`) |
| `--out-dir` | where `results.json` and `figs/` are written (default `output`) |
| `--draws` | Monte Carlo draws per record in the stress test (default 200) |
| `--no-figures` | compute results only |
| `--no-benchmarks` | skip the wall-clock timings, so `timings.json` is not written |

Two output files keep fixed and variable numbers apart. `results.json` holds every
deterministic value, and two runs on any machine give the same file because each random draw
has a fixed seed. `timings.json` holds the runtimes and the machine they were measured on, and
these change from run to run. Their keys follow the same paths, so `benchmark.merge` can join
them when both are needed.

A full run takes about a minute, mostly the benchmarks and the 200-draw stress test. A quick
check (`--draws 3 --no-benchmarks --no-figures`) takes a few seconds.

## Tests

```
pytest                                   # unit and end-to-end tests on synthetic data
EEG_DATA_DIR=dataset pytest              # also check the supplied recordings
```

| File | What it checks |
| --- | --- |
| `test_signal_processing.py` | FFT, spectrum, convolution, filters, detectors and statistics against known answers |
| `test_timefreq.py` | STFT axes and resolution, CWT frequency grid and edge timing, DWT reconstruction, operation counts |
| `test_pipeline.py` | the whole analysis, figures, benchmarks and command line on a synthetic three-record dataset |
| `test_dataset.py` | header calibration, quantisation grid and pulse detection on the supplied files |
| `test_revisions.py` | pulse subtraction before burst timing (including pulses too close to an edge, which are skipped), the CWT cone of influence, the elliptic stopband edge and the ripple trade-off |
| `test_stress_loop.py` | the task split and the shared stress loop, which pairs the ripple study with the main stress test draw for draw |
| `test_regression.py` | every deterministic number in the report against `tests/data/expected_results.json` |

The tests cover 98 % of the package (measured with `coverage run --source=eeg_a01 -m pytest`).

## Layout

| Module | Contents |
| --- | --- |
| `config.py` | bands, filter specifications, analysis windows relative to events, Monte Carlo settings |
| `edf_io.py` | EDF loading; physical and digital range read from the file header |
| `spectral.py` | custom radix-2 FFT, amplitude spectrum, band power, peak picking, alias frequency |
| `convolution.py` | direct convolution, Gaussian and moving-average kernels, rise time |
| `timefreq.py` | STFT, complex Morlet CWT (pywt and batched), DWT, operation counts |
| `filters.py` | Butterworth, elliptic and notch design, zero-phase filtering, mains detector |
| `events.py` | pulse and theta-burst detection, pulse subtraction before the burst is measured, windows anchored on a detected pulse |
| `evaluation.py` | stress-test contamination, SNR metrics, paired statistics |
| `analysis.py` | `Study`, assembled from the task modules below; `Study.run()` returns the contents of `results.json` |
| `study_base.py` | recordings, detected events, designed filters and shared helpers |
| `task1.py` | FFT check, signal characteristics and aliasing |
| `task2.py` | convolution, time-frequency localisation, pulse timing and cost |
| `task3.py` | filter design and parameter study, notch, clean-record performance |
| `stress.py` | one shared scoring loop used by the main stress test and the ripple study, plus the mains checks |
| `names.py` | filter labels shared by the analysis, benchmarks and plots |
| `benchmark.py` | every wall-clock measurement, written to `timings.json` |
| `plotting.py` | one function per report figure |
| `cli.py` | argument parsing and `main()` |

Nothing about the recordings is typed in. Sampling rate, length and amplitude range come from
the EDF header, the spectral lines used as probe frequencies are the strongest peaks found in
each band, and the pulse and theta-burst windows are detected from the signal.
