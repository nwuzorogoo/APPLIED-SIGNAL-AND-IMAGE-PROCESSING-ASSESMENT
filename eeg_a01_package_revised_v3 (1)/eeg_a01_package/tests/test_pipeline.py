"""End-to-end tests of the analysis, benchmarks, figures and command line.

The synthetic dataset in conftest.py lets these run without the EDF files. Three Monte Carlo
draws keep the whole pipeline to a few seconds.
"""
import json

import numpy as np
import pytest

from eeg_a01 import benchmark, cli, config
from eeg_a01.analysis import STRESS_NAMES, Study, freq_key
from eeg_a01.plotting import make_all

from conftest import PULSES


def test_results_have_every_section(synthetic_study):
    results = synthetic_study.results
    for key in ("fft_max_abs_error_vs_numpy", "chars", "peaks", "events", "alias_table",
                "alias_demo", "conv", "complexity", "loc", "pulse_timing", "dwt_bands",
                "min_orders_40_50_zero_phase", "param", "perf", "notch", "stress_mc"):
        assert key in results, key
    assert "timing_ms" not in results and "fft_custom_ms" not in results


def test_results_are_json_serialisable(synthetic_study):
    text = json.dumps(synthetic_study.results, default=cli.to_json)
    assert "NaN" not in text


def test_events_found_in_synthetic_records(synthetic_study):
    for record_id, spans in PULSES.items():
        found = [(p.onset, p.offset) for p in synthetic_study.pulses[record_id]]
        assert found == spans
    start, end, _ = synthetic_study.bursts[1]
    assert start == pytest.approx(4.4, abs=0.2) and end == pytest.approx(5.4, abs=0.2)


def test_probe_frequencies_come_from_the_data(synthetic_study):
    assert synthetic_study.probe(1, "beta") == pytest.approx(29.4, abs=0.05)
    assert synthetic_study.probe(3, "alpha") == pytest.approx(10.6, abs=0.05)
    assert freq_key(10.59) == "10p6"


def test_header_values_feed_the_characteristics(synthetic_study):
    chars = synthetic_study.results["chars"][1]
    assert chars["lsb_uV"] == pytest.approx(400 / 65535)
    assert chars["fs"] == 256.0 and chars["n"] == 2560


def test_filters_improve_snr_and_butterworth_keeps_peaks(synthetic_study):
    results = synthetic_study.results
    for record_id in synthetic_study.ids:
        for name in STRESS_NAMES:
            assert results["stress_mc"][record_id][name]["gain"]["mean"] > 5
        butter = results["perf"]["Butterworth (N=4)"][record_id]
        assert all(abs(v) < 0.6 for v in butter["peak_change_dB_zp"].values())
    assert results["perf"]["Butterworth (N=4)"]["zero_phase_work_ratio"] == pytest.approx(3.2)


def test_cwt_times_pulse_edges_better_than_stft(synthetic_study):
    summary = synthetic_study.results["pulse_timing"]["summary"]
    assert summary["cwt"]["pulses_resolved"] >= summary["cwt"]["pulses"] - 1
    assert summary["cwt"]["resolved_mean_abs_err_ms"] < 10
    for window in config.STFT_WINDOWS:
        assert summary[f"stft{window}"]["mean_abs_err_ms"] > summary["cwt"]["mean_abs_err_ms"]


def test_study_rejects_too_few_draws(synthetic_records):
    with pytest.raises(ValueError):
        Study(synthetic_records, n_draws=2)


def test_study_needs_a_pulse_in_the_transient_record(synthetic_records):
    quiet = dict(synthetic_records)
    record = quiet[config.TRANSIENT_RECORD]
    flat = record.data.copy()
    for onset, offset in PULSES[config.TRANSIENT_RECORD]:
        flat[onset:offset] -= 90.0
    quiet[config.TRANSIENT_RECORD] = type(record)(record.record_id, flat, record.fs,
                                                  record.calibration)
    with pytest.raises(ValueError):
        Study(quiet, n_draws=3)


# ---------------------------------------------------------------- benchmarks
def fast_time_call(fn, calls=1, blocks=1):
    fn()
    return 1.0


def test_benchmarks_mirror_result_keys(synthetic_study, monkeypatch):
    monkeypatch.setattr(benchmark, "time_call", fast_time_call)
    monkeypatch.setattr(benchmark, "time_once", lambda fn: fast_time_call(fn))
    monkeypatch.setattr(config, "SCALING_LENGTHS", 2)
    timings = benchmark.run_benchmarks(synthetic_study)
    assert set(timings["timing_ms"]) == {"FFT", "STFT_0.5s", "STFT_2s", "CWT",
                                         "CWT_direct_conv", "DWT"}
    assert set(timings["perf"]) == set(synthetic_study.results["perf"])
    merged = benchmark.merge(synthetic_study.results, timings)
    assert "units_N2560" in merged["complexity"] and "runtime_rel_FFT" in merged["complexity"]
    assert "corr_ref" in merged["perf"]["Butterworth (N=4)"][1]
    assert "runtime_ms_zero_phase" in merged["perf"]["Butterworth (N=4)"]


def test_time_call_and_scaling_exponent():
    assert benchmark.time_call(lambda: None, calls=2, blocks=2) >= 0
    with pytest.raises(ValueError):
        benchmark.time_call(lambda: None, calls=0)
    lengths = [100, 200, 400, 800]
    assert benchmark.scaling_exponent(lengths, [n ** 1.5 for n in lengths]) == pytest.approx(1.5)


# ---------------------------------------------------------------- figures and CLI
def test_every_figure_is_written(synthetic_study, tmp_path):
    make_all(synthetic_study, synthetic_study.results, tmp_path)
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == sorted(["fig1_time_freq.png", "fig2_aliasing.png", "fig3_convolution.png",
                            "fig4_tf.png", "fig5_filter_params.png", "fig6_filter_perf.png",
                            "fig7_notch_mc.png", "figA_dwt.png", "figA_families.png"])


def test_cli_without_benchmarks(monkeypatch, tmp_path, synthetic_records):
    monkeypatch.setattr(cli, "load_recordings", lambda folder: synthetic_records)
    out = tmp_path / "out"
    assert cli.main(["--data-dir", str(tmp_path), "--out-dir", str(out), "--draws", "3",
                     "--no-figures", "--no-benchmarks"]) == 0
    assert (out / "results.json").is_file()
    assert not (out / "timings.json").exists()
    results = json.loads((out / "results.json").read_text())
    assert results["stress_mc"]["n_draws"] == 3


def test_cli_writes_timings_separately(monkeypatch, tmp_path, synthetic_records):
    monkeypatch.setattr(cli, "load_recordings", lambda folder: synthetic_records)
    monkeypatch.setattr(benchmark, "run_benchmarks", lambda study: {"timing_ms": {"FFT": 1.0}})
    out = tmp_path / "out"
    cli.main(["--data-dir", str(tmp_path), "--out-dir", str(out), "--draws", "3", "--no-figures"])
    assert json.loads((out / "timings.json").read_text()) == {"timing_ms": {"FFT": 1.0}}


@pytest.mark.parametrize("argv", [["--draws", "2"], ["--data-dir", "/no/such/folder"]])
def test_cli_rejects_bad_arguments(argv, tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args(argv + ([] if "--data-dir" in argv else ["--data-dir", str(tmp_path)]))


def test_to_json_handles_numpy_scalars():
    assert cli.to_json(np.float64(1.5)) == 1.5 and cli.to_json(np.bool_(True)) is True
    with pytest.raises(TypeError):
        cli.to_json(object())
