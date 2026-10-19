"""Command-line entry point: python -m eeg_a01 --data-dir dataset --out-dir output

results.json holds every deterministic value. timings.json holds the wall-clock measurements,
which vary between runs and machines, so the two never need to be compared together.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config
from .analysis import Study
from .edf_io import load_recordings


def to_json(value):
    """Convert NumPy scalars for json.dump."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=to_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="eeg_a01", description="EEG analysis and filter design for CMP9780 A01.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"),
                        help="folder holding eeg_dataset_01.edf to eeg_dataset_03.edf")
    parser.add_argument("--out-dir", type=Path, default=Path("output"),
                        help="folder for results.json and figs/")
    parser.add_argument("--draws", type=int, default=config.N_DRAWS,
                        help="Monte Carlo draws per record (default %(default)s)")
    parser.add_argument("--no-figures", action="store_true", help="skip drawing figures")
    parser.add_argument("--no-benchmarks", action="store_true",
                        help="skip the wall-clock timings (timings.json is not written)")
    args = parser.parse_args(argv)
    if args.draws < 3:
        parser.error("--draws must be at least 3")
    if not args.data_dir.is_dir():
        parser.error(f"data folder {args.data_dir} does not exist")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    records = load_recordings(args.data_dir)
    study = Study(records, n_draws=args.draws)
    results = study.run()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "results.json", results)
    if not args.no_benchmarks:
        from .benchmark import run_benchmarks
        write_json(args.out_dir / "timings.json", run_benchmarks(study))
    if not args.no_figures:
        from .plotting import make_all
        make_all(study, results, args.out_dir / "figs")
    print(f"done in {time.perf_counter() - started:.0f} s, results in {args.out_dir}")
    return 0
