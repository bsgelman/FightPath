"""Run the full UFC prediction pipeline end-to-end.

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --start features  # skip ingest, start from features
    python run_pipeline.py --skip backtest   # skip evaluation steps
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable

STEPS = [
    ("ingest",   "scripts/01_ingest.py",        "Ingesting raw data -> ledger"),
    ("features", "scripts/02_build_features.py", "Building features"),
    ("train",    "scripts/03_train.py",           "Training models"),
    ("backtest", "scripts/04_backtest.py",        "Running backtest"),
    ("props",    "scripts/05_evaluate_props.py",  "Evaluating prop calibration"),
]

STEP_NAMES = [s[0] for s in STEPS]


def run_step(name: str, script: str, label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([PYTHON, script], cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ERROR: {name} failed (exit {result.returncode}) after {elapsed:.0f}s")
        return False
    print(f"\n  DONE: {name} completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run UFC prediction pipeline")
    parser.add_argument("--start", choices=STEP_NAMES, default=None,
                        help="Start from this step (skip earlier steps)")
    parser.add_argument("--stop", choices=STEP_NAMES, default=None,
                        help="Stop after this step")
    parser.add_argument("--skip", choices=STEP_NAMES, nargs="+", default=[],
                        help="Skip specific steps")
    args = parser.parse_args()

    start_idx = STEP_NAMES.index(args.start) if args.start else 0
    stop_idx  = STEP_NAMES.index(args.stop) + 1 if args.stop else len(STEPS)
    steps_to_run = STEPS[start_idx:stop_idx]

    pipeline_start = time.time()
    print(f"\nUFC Prediction Pipeline")
    print(f"Steps: {' -> '.join(s[0] for s in steps_to_run)}")
    if args.skip:
        print(f"Skipping: {args.skip}")

    for name, script, label in steps_to_run:
        if name in args.skip:
            print(f"\n  Skipping {name}")
            continue
        ok = run_step(name, script, label)
        if not ok:
            print(f"\nPipeline aborted at step '{name}'.")
            sys.exit(1)

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
