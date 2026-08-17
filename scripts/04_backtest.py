"""Phase 4: Backtest on test set and generate report card.

Run: python scripts/04_backtest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from datetime import date

from ufc.io import paths, parquet
from ufc.models.winner import WinnerEnsemble
from ufc.evaluation.backtest import run_backtest
from ufc.evaluation.reportcard import write_report


def main():
    print("=== Phase 4: Backtest ===")

    # Find latest winner model
    model_dir = paths.outputs_models()
    winner_files = sorted(model_dir.glob("winner_ensemble_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not winner_files:
        print("ERROR: No winner model found. Run 03_train.py first.")
        sys.exit(1)

    model_path = winner_files[-1]
    print(f"  Loading model: {model_path.name}")
    ensemble = WinnerEnsemble.load(model_path)

    features_winner = parquet.read(paths.processed("features_winner"))
    import pandas as pd
    features_winner["event_date"] = pd.to_datetime(features_winner["event_date"])

    print("  Running backtest on test set...")
    metrics = run_backtest(ensemble, features_winner)

    print("\n  Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    report_path = paths.outputs_reports() / f"backtest_{date.today()}.md"
    write_report(metrics, report_path)


if __name__ == "__main__":
    main()
