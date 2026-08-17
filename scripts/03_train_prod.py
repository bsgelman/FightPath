"""Train the production model tier on all data up to the latest available.

Prod artifacts go to outputs/models/prod/ and are preferred by the serving
loader (predict_core._find_latest_model) over the locked eval-tier artifacts.
The eval-tier models in outputs/models/ are NOT touched; Gate A-D scripts
still evaluate the eval tier.

Usage:
    python scripts/03_train_prod.py            # use configs/split_prod.yaml as-is
    python scripts/03_train_prod.py --auto     # derive split dates from latest data
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd
import yaml

from ufc.io import paths, parquet


def resolve_prod_split_dates(df: pd.DataFrame) -> dict:
    """Prod split: train on ALL data, val = recent overlap window for calibration only.

    No held-out test set — prod model predicts genuinely future fights, so there is
    no leakage concern. All historical fights go into LGBM training. The most recent
    6 months are ALSO used as the val window for Platt scaling, rate_calib_factor,
    and halflife search — slight calibration overlap is intentional and ensures
    calibration tracks the current era rather than end-2023 patterns.

    test_start is set to tomorrow so get_splits("test") returns an empty mask
    (no holdout rows exist yet — they are the future fights we're predicting).
    """
    latest = df["event_date"].max()
    val_start = latest - pd.DateOffset(months=6)
    val_mid = val_start + (latest - val_start) / 2

    return {
        "train_start": "2010-01-01",
        "train_end": latest.strftime("%Y-%m-%d"),
        "val_a_start": val_start.strftime("%Y-%m-%d"),
        "val_a_end": (val_mid - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "val_b_start": val_mid.strftime("%Y-%m-%d"),
        "val_b_end": latest.strftime("%Y-%m-%d"),
        "val_start": val_start.strftime("%Y-%m-%d"),
        "val_end": latest.strftime("%Y-%m-%d"),
        "test_start": (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "test_end": (latest + pd.DateOffset(years=1)).strftime("%Y-%m-%d"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                        help="Derive split dates from latest data instead of split_prod.yaml")
    args = parser.parse_args()

    # Set environment variable so all split readers pick up the prod config.
    os.environ["UFC_SPLIT_CONFIG"] = "split_prod.yaml"

    if args.auto:
        props_df = parquet.read(paths.processed("features_props"))
        props_df["event_date"] = pd.to_datetime(props_df["event_date"])
        dates = resolve_prod_split_dates(props_df)
        split_path = paths.root() / "configs" / "split_prod.yaml"
        split_path.write_text(yaml.dump(dates, default_flow_style=False, sort_keys=True))
        print(f"[prod] --auto: updated {split_path.name}")
        for k, v in sorted(dates.items()):
            print(f"  {k}: {v}")

    from ufc.io import paths as _paths
    model_dir = _paths.outputs_models_prod()
    print(f"[prod] Saving artifacts to: {model_dir}")

    from ufc.training.train_all import train
    result = train(model_dir=model_dir, prod_mode=True)
    print(f"\n[prod] Done. Artifacts in {result['model_dir']}")
    return result


if __name__ == "__main__":
    main()
