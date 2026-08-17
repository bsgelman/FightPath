"""Retrain only the duration model — used after props are already saved.

Run after 03_train.py crashes/completes for all other models:
    python scripts/_retrain_duration_only.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.tune_props import get_prop_feature_cols
from ufc.models.props_duration import DurationModel


def _gitsha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parents[1],
        ).decode().strip()
    except Exception:
        return "nogit"


def main():
    print("=== Duration-only retrain ===\n")

    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])

    from ufc.features.interactions import compute_interactions
    props_df = compute_interactions(props_df)

    from ufc.training.splits import get_splits
    props_splits = get_splits(props_df)

    prop_cols_all = get_prop_feature_cols(props_df, model_name="props")
    prop_cols = [c for c in prop_cols_all if c in props_df.columns]

    dur_train = props_df[props_splits["train"]].dropna(subset=["total_fight_sec"]).copy()
    dur_val = props_df[props_splits["val"]].dropna(subset=["total_fight_sec"]).copy()
    dur_train = dur_train.drop_duplicates(subset=["fight_id"])
    dur_val = dur_val.drop_duplicates(subset=["fight_id"])

    event_obs_train = ~dur_train["method"].isin(["U-DEC", "S-DEC", "M-DEC"])
    event_obs_val = ~dur_val["method"].isin(["U-DEC", "S-DEC", "M-DEC"])

    dur_cols = [c for c in prop_cols if c in dur_train.columns]
    # v8.27: re-add finish-propensity specialist features for method-specific timing
    _duration_extras = [c for c in [
        "ko_specialist_idx_a", "ko_specialist_idx_b",
        "sub_specialist_idx_a", "sub_specialist_idx_b",
        "finish_share_a", "finish_share_b",
    ] if c in dur_train.columns and c not in dur_cols]
    dur_cols = dur_cols + _duration_extras
    if _duration_extras:
        print(f"  Duration extras added ({len(_duration_extras)}): {_duration_extras}")

    print(f"  Train rows: {len(dur_train)}  Val rows: {len(dur_val)}")
    print(f"  Feature cols: {len(dur_cols)}")

    dur_model = DurationModel()
    dur_model.fit(
        dur_train, dur_train["total_fight_sec"], event_obs_train,
        dur_val, dur_val["total_fight_sec"], event_obs_val,
        dur_cols,
        train_dates=dur_train.get("event_date"),
        val_dates=dur_val.get("event_date"),
        train_method=dur_train.get("method"),
        val_method=dur_val.get("method"),
    )

    model_dir = paths.outputs_models()
    gitsha = _gitsha()
    dp = dur_model.save(model_dir, gitsha)
    print(f"  Saved: {dp.name}")
    print("\nDuration retrain complete.")


if __name__ == "__main__":
    main()
