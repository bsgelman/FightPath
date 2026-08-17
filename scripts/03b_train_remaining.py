"""Train method, props, and duration models only — skip winner re-train.

Use when winner_ensemble already exists (e.g. after a crash mid-pipeline).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import subprocess

import numpy as np
import pandas as pd

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.symmetrize import symmetrize
from ufc.training.tune_props import get_prop_feature_cols
from ufc.models.method import MethodClassifier
from ufc.models.props_count import CountModel, HurdleCountModel
from ufc.models.props_duration import DurationModel


def _gitsha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.root(), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")


def main():
    gitsha = _gitsha()
    model_dir = paths.outputs_models()
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training: method + props + duration (winner skipped) ===")
    print(f"  gitsha: {gitsha}")

    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])

    prop_feature_cols = get_prop_feature_cols(props_df)
    props_splits = get_splits(props_df)

    # ── Method classifier ─────────────────────────────────────────────────
    print("\n[1/3] Training method classifier...")
    p_train = props_df[props_splits["train"]].dropna(subset=["method"])
    p_val = props_df[props_splits["val"]].dropna(subset=["method"])

    valid_methods = ["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"]
    p_train = p_train[p_train["method"].isin(valid_methods)].copy()
    p_val = p_val[p_val["method"].isin(valid_methods)].copy()

    p_train_sym = symmetrize(p_train)
    p_train_sym = p_train_sym.dropna(subset=["method"])

    method_clf = MethodClassifier()
    method_clf.fit(
        p_train_sym, p_train_sym["method"],
        p_val, p_val["method"],
        [c for c in prop_feature_cols if c in p_train_sym.columns],
    )
    method_path = method_clf.save(model_dir, gitsha)
    print(f"  Saved method model: {method_path.name}")

    # ── Prop count models ─────────────────────────────────────────────────
    print("\n[2/3] Training prop count models...")
    prop_cols = [c for c in prop_feature_cols if c in props_df.columns]

    for target, raw_col_a in [
        ("sig_strikes", "sig_str_landed_a"),
        ("takedowns", "td_landed_a"),
        ("r1_sig_strikes", "r1_sig_str_landed_a"),
    ]:
        if raw_col_a not in props_df.columns:
            print(f"  Skipping {target} — column {raw_col_a} not found")
            continue

        rows_a = props_df[props_splits["train"]].copy()
        rows_a["_y"] = rows_a[raw_col_a] if raw_col_a in rows_a.columns else 0
        X_tr = rows_a[prop_cols].fillna(0)
        y_tr = rows_a["_y"].fillna(0)

        rows_a_val = props_df[props_splits["val"]].copy()
        rows_a_val["_y"] = rows_a_val[raw_col_a] if raw_col_a in rows_a_val.columns else 0
        X_vl = rows_a_val[prop_cols].fillna(0)
        y_vl = rows_a_val["_y"].fillna(0)

        if target != "r1_sig_strikes" and "total_fight_sec" in rows_a.columns and "scheduled_rounds" in rows_a.columns:
            sched_sec = (rows_a["scheduled_rounds"].fillna(3).astype(float) * 300.0).clip(lower=1)
            sample_weight = (rows_a["total_fight_sec"].fillna(sched_sec) / sched_sec).clip(0.1, 1.0).values
        else:
            sample_weight = None

        print(f"  Training {target} model...")
        if target == "takedowns":
            cm = HurdleCountModel(target=target)
        else:
            cm = CountModel(target=target)
        cm.fit(X_tr, y_tr, X_vl, y_vl, prop_cols, sample_weight=sample_weight)
        cp = cm.save(model_dir, gitsha)
        print(f"  Saved: {cp.name}")

    # ── Duration model ────────────────────────────────────────────────────
    print("\n[3/3] Training fight duration model...")
    dur_train = props_df[props_splits["train"]].dropna(subset=["total_fight_sec"]).copy()
    dur_val = props_df[props_splits["val"]].dropna(subset=["total_fight_sec"]).copy()

    dur_train = dur_train.drop_duplicates(subset=["fight_id"])
    dur_val = dur_val.drop_duplicates(subset=["fight_id"])

    event_obs_train = ~dur_train["method"].isin(["U-DEC", "S-DEC", "M-DEC"])
    event_obs_val = ~dur_val["method"].isin(["U-DEC", "S-DEC", "M-DEC"])

    dur_model = DurationModel()
    dur_cols = [c for c in prop_cols if c in dur_train.columns]
    dur_model.fit(
        dur_train, dur_train["total_fight_sec"],
        event_obs_train,
        dur_val, dur_val["total_fight_sec"],
        event_obs_val,
        dur_cols,
    )
    dp = dur_model.save(model_dir, gitsha)
    print(f"  Saved: {dp.name}")

    print("\n=== Done. Method + props + duration trained with v4 features. ===")


if __name__ == "__main__":
    main()
