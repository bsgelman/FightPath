"""Retrain only the method model — keeps winner/count/duration models byte-identical.

Run after train_all.py or when only the method model needs updating:
    python scripts/_retrain_method_only.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.tune_props import get_prop_feature_cols
from ufc.training.recency import search_halflife_method
from ufc.training.feature_pruning import compute_dead_features_from_importances, save_dead_features
from ufc.models.method import MethodClassifier


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
    print("=== Method-only retrain ===\n")

    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])

    props_splits = get_splits(props_df)

    valid_methods = ["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"]
    p_train = props_df[props_splits["train"]].dropna(subset=["method"])
    p_train = p_train[p_train["method"].isin(valid_methods)].copy()
    p_val = props_df[props_splits["val"]].dropna(subset=["method"])
    p_val = p_val[p_val["method"].isin(valid_methods)].copy()

    print(f"  Train rows: {len(p_train)}  Val rows: {len(p_val)}")

    # Base feature set (count/duration exclude list)
    prop_feature_cols = get_prop_feature_cols(props_df, model_name="method")

    # v8.25: re-add specialist/KO-matchup features excluded from count/duration models
    # but designed for winner/method use (tune_props.py:31-44 comments say "method/winner only").
    # v8.26: removed era_ko/sub_share — dominated importances, anchored every fight to era mean.
    _method_extras = [c for c in [
        # Specialist finish amplifiers — strongest finish discriminators (interactions.py:116-140)
        "ko_specialist_idx_a", "ko_specialist_idx_b",
        "ko_specialist_x_weakness_a", "ko_specialist_x_weakness_b",
        "ko_specialist_x_chin_a", "ko_specialist_x_chin_b",
        "sub_specialist_idx_a", "sub_specialist_idx_b",
        "sub_specialist_x_weakness_a", "sub_specialist_x_weakness_b",
        "finish_share_a", "finish_share_b",
        "grappling_control_threat_a", "grappling_control_threat_b",
        # KO matchup discrimination features — method/winner only (interactions.py:175-188)
        "ko_matchup_a", "ko_matchup_b",
        "chin_decay_threat_a", "chin_decay_threat_b",
        "layoff_chin_a", "layoff_chin_b",
    ] if c in p_train.columns]

    method_feature_cols = [c for c in prop_feature_cols if c in p_train.columns]
    for _c in _method_extras:
        if _c not in method_feature_cols:
            method_feature_cols.append(_c)

    print(f"  Feature cols: {len(method_feature_cols)}  (extras added: {len(_method_extras)})")

    best_h, sw = search_halflife_method(
        p_train, p_train["method"],
        p_val, p_val["method"],
        method_feature_cols=method_feature_cols,
        train_dates=p_train["event_date"],
        grid=[365, 730, 1095, 1460, None],
        anchor="2023-12-31",
        verbose=True,
    )
    print(f"  Method halflife selected: {best_h}")

    method_clf = MethodClassifier()
    method_clf.fit(
        p_train, p_train["method"],
        p_val, p_val["method"],
        method_feature_cols,
        sample_weight=sw,
        train_dates=p_train["event_date"],
    )

    model_dir = paths.outputs_models()
    gitsha = _gitsha()
    mp = method_clf.save(model_dir, gitsha)
    print(f"  Saved: {mp.name}")

    # Export dead features
    try:
        dead = compute_dead_features_from_importances(
            {"lgbm": method_clf.model.feature_importances_},
            method_clf.feature_cols,
        )
        if dead:
            dead_path = save_dead_features(dead, "method")
            print(f"  Exported {len(dead)} method dead features -> {dead_path.name}")
    except Exception as e:
        print(f"  [warn] dead feature export: {e}")

    print("\nMethod retrain complete.")


if __name__ == "__main__":
    main()
