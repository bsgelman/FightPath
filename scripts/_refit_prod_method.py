"""Re-fit ONLY the prod-tier method classifier with temporal-OOF temperature.

Targeted fix for v8.32: the prod method model's temperature was fit on the
in-sample val window (overlaps training) → T=0.1 (floor) → ~97% finish prob.
This re-fits temperature on temporal-OOF logits (prod_mode), overwriting the
prod method joblib in place WITHOUT the expensive winner OOF retrain.

Mirrors the method-training prep in train_all.train() exactly. Future prod
retrains via 03_train_prod.py --auto already pass prod_mode=True, so this
script is a one-off to avoid re-running the 5h winner OOF.

Run:
    python scripts/_refit_prod_method.py
"""
import os
import sys
from pathlib import Path

# Prod split MUST be set before any ufc import that reads the split config.
os.environ["UFC_SPLIT_CONFIG"] = "split_prod.yaml"

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import yaml
import pandas as pd

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.tune_props import get_prop_feature_cols
from ufc.models.method import MethodClassifier
from ufc.features.interactions import compute_interactions
from ufc.training.recency import search_halflife_method

# Stamp with the same sha as the rest of the prod tier so the one-per-key
# whitelist and loader stay consistent.
PROD_GITSHA = "b9db43c"


def main():
    model_dir = paths.outputs_models_prod()
    anchor = yaml.safe_load(
        (paths.root() / "configs" / "split_prod.yaml").read_text()
    )["train_end"]
    print(f"[refit-method] split=split_prod.yaml  anchor={anchor}  dir={model_dir}")

    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])
    props_df = compute_interactions(props_df)

    prop_feature_cols = get_prop_feature_cols(props_df, model_name="method")
    splits = get_splits(props_df)
    p_train = props_df[splits["train"]].dropna(subset=["method"])
    p_val = props_df[splits["val"]].dropna(subset=["method"])

    valid_methods = ["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"]
    p_train = p_train[p_train["method"].isin(valid_methods)].copy()
    p_val = p_val[p_val["method"].isin(valid_methods)].copy()

    # Same method-only feature extras as train_all.train()
    _method_extras = [c for c in [
        "ko_specialist_idx_a", "ko_specialist_idx_b",
        "ko_specialist_x_weakness_a", "ko_specialist_x_weakness_b",
        "ko_specialist_x_chin_a", "ko_specialist_x_chin_b",
        "sub_specialist_idx_a", "sub_specialist_idx_b",
        "sub_specialist_x_weakness_a", "sub_specialist_x_weakness_b",
        "finish_share_a", "finish_share_b",
        "grappling_control_threat_a", "grappling_control_threat_b",
        "ko_matchup_a", "ko_matchup_b",
        "chin_decay_threat_a", "chin_decay_threat_b",
        "layoff_chin_a", "layoff_chin_b",
    ] if c in p_train.columns]
    method_feature_cols = [c for c in prop_feature_cols if c in p_train.columns]
    for _c in _method_extras:
        if _c not in method_feature_cols:
            method_feature_cols.append(_c)

    best_h_method, sw_method = search_halflife_method(
        p_train, p_train["method"],
        p_val, p_val["method"],
        method_feature_cols=method_feature_cols,
        train_dates=p_train["event_date"],
        grid=[365, 730, 1095, 1460, None],
        anchor=anchor,
        verbose=True,
    )
    print(f"  Method halflife selected: {best_h_method}")

    method_clf = MethodClassifier()
    method_clf.fit(
        p_train, p_train["method"],
        p_val, p_val["method"],
        method_feature_cols,
        sample_weight=sw_method,
        train_dates=p_train["event_date"],
        temporal_oof=True,   # ← the fix: OOF temperature, not in-sample val
    )
    out = method_clf.save(model_dir, PROD_GITSHA)
    print(f"[refit-method] Saved: {out}")
    print(f"[refit-method] Temperature = {method_clf.temperature:.4f} "
          f"(was 0.1000 in-sample)")


if __name__ == "__main__":
    main()
