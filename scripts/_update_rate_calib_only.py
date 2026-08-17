"""Update rate_calib_factor only in existing count models — no LightGBM retrain.

Run after the method clf T-bounds change to re-anchor rate calibration to the
new method-conditional duration distribution. Much faster than _retrain_count_only.py.
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import subprocess
import numpy as np
import pandas as pd

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.prop_targets import PROP_TARGET_SPECS
from ufc.models.props_count import RateHurdleCountModel
from ufc.models.props_duration import DurationModel
from ufc.models.method import MethodClassifier, METHOD_CLASSES


def _gitsha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.root(), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")


def _fit_rate_calib_factor(count_model, duration_model, X_val, y_val, method_proba):
    """Val-anchored multiplicative rate factor on the METHOD-MARGINAL forecast."""
    dur_cdfs = duration_model.predict_cdf(X_val, use_boundary_mass=False)
    dur_by_method = {
        m: duration_model.predict_cdf(X_val, method_override=m, use_boundary_mass=False)
        for m in ("KO/TKO", "SUB", "DEC")
    }
    saved_adj = count_model.method_log_rate_adj
    saved_f = count_model.rate_calib_factor
    count_model.method_log_rate_adj = None
    count_model.rate_calib_factor = 1.0
    try:
        cdfs = count_model.predict_cdf(
            X_val, duration_cdfs=dur_cdfs,
            method_proba=method_proba, duration_cdfs_by_method=dur_by_method,
        )
        pred_mean = float(np.mean([c._samples.mean() for c in cdfs]))
    finally:
        count_model.method_log_rate_adj = saved_adj
        count_model.rate_calib_factor = saved_f
    actual_mean = float(np.asarray(y_val.fillna(0), dtype=float).mean())
    return actual_mean / pred_mean if pred_mean > 0 else 1.0


def main():
    gitsha = _gitsha()
    model_dir = paths.outputs_models()
    print(f"=== Rate-calib-only update (gitsha={gitsha}) ===\n")

    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])
    props_splits = get_splits(props_df)
    val_rows = props_df[props_splits["val"]].copy().reset_index(drop=True)

    # Load newest method clf and duration model
    dur_files = sorted(model_dir.glob("props_duration_*.joblib"), key=lambda p: p.stat().st_mtime)
    meth_files = sorted(model_dir.glob("method_clf_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not dur_files or not meth_files:
        print("ERROR: No duration or method model found"); return

    dur_model = DurationModel.load(dur_files[-1])
    method_clf = MethodClassifier.load(meth_files[-1])
    print(f"  Duration: {dur_files[-1].name}")
    print(f"  Method:   {meth_files[-1].name}  T={method_clf.temperature:.4f}\n")

    _mp = method_clf.predict_proba_dict(val_rows)
    method_proba = np.column_stack([_mp[c] for c in METHOD_CLASSES])

    for spec in PROP_TARGET_SPECS:
        if not spec.rate_calib:
            continue
        pattern = f"props_{spec.target}_*.joblib"
        files = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not files:
            print(f"  SKIP {spec.target} — no model found")
            continue
        count_model = RateHurdleCountModel.load(files[-1])
        print(f"  {spec.target}: loaded {files[-1].name}  (old factor={count_model.rate_calib_factor:.4f})")

        y_val = val_rows[spec.raw_col_a]
        factor = _fit_rate_calib_factor(count_model, dur_model, val_rows, y_val, method_proba)
        print(f"    new factor = {factor:.4f}", end="")
        if not (0.90 <= factor <= 1.10):
            print(f"  WARNING: outside [0.90, 1.10]", end="")
        print()

        count_model.rate_calib_factor = factor
        out_path = count_model.save(model_dir, gitsha)
        print(f"    saved: {out_path.name}")

    print("\n=== Rate-calib update complete ===")


if __name__ == "__main__":
    main()
