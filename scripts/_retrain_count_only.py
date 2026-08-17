"""Retrain all count models (prop targets from PROP_TARGET_SPECS).

Skips winner/method/duration — those are unchanged. Use when you only need
to pick up changes in props_count.py or when adding new prop targets.
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import subprocess
import yaml
import numpy as np
import pandas as pd

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.tune_props import get_prop_feature_cols
from ufc.training.prop_targets import PROP_TARGET_SPECS
from ufc.models.props_count import RateHurdleCountModel, ControlShareModel
from ufc.models.props_duration import DurationModel
from ufc.models.method import MethodClassifier, METHOD_CLASSES
from ufc.features.interactions import compute_interactions


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


def _fit_finish_draw_scale(count_model, duration_model, X_val, y_val, method_proba,
                           ceiling: float) -> float:
    """Val-fit the marginal R1 finish-draw shrink so overall val meanPIT -> 0.5.

    Corrects the stable r1_end finishing-burst over-prediction (meanPIT~0.42 on both
    val and test). Does NOT fix the 2025 survivor (past_r1) drift — r1 stays a
    documented Gate-B FAIL; this only reduces the failing margin honestly (val-fit).
    """
    from scipy import optimize as _opt
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location(
        "_eval05fds", str(Path(__file__).resolve().parent / "05_evaluate_props.py"))
    _evm = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_evm)
    dur_cdfs = duration_model.predict_cdf(X_val, use_boundary_mass=False)
    y_arr = np.asarray(y_val.fillna(0), dtype=float)
    _saved = getattr(count_model, "finish_draw_scale", 1.0)

    def _mean_pit(scale: float) -> float:
        count_model.finish_draw_scale = float(scale)
        cdfs = count_model.predict_cdf(
            X_val, duration_cdfs=dur_cdfs, active_minutes_ceiling=ceiling,
            method_proba=method_proba, duration_cdfs_by_method=None,
            apply_burst=False, apply_method_hurdle=False, use_binned_rate_adj=False,
            use_finish_head=True, use_cond_hurdle=False,
            mean_preserve_cond_hurdle=True, use_sub_count_head=False)
        pit = _evm._compute_pit_vals(cdfs, y_arr, np.random.default_rng(42))
        return float(np.mean(pit)) - 0.5

    try:
        if _mean_pit(0.5) * _mean_pit(1.0) < 0:
            scale = float(_opt.brentq(_mean_pit, 0.5, 1.0))
        else:
            scale = 1.0
    except Exception:
        scale = 1.0
    finally:
        count_model.finish_draw_scale = _saved
    return float(np.clip(scale, 0.5, 1.0))


def main():
    gitsha = _gitsha()
    model_dir = paths.outputs_models()
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Count-only retrain (gitsha={gitsha}) ===")

    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])

    print("  Re-applying interaction features...")
    props_df = compute_interactions(props_df)

    prop_cols_all = get_prop_feature_cols(props_df, model_name="props")
    prop_cols = [c for c in prop_cols_all if c in props_df.columns]
    _RATE_EXCLUDE = {"referee_stoppage_threshold"}
    rate_prop_cols = [c for c in prop_cols if c not in _RATE_EXCLUDE]
    print(f"  Prop feature count: {len(prop_cols)}  rate cols: {len(rate_prop_cols)}")

    props_splits = get_splits(props_df)

    _split_cfg = yaml.safe_load((paths.root() / "configs" / "split.yaml").read_text())
    _train_end_dt = pd.to_datetime(_split_cfg["train_end"])

    def _compute_active_minutes(rows: pd.DataFrame, ceiling: float | None) -> np.ndarray:
        if ceiling is not None:
            end_round = rows["end_round"].fillna(99).astype(float)
            end_time = rows["end_time_sec"].fillna(ceiling * 60.0).astype(float)
            r1_sec = np.where(end_round == 1, np.clip(end_time, 5.0, ceiling * 60.0), ceiling * 60.0)
            return (r1_sec / 60.0)
        else:
            sched_sec = (rows["scheduled_rounds"].fillna(3).astype(float) * 300.0).clip(lower=1)
            return (rows["total_fight_sec"].fillna(sched_sec) / 60.0).clip(lower=5.0 / 60.0).values

    def _compute_sample_weight(rows: pd.DataFrame, scheme: str | None) -> np.ndarray | None:
        if scheme == "censor" and "total_fight_sec" in rows.columns and "scheduled_rounds" in rows.columns:
            sched_sec = (rows["scheduled_rounds"].fillna(3).astype(float) * 300.0).clip(lower=1)
            return (rows["total_fight_sec"].fillna(sched_sec) / sched_sec).clip(0.1, 1.0).values
        elif scheme == "recency" and "event_date" in rows.columns:
            days_old = (_train_end_dt - pd.to_datetime(rows["event_date"])).dt.days.clip(lower=0).astype(float)
            return np.power(0.5, days_old / 730.0).clip(lower=0.05).values
        return None

    trained_count_models: dict = {}

    import os
    _only = {t for t in os.environ.get("RETRAIN_ONLY", "").split(",") if t}
    if _only:
        print(f"  RETRAIN_ONLY active -> {sorted(_only)}")

    for spec in PROP_TARGET_SPECS:
        target, raw_col_a = spec.target, spec.raw_col_a
        if _only and target not in _only:
            continue
        if raw_col_a not in props_df.columns:
            print(f"  Skipping {target} — column {raw_col_a} not found")
            continue

        rows_a = props_df[props_splits["train"]].copy().reset_index(drop=True)
        rows_a["_y"] = rows_a[raw_col_a]
        X_tr = rows_a[prop_cols].fillna(0)
        y_tr = rows_a["_y"].fillna(0)

        rows_a_val = props_df[props_splits["val"]].copy().reset_index(drop=True)
        rows_a_val["_y"] = rows_a_val[raw_col_a]
        X_vl = rows_a_val[prop_cols].fillna(0)
        y_vl = rows_a_val["_y"].fillna(0)

        act_min_tr = _compute_active_minutes(rows_a, spec.ceiling)
        act_min_vl = _compute_active_minutes(rows_a_val, spec.ceiling)
        sample_weight = _compute_sample_weight(rows_a, spec.weight)

        if getattr(spec, "model_kind", "rate_hurdle") == "control_share":
            print(f"\n  Training {target} (ControlShareModel)...")
            total_sec_tr = rows_a["total_fight_sec"].fillna(
                rows_a["scheduled_rounds"].fillna(3) * 300.0
            ).clip(lower=1.0).values
            total_sec_vl = rows_a_val["total_fight_sec"].fillna(
                rows_a_val["scheduled_rounds"].fillna(3) * 300.0
            ).clip(lower=1.0).values
            cm = ControlShareModel(target=target)
            cm.fit(
                rows_a[prop_cols].fillna(0), y_tr, total_sec_tr,
                rows_a_val[prop_cols].fillna(0), y_vl, total_sec_vl,
                rate_prop_cols, sample_weight=sample_weight,
            )
            if "method" in rows_a.columns:
                cm.fit_method_adjustments(
                    rows_a[prop_cols].fillna(0), y_tr, total_sec_tr,
                    rows_a["method"], sample_weight=sample_weight,
                )
        else:
            print(f"\n  Training {target} (RateHurdleCountModel, ceiling={spec.ceiling})...")
            cm = RateHurdleCountModel(
                target=target,
                active_minutes_ceiling=spec.ceiling,
                rate_ceiling=spec.rate_ceiling,
            )
            cm.fit(X_tr, y_tr, act_min_tr, X_vl, y_vl, act_min_vl,
                   rate_prop_cols, sample_weight=sample_weight)

            if "method" in rows_a.columns:
                cm.fit_method_adjustments(
                    X_tr, y_tr, act_min_tr, rows_a["method"],
                    sample_weight=sample_weight,
                    event_dates_train=rows_a["event_date"] if "event_date" in rows_a.columns else None,
                    X_val=X_vl, y_val=y_vl,
                    active_minutes_val=act_min_vl,
                    method_val=rows_a_val["method"] if "method" in rows_a_val.columns else None,
                )

        cp = cm.save(model_dir, gitsha)
        print(f"  Saved: {cp.name}")
        trained_count_models[target] = cm

    # Val-anchored rate calibration for all rate_calib targets
    dur_files = sorted(model_dir.glob("props_duration_*.joblib"), key=lambda p: p.stat().st_mtime)
    meth_files = sorted(model_dir.glob("method_clf_*.joblib"), key=lambda p: p.stat().st_mtime)

    if dur_files and meth_files:
        dur_model = DurationModel.load(dur_files[-1])
        method_clf = MethodClassifier.load(meth_files[-1])
        _val_rows_calib = props_df[props_splits["val"]].copy().reset_index(drop=True)
        _mp_calib = method_clf.predict_proba_dict(_val_rows_calib)
        _method_proba_calib = np.column_stack([_mp_calib[c] for c in METHOD_CLASSES])

        for spec in PROP_TARGET_SPECS:
            if not spec.rate_calib:
                continue
            if spec.target not in trained_count_models or spec.raw_col_a not in props_df.columns:
                continue
            _cm_rc = trained_count_models[spec.target]
            _y_val_rc = _val_rows_calib[spec.raw_col_a]
            print(f"\n  [rate_calib] Computing {spec.target} rate_calib_factor on val set...")
            _factor = _fit_rate_calib_factor(_cm_rc, dur_model, _val_rows_calib, _y_val_rc, _method_proba_calib)
            print(f"  [rate_calib] {spec.target} rate_calib_factor = {_factor:.4f}  (expected [0.90, 1.10])")
            if not (0.90 <= _factor <= 1.10):
                print(f"  [rate_calib] WARNING: factor {_factor:.4f} outside [0.90, 1.10] -- "
                      f"check val set or rate model before shipping")
            _cm_rc.rate_calib_factor = _factor
            _rc_path = _cm_rc.save(model_dir, gitsha)
            print(f"  [rate_calib] Re-saved {spec.target}: {_rc_path.name}")

        # finish_draw_scale val-fit for r1_sig_strikes (drift-limited FAIL margin shrink)
        if "r1_sig_strikes" in trained_count_models:
            _cm_r1 = trained_count_models["r1_sig_strikes"]
            _y_r1 = _val_rows_calib["r1_sig_str_landed_a"]
            print("\n  [finish_scale] Fitting r1_sig_strikes finish_draw_scale on val...")
            _fs = _fit_finish_draw_scale(_cm_r1, dur_model, _val_rows_calib, _y_r1,
                                         _method_proba_calib, ceiling=5.0)
            print(f"  [finish_scale] r1_sig_strikes finish_draw_scale = {_fs:.4f}")
            _cm_r1.finish_draw_scale = _fs
            _fs_path = _cm_r1.save(model_dir, gitsha)
            print(f"  [finish_scale] Re-saved r1_sig_strikes: {_fs_path.name}")
    else:
        print("  [rate_calib] No duration or method model found — skipping rate calib")

    print(f"\n=== Count-only retrain complete. {len(trained_count_models)} models trained ===")


if __name__ == "__main__":
    main()
