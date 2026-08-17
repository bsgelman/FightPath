"""Post-hoc rate_calib_factor optimization for sig_strikes and r1_sig_strikes.

Instead of mean-matching (train_all.py), optimize directly for minimal KS on
the val set — same metric the Gate B uses.  Operates on val data (2023) to
avoid test-set leakage.  Updates the saved model files in-place.

Run: python scripts/_recalib_rate_factor.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_1samp

from ufc.io import paths, parquet
from ufc.training.splits import get_splits
from ufc.training.tune_props import get_prop_feature_cols
from ufc.models.method import MethodClassifier, METHOD_CLASSES  # noqa: F401
from ufc.models.props_duration import DurationModel
from ufc.models.props_count import RateHurdleCountModel


def compute_pit(cdfs, y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized PIT — matches _compute_pit_vals in 05_evaluate_props.py."""
    pit = np.zeros(len(y_true))
    for i, (cdf, y) in enumerate(zip(cdfs, y_true)):
        if y == 0:
            p0 = cdf.cdf(0)
            pit[i] = float(rng.uniform(0, p0)) if p0 > 0 else 0.0
        else:
            pit[i] = cdf.cdf(float(y))
    return pit


def ks_for_factor(model: RateHurdleCountModel, factor: float, df: pd.DataFrame,
                  y: np.ndarray, dur_cdfs, method_proba,
                  dur_by_method, extra_kwargs: dict) -> float:
    """Compute KS statistic on df for a given rate_calib_factor."""
    orig_adj = model.method_log_rate_adj
    orig_f   = model.rate_calib_factor
    model.method_log_rate_adj = None
    model.rate_calib_factor   = factor
    try:
        cdfs = model.predict_cdf(
            df, duration_cdfs=dur_cdfs,
            method_proba=method_proba,
            duration_cdfs_by_method=dur_by_method,
            **extra_kwargs,
        )
        rng = np.random.default_rng(42)
        pit = compute_pit(cdfs, y, rng)
        stat, _ = ks_1samp(pit, lambda x: x)
        return float(stat)
    finally:
        model.method_log_rate_adj = orig_adj
        model.rate_calib_factor   = orig_f


def grid_search(model, df, y, dur_cdfs, method_proba, dur_by_method,
                extra_kwargs, lo=0.85, hi=1.10, n=50) -> tuple[float, float]:
    """Coarse grid + refinement around best."""
    factors = np.linspace(lo, hi, n)
    ks_vals  = [ks_for_factor(model, f, df, y, dur_cdfs, method_proba, dur_by_method, extra_kwargs)
                for f in factors]
    best_i  = int(np.argmin(ks_vals))
    best_f  = factors[best_i]
    best_ks = ks_vals[best_i]

    # Refine in ±5% window around best
    lo2, hi2 = max(lo, best_f - 0.05), min(hi, best_f + 0.05)
    factors2 = np.linspace(lo2, hi2, 30)
    ks_vals2 = [ks_for_factor(model, f, df, y, dur_cdfs, method_proba, dur_by_method, extra_kwargs)
                for f in factors2]
    best_i2  = int(np.argmin(ks_vals2))
    if ks_vals2[best_i2] < best_ks:
        best_f  = factors2[best_i2]
        best_ks = ks_vals2[best_i2]
    return best_f, best_ks


def main():
    print("=== Rate Calib Factor Re-Optimisation (KS-minimising, val set) ===\n")

    model_dir = paths.outputs_models()

    # Load val set
    props_df = parquet.read(paths.processed("features_props"))
    props_df["event_date"] = pd.to_datetime(props_df["event_date"])
    splits = get_splits(props_df)
    val_df = props_df[splits["val"]].copy()
    print(f"Val rows: {len(val_df)}")

    # Method proba on val
    method_files = sorted(model_dir.glob("method_clf_*.joblib"),
                          key=lambda p: p.stat().st_mtime)
    method_clf = MethodClassifier.load(method_files[-1])
    probs_dict = method_clf.predict_proba_dict(val_df)
    method_proba = np.column_stack([probs_dict[c] for c in METHOD_CLASSES])  # (n, 3)
    print(f"Method proba shape: {method_proba.shape}  "
          f"mean KO={method_proba[:,0].mean():.3f}  "
          f"SUB={method_proba[:,1].mean():.3f}  "
          f"DEC={method_proba[:,2].mean():.3f}")

    # Duration CDFs on val
    dur_files = sorted(model_dir.glob("props_duration_*.joblib"),
                       key=lambda p: p.stat().st_mtime)
    dur_model: DurationModel = joblib.load(dur_files[-1])
    dur_cdfs_full = dur_model.predict_cdf(val_df)
    dur_by_method = {
        m: dur_model.predict_cdf(val_df, method_override=m, use_boundary_mass=False)
        for m in ("KO/TKO", "SUB", "DEC")
    }

    # ── sig_strikes ────────────────────────────────────────────────────────
    print("\n--- sig_strikes ---")
    ss_files = sorted(model_dir.glob("props_sig_strikes_*.joblib"),
                      key=lambda p: p.stat().st_mtime)
    ss_path  = ss_files[-1]
    ss_model: RateHurdleCountModel = joblib.load(ss_path)

    y_ss = val_df["sig_str_landed_a"].fillna(0).values
    # Gate uses zero_method_rate_adj=True → we zero adj in ks_for_factor already
    # duration_cdfs_by_method is passed explicitly in ks_for_factor; no extra kwargs needed
    ss_kwargs = {}

    old_f_ss = ss_model.rate_calib_factor
    old_ks_ss = ks_for_factor(ss_model, old_f_ss, val_df, y_ss,
                               dur_cdfs_full, method_proba, dur_by_method, ss_kwargs)
    print(f"Current factor={old_f_ss:.4f}  val KS={old_ks_ss:.4f}")

    best_f_ss, best_ks_ss = grid_search(
        ss_model, val_df, y_ss,
        dur_cdfs_full, method_proba, dur_by_method, ss_kwargs,
    )
    print(f"Best    factor={best_f_ss:.4f}  val KS={best_ks_ss:.4f}")

    if best_ks_ss < old_ks_ss - 0.001:
        ss_model.rate_calib_factor = best_f_ss
        joblib.dump(ss_model, ss_path)
        print(f"  -> Saved updated sig_strikes model ({ss_path.name})")
    else:
        print("  -> No meaningful improvement; keeping current factor")

    # ── r1_sig_strikes ─────────────────────────────────────────────────────
    print("\n--- r1_sig_strikes ---")
    r1_files = sorted(model_dir.glob("props_r1_sig_strikes_*.joblib"),
                      key=lambda p: p.stat().st_mtime)
    r1_path  = r1_files[-1]
    r1_model: RateHurdleCountModel = joblib.load(r1_path)

    y_r1 = val_df["r1_sig_str_landed_a"].fillna(0).values
    # Gate flags: use_finish_head=True, active_minutes_ceiling=5.0, apply_burst=False
    # method_proba passed (v8.9 finish-head conditioning)
    r1_kwargs = dict(
        active_minutes_ceiling=5.0,
        apply_burst=False,
        use_finish_head=True,
    )

    old_f_r1 = r1_model.rate_calib_factor
    old_ks_r1 = ks_for_factor(r1_model, old_f_r1, val_df, y_r1,
                               dur_cdfs_full, method_proba, dur_by_method, r1_kwargs)
    print(f"Current factor={old_f_r1:.4f}  val KS={old_ks_r1:.4f}")

    best_f_r1, best_ks_r1 = grid_search(
        r1_model, val_df, y_r1,
        dur_cdfs_full, method_proba, dur_by_method, r1_kwargs,
    )
    print(f"Best    factor={best_f_r1:.4f}  val KS={best_ks_r1:.4f}")

    if best_ks_r1 < old_ks_r1 - 0.001:
        r1_model.rate_calib_factor = best_f_r1
        joblib.dump(r1_model, r1_path)
        print(f"  -> Saved updated r1_sig_strikes model ({r1_path.name})")
    else:
        print("  -> No meaningful improvement; keeping current factor")

    print("\n=== Done. Re-run 05_evaluate_props.py to verify gate results. ===")


if __name__ == "__main__":
    main()
