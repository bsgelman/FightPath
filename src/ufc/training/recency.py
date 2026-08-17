"""Val-based halflife search for recency weighting.

Provides search_halflife_method and search_halflife_winner.  Both run a
grid of candidate halflives, score on the val set.

Winner uses an ECE guardrail (avoid Step-8 failure: halflife=1095d
degraded winner ECE 0.032->0.038 and DEC ECE 0.047->0.061).

Method uses a Brier-score guardrail: candidate must beat the uniform-
weight baseline by brier_floor_margin.  ECE is not a valid guard for the
two-stage product-probability architecture (P(KO)=P(fin)*P(KO|fin) is not
inherently calibrated one-vs-rest), so ECE rejects all candidates.

If no candidate beats the baseline the search falls back to None
(uniform weighting) and callers train without sample_weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ufc import SEED
from ufc.evaluation.metrics import expected_calibration_error


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_recency_weights(
    dates: pd.Series,
    anchor: str,
    halflife_days: float | None,
    floor: float = 0.05,
) -> np.ndarray:
    """Per-row exponential decay weight.

    w = 0.5 ** (days_to_anchor / halflife_days), clipped to [floor, 1].
    If halflife_days is None, returns all-ones (uniform).
    """
    anchor_dt = pd.to_datetime(anchor)
    days_old = (anchor_dt - pd.to_datetime(dates)).dt.days.clip(lower=0).astype(float)
    if halflife_days is None:
        return np.ones(len(days_old))
    return np.power(0.5, days_old / halflife_days).clip(lower=floor).values


def _multiclass_brier(probs: np.ndarray, y_int: np.ndarray, n_classes: int = 3) -> float:
    """Multiclass Brier score (mean squared error over one-hot targets)."""
    y_oh = np.eye(n_classes)[np.clip(y_int, 0, n_classes - 1)]
    return float(np.mean((probs - y_oh) ** 2))


def _per_class_ece(probs: np.ndarray, y_int: np.ndarray, n_classes: int = 3) -> list[float]:
    """Per-class binary ECE (one-vs-rest)."""
    ece_list = []
    for c in range(n_classes):
        y_bin = (y_int == c).astype(float)
        ece_list.append(expected_calibration_error(y_bin, probs[:, c]))
    return ece_list


# ---------------------------------------------------------------------------
# Method halflife search
# ---------------------------------------------------------------------------

def _temporal_oof_probe_mask(train_dates: pd.Series, months: int = 6,
                              min_rows: int = 30) -> np.ndarray:
    """Boolean mask (len == len(train_dates)) marking a recent tail-of-train
    holdout — true out-of-sample even in prod mode, where val ⊂ train.
    Mirrors the pattern in WinnerModel.fit / MethodClassifier.fit (Step 1 /
    base-fit early-stopping guards)."""
    dates_vec = pd.to_datetime(train_dates).values
    cutoff = (dates_vec.max() - pd.DateOffset(months=months)).to_datetime64()
    mask = dates_vec >= cutoff
    if mask.sum() < min_rows:
        mask = np.zeros(len(dates_vec), dtype=bool)
        mask[np.argsort(dates_vec)[-min_rows:]] = True
    return mask


def search_halflife_method(
    X_train: pd.DataFrame,
    y_train_method: pd.Series,
    X_val: pd.DataFrame,
    y_val_method: pd.Series,
    method_feature_cols: list[str],
    train_dates: pd.Series,
    grid: list[float | None] | None = None,
    anchor: str = "2023-12-31",
    brier_floor_margin: float = 0.001,
    verbose: bool = True,
    temporal_oof: bool = False,
) -> tuple[float | None, np.ndarray | None]:
    """Search for the best recency halflife for the method classifier.

    Uses a fast 3-class LGBM probe fitted on train-only, evaluated on held-out
    val (true out-of-sample).  Mirrors search_halflife_winner; avoids the in-
    sample bias that occurs when the full MethodClassifier.fit() is used as a
    probe (it includes val in training via the CalibratedClassifierCV step).

    Selects by multiclass Brier score; candidate must beat uniform baseline by
    brier_floor_margin.  ECE not used (product-probability architecture).

    temporal_oof : bool
        Prod tier. X_val/y_val are in-sample (val ⊂ train) in prod mode — the
        probe would be scored on data it can already see. When True, score
        against a temporal holdout carved from the tail of train instead.

    Returns (best_halflife, sample_weights_for_train). sample_weights_for_train
    is always full-length (len(X_train)) regardless of temporal_oof.
    If no candidate beats the baseline, returns (None, None).
    """
    from ufc.models.method import METHOD_MAP

    if grid is None:
        grid = [365, 730, 1095, 1460, None]

    feat_cols = [c for c in method_feature_cols if c in X_train.columns and c in X_val.columns]
    X_tr = X_train[feat_cols].fillna(0)
    X_vl = X_val[feat_cols].fillna(0)
    y_tr_int = y_train_method.map(METHOD_MAP).fillna(2).astype(int).values
    y_vl_int = y_val_method.map(METHOD_MAP).fillna(2).astype(int).values

    if temporal_oof and train_dates is not None:
        oof_val_mask = _temporal_oof_probe_mask(train_dates)
        oof_tr_mask = ~oof_val_mask
        probe_X_tr, probe_y_tr_int = X_tr[oof_tr_mask], y_tr_int[oof_tr_mask]
        probe_X_vl, probe_y_vl_int = X_tr[oof_val_mask], y_tr_int[oof_val_mask]
    else:
        oof_tr_mask = None
        probe_X_tr, probe_y_tr_int = X_tr, y_tr_int
        probe_X_vl, probe_y_vl_int = X_vl, y_vl_int

    # Baseline: uniform weights (train-only fit → true out-of-sample val)
    baseline_brier = _probe_method_brier(probe_X_tr, probe_y_tr_int, probe_X_vl, probe_y_vl_int)
    best_brier = baseline_brier - brier_floor_margin
    best_h: float | None = None
    best_w: np.ndarray | None = None

    if verbose:
        print(f"  [halflife-search/method] grid={grid}, anchor={anchor}, brier_floor_margin={brier_floor_margin}")
        print(f"    baseline (uniform) val_brier={baseline_brier:.4f}")

    for h in grid:
        if h is None:
            continue  # already evaluated as baseline
        w = compute_recency_weights(train_dates, anchor, h)  # full-length, len(X_train)
        w_probe = w[oof_tr_mask] if oof_tr_mask is not None else w
        mb = _probe_method_brier(probe_X_tr, probe_y_tr_int, probe_X_vl, probe_y_vl_int, w=w_probe)

        if verbose:
            flag = " [new best]" if mb < best_brier else ""
            print(f"    h={str(h):>6} -> val_brier={mb:.4f}{flag}")

        if mb < best_brier:
            best_brier = mb
            best_h = h
            best_w = w

    if best_h is None and verbose:
        print("  [halflife-search/method] No candidate beat baseline — using uniform weights")

    if verbose and best_h is not None:
        print(f"  [halflife-search/method] Selected halflife={best_h} (val_brier={best_brier:.4f})")

    return best_h, best_w


def _method_probe_params() -> dict:
    """Single-seed 3-class LGBM params for method halflife probe."""
    import yaml
    from ufc.io import paths
    with open(paths.root() / "configs" / "model_props.yaml") as f:
        cfg = yaml.safe_load(f)["method_clf"]
    return {
        "n_estimators": cfg["n_estimators"],
        "num_leaves": cfg["num_leaves"],
        "learning_rate": cfg["learning_rate"],
        "verbosity": -1,
        "random_state": SEED,
        "objective": "multiclass",
        "num_class": 3,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
    }


def _probe_method_brier(X_tr, y_tr_int, X_vl, y_vl_int, w=None) -> float:
    """Train 3-class probe on train only, return out-of-sample val multiclass Brier."""
    import lightgbm as lgb
    params = _method_probe_params()
    m = lgb.LGBMClassifier(**params)
    sw_kw = {"sample_weight": w} if w is not None else {}
    m.fit(X_tr.values, y_tr_int, **sw_kw)
    p = m.predict_proba(X_vl.values)
    if p.shape[1] < 3:
        p = np.hstack([p, np.zeros((len(p), 3 - p.shape[1]))])
    return _multiclass_brier(p, y_vl_int)


def _get_method_probs(clf, X_val: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Get (n, 3) calibrated probability array from MethodClassifier."""
    probs_dict = clf.predict_proba_dict(X_val)
    classes = ["KO/TKO", "SUB", "DEC"]
    return np.column_stack([probs_dict[c] for c in classes])


# ---------------------------------------------------------------------------
# Winner halflife search
# ---------------------------------------------------------------------------

def search_halflife_winner(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
    train_dates: pd.Series,
    grid: list[float | None] | None = None,
    anchor: str = "2023-12-31",
    ece_cap: float = 0.04,
    brier_floor_margin: float = 0.001,
    verbose: bool = True,
    temporal_oof: bool = False,
) -> tuple[float | None, np.ndarray | None]:
    """Search for the best recency halflife for the winner model.

    Uses a single-seed probe LGBM (not the full 5-seed ensemble) to keep
    the search affordable.  Applies ECE guardrail (ece_cap) to prevent
    the Step-8 failure mode.

    temporal_oof : bool
        Prod tier. X_val/y_val are in-sample (val ⊂ train) in prod mode — the
        probe would be scored on data it can already see. When True, score
        against a temporal holdout carved from the tail of train instead.

    Returns (best_halflife, sample_weights_for_train). sample_weights_for_train
    is always full-length (len(X_train)) regardless of temporal_oof.
    If no candidate passes the guardrail, returns (None, None) and the
    caller should train without sample_weight.
    """
    if grid is None:
        grid = [730, 1095, 1460, 1825, None]

    X_tr = X_train[feature_cols].fillna(0)
    X_vl = X_val[feature_cols].fillna(0)
    y_tr = y_train.values.astype(float)
    y_vl = y_val.values.astype(float)

    if temporal_oof and train_dates is not None:
        oof_val_mask = _temporal_oof_probe_mask(train_dates)
        oof_tr_mask = ~oof_val_mask
        probe_X_tr, probe_y_tr = X_tr[oof_tr_mask], y_tr[oof_tr_mask]
        probe_X_vl, probe_y_vl = X_tr[oof_val_mask], y_tr[oof_val_mask]
    else:
        oof_tr_mask = None
        probe_X_tr, probe_y_tr = X_tr, y_tr
        probe_X_vl, probe_y_vl = X_vl, y_vl

    # baseline: uniform weights
    baseline_brier = _probe_winner_brier(probe_X_tr, probe_y_tr, probe_X_vl, probe_y_vl)
    best_brier = baseline_brier - brier_floor_margin  # must beat baseline by margin
    best_h: float | None = None
    best_w: np.ndarray | None = None

    if verbose:
        print(f"  [halflife-search/winner] grid={grid}, anchor={anchor}, ece_cap={ece_cap}")
        print(f"    baseline (uniform) val_brier={baseline_brier:.4f}")

    for h in grid:
        if h is None:
            continue  # already computed as baseline
        w = compute_recency_weights(train_dates, anchor, h)  # full-length, len(X_train)
        w_probe = w[oof_tr_mask] if oof_tr_mask is not None else w
        brier, ece = _probe_winner_brier_ece(probe_X_tr, probe_y_tr, probe_X_vl, probe_y_vl, w_probe)

        if verbose:
            flag = "" if ece <= ece_cap else " [ECE FAIL]"
            print(f"    h={str(h):>6} -> val_brier={brier:.4f}  ECE={ece:.4f}{flag}")

        if ece > ece_cap:
            continue
        if brier < best_brier:
            best_brier = brier
            best_h = h
            best_w = w

    if best_h is None and verbose:
        print("  [halflife-search/winner] No candidate beat baseline + passed ECE — using uniform weights")

    if verbose and best_h is not None:
        print(f"  [halflife-search/winner] Selected halflife={best_h} (val_brier={best_brier:.4f})")

    return best_h, best_w


def _probe_winner_brier(X_tr, y_tr, X_vl, y_vl, w=None):
    """Train a 1-seed probe LightGBM and return val Brier."""
    import lightgbm as lgb
    from sklearn.metrics import brier_score_loss

    params = _winner_probe_params()
    m = lgb.LGBMClassifier(**params)
    sw_kw = {"sample_weight": w} if w is not None else {}
    m.fit(X_tr.values, y_tr,
          eval_set=[(X_vl.values, y_vl)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
          **sw_kw)
    p = m.predict_proba(X_vl.values)[:, 1]
    return float(brier_score_loss(y_vl, p))


def _probe_winner_brier_ece(X_tr, y_tr, X_vl, y_vl, w):
    """Train a 1-seed probe, return (val_brier, val_ece)."""
    import lightgbm as lgb
    from sklearn.metrics import brier_score_loss

    params = _winner_probe_params()
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr.values, y_tr,
          eval_set=[(X_vl.values, y_vl)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
          sample_weight=w)
    p = m.predict_proba(X_vl.values)[:, 1]
    brier = float(brier_score_loss(y_vl, p))
    ece = expected_calibration_error(y_vl, p)
    return brier, ece


def _winner_probe_params() -> dict:
    """Single-seed LightGBM params for probe (mirrors WinnerModel _LGBM_PARAMS)."""
    from ufc.models.winner import _LGBM_PARAMS
    params = {**_LGBM_PARAMS, "objective": "binary", "verbosity": -1, "random_state": SEED}
    return params
