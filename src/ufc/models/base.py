"""Base utilities for model wrappers and calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


def isotonic_calibrate(y_pred_proba: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression calibrator on validation predictions."""
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(y_pred_proba, y_true)
    return cal


def calibrate_proba(model, X_val, y_val, method="isotonic"):
    """Fit and return a calibrated wrapper."""
    cal = CalibratedClassifierCV(model, cv="prefit", method=method)
    cal.fit(X_val, y_val)
    return cal


def get_feature_cols(df: pd.DataFrame, exclude_patterns: list[str] | None = None) -> list[str]:
    """Get numeric feature column names, excluding targets and IDs."""
    default_exclude = [
        "fight_id", "event_id", "fighter_id_a", "fighter_id_b",
        "event_date", "event_rank", "won_a", "method", "end_round",
        "end_time_sec", "total_fight_sec",
        # Post-fight raw outcome stats — would leak target into features
        "sig_str_landed_a", "sig_str_landed_b",
        "td_landed_a", "td_landed_b",
        "ctrl_sec_a", "ctrl_sec_b",
        "ctrl_sec_absorbed_a", "ctrl_sec_absorbed_b",
        "sub_att_for_a", "sub_att_for_b",
        "sub_att_against_a", "sub_att_against_b",
        "r1_sig_str_landed_a", "r1_sig_str_landed_b",
        "r1_td_landed_a", "r1_td_landed_b",
        # New prop target labels — unknown at inference time
        "kd_for_a", "kd_for_b",
        "body_landed_a", "body_landed_b",
        "leg_landed_a", "leg_landed_b",
        # Categorical / passthrough
        "weight_class", "stance_pair", "referee", "location",
        # Post-fight outcome descriptor (leak) — bool dtype skipped by numeric filter,
        # explicit entry survives dtype changes and documents intent
        "injury_freak",
    ]
    if exclude_patterns:
        default_exclude.extend(exclude_patterns)

    cols = []
    for c in df.columns:
        if c in default_exclude:
            continue
        if any(c.startswith(pat) for pat in ["stance_", "ref_", "method_"]):
            continue
        if df[c].dtype in [np.float64, np.float32, np.int64, np.int32, "Int8", "Int16", "Int32"]:
            cols.append(c)
        elif df[c].dtype == object:
            # Allow categorical features explicitly
            pass
    return cols
