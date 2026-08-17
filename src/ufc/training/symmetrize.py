"""Symmetric training: create (B,A) flip rows to eliminate corner bias."""
from __future__ import annotations

import pandas as pd
import numpy as np


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """Double the dataset by creating flipped (B,A) rows.

    For each row with fighter A vs B:
    - Original row: A=perspective, label=won_a
    - Flipped row: B=perspective (A becomes B), label=1-won_a

    Columns with _a / _b suffixes are swapped.
    Delta features with _diff suffix are negated.
    """
    df_orig = df.copy()
    df_orig["_is_flipped"] = False

    df_flip = df.copy()
    df_flip["_is_flipped"] = True

    # Get all _a / _b column pairs
    a_cols = [c for c in df.columns if c.endswith("_a") and c[:-2] + "_b" in df.columns]
    b_cols = [c[:-2] + "_b" for c in a_cols]

    # Swap _a and _b columns in the flipped frame
    for a_col, b_col in zip(a_cols, b_cols):
        df_flip[a_col] = df[b_col].values
        df_flip[b_col] = df[a_col].values

    # Negate diff columns
    diff_cols = [c for c in df.columns if c.endswith("_diff") or c == "reach_diff"
                 or c == "height_diff" or c == "age_diff" or c == "weight_diff"]
    for dc in diff_cols:
        if dc in df_flip.columns:
            df_flip[dc] = -df_flip[dc].values

    # Flip label
    if "won_a" in df_flip.columns:
        df_flip["won_a"] = (1 - df_flip["won_a"].astype(float)).where(
            df_flip["won_a"].notna(), other=np.nan
        )

    # Flip other A/B-directional features
    flip_signed = [
        # NOTE: do not list any "_diff" cols here — already negated above.
        # NOTE: _a/_b cols are already swapped by the suffix-swap loop above.
        "cardio_gap_5rd",
    ]
    for col in flip_signed:
        if col in df_flip.columns:
            df_flip[col] = -df_flip[col].values

    combined = pd.concat([df_orig, df_flip], ignore_index=True)
    return combined


def _smoke_check():
    """Self-test: symmetrize must invert sign on _diff cols and swap _a/_b cols."""
    test = pd.DataFrame({
        "fight_id": ["F1"],
        "won_a": [1.0],
        "pace_diff": [5.0],
        "elo_diff": [100.0],
        "reach_diff": [-2.0],
        "cardio_gap_5rd": [0.3],
        "slpm_decay_a": [4.0], "slpm_decay_b": [3.0],
        "sub_trap_a": [1.5], "sub_trap_b": [0.5],
    })
    out = symmetrize(test)
    flipped = out.iloc[1]
    assert flipped["won_a"] == 0.0, "won_a flip broken"
    assert flipped["pace_diff"] == -5.0, f"pace_diff = {flipped['pace_diff']} (expected -5.0)"
    assert flipped["elo_diff"] == -100.0
    assert flipped["reach_diff"] == 2.0
    assert flipped["cardio_gap_5rd"] == -0.3
    assert flipped["slpm_decay_a"] == 3.0 and flipped["slpm_decay_b"] == 4.0
    assert flipped["sub_trap_a"] == 0.5 and flipped["sub_trap_b"] == 1.5
    print("symmetrize smoke OK")


if __name__ == "__main__":
    _smoke_check()


def inference_average(prob_a_view: float, prob_b_view: float) -> float:
    """Average the A-wins probability from both orderings.

    prob_a_view: P(A wins | features from A perspective)
    prob_b_view: P(B wins | features from B perspective) = P(A wins complement)
    Returns: symmetric estimate of P(A wins).
    """
    # prob_b_view is P(B wins), so 1 - prob_b_view is P(A wins from B perspective)
    return (prob_a_view + (1.0 - prob_b_view)) / 2.0
