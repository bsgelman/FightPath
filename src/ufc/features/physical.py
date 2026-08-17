"""Physical / biographical delta features."""
from __future__ import annotations

import pandas as pd


def compute_physical(ledger_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute physical delta features from a wide (A vs B) fight-level DataFrame.

    ledger_wide has both A and B fighter columns with _a / _b suffixes.
    Returns the same DataFrame with physical delta columns added.
    """
    df = ledger_wide.copy()

    # Reach differential (A - B)
    if "reach_in_a" in df.columns and "reach_in_b" in df.columns:
        df["reach_diff"] = df["reach_in_a"].fillna(70) - df["reach_in_b"].fillna(70)
    else:
        df["reach_diff"] = 0.0

    # Height differential
    if "height_in_a" in df.columns and "height_in_b" in df.columns:
        df["height_diff"] = df["height_in_a"].fillna(70) - df["height_in_b"].fillna(70)
    else:
        df["height_diff"] = 0.0

    # Age differential
    if "age_years_a" in df.columns and "age_years_b" in df.columns:
        df["age_diff"] = df["age_years_a"].fillna(30) - df["age_years_b"].fillna(30)
    else:
        df["age_diff"] = 0.0

    # Weight differential (usually 0 within weight class, nonzero for catch-weights)
    if "weight_lbs_a" in df.columns and "weight_lbs_b" in df.columns:
        df["weight_diff"] = df["weight_lbs_a"].fillna(155) - df["weight_lbs_b"].fillna(155)
    else:
        df["weight_diff"] = 0.0

    # Stance pair
    if "stance_a" in df.columns and "stance_b" in df.columns:
        def _stance_pair(s_a, s_b):
            a = str(s_a)[:1].upper() if pd.notna(s_a) else "U"
            b = str(s_b)[:1].upper() if pd.notna(s_b) else "U"
            # O=Orthodox, S=Southpaw, W=Switch, P=Open
            def _abbr(s):
                if "SOUTH" in s.upper():
                    return "S"
                if "SWITCH" in s.upper():
                    return "W"
                if "OPEN" in s.upper():
                    return "P"
                return "O"  # default Orthodox
            return f"{_abbr(str(s_a))}_{_abbr(str(s_b))}"

        df["stance_pair"] = df.apply(
            lambda r: _stance_pair(r.get("stance_a"), r.get("stance_b")), axis=1
        )
        df["is_southpaw_vs_orthodox"] = (df["stance_pair"] == "S_O") | (df["stance_pair"] == "O_S")
    else:
        df["stance_pair"] = "O_O"
        df["is_southpaw_vs_orthodox"] = False

    return df
