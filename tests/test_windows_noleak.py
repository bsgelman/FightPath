"""THE most critical test: verify no temporal data leakage in rolling features.

Two tests:
1. Manual recomputation — rolling feature at index i matches manual computation
   from strict prefix {0..i-1}.
2. Mutation test — mutating a fight's stats does NOT change its own feature value.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from ufc.features.windows import (
    causal_expanding, causal_rolling, causal_decay, causal_date_window
)


def _make_fighter_df(n_fights=10, n_fighters=3, seed=42) -> pd.DataFrame:
    """Create synthetic per-fighter ledger with known values."""
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp("2010-01-01")
    rank = 0
    for fid in range(n_fighters):
        for i in range(n_fights):
            rows.append({
                "fighter_id": str(fid),
                "event_rank": rank,
                "event_date": base_date + pd.Timedelta(days=i * 90 + fid * 30),
                "sig_str_landed": rng.integers(10, 100),
            })
            rank += 1
    return pd.DataFrame(rows)


class TestCausalExpanding:
    def test_manual_recomputation(self):
        df = _make_fighter_df(n_fights=8, n_fighters=2)
        result = causal_expanding(df, by="fighter_id", sort_col="event_rank",
                                  value_col="sig_str_landed")
        result = result.reindex(df.index)

        for fid, grp in df.groupby("fighter_id"):
            grp_sorted = grp.sort_values("event_rank")
            vals = grp_sorted["sig_str_landed"].values

            for i, idx in enumerate(grp_sorted.index):
                expected = float(np.mean(vals[:i])) if i > 0 else np.nan
                actual = result.loc[idx]
                if i == 0:
                    assert np.isnan(actual), f"First fight should be NaN, got {actual}"
                else:
                    assert abs(actual - expected) < 1e-9, \
                        f"Fighter {fid}, fight {i}: expected {expected}, got {actual}"

    def test_mutation_does_not_affect_current_row(self):
        df = _make_fighter_df(n_fights=6, n_fighters=2)

        # Compute original features
        result_orig = causal_expanding(df, by="fighter_id", sort_col="event_rank",
                                       value_col="sig_str_landed")

        # Pick a fight at index 3 of fighter 0
        fid = "0"
        grp = df[df["fighter_id"] == fid].sort_values("event_rank")
        target_idx = grp.index[3]

        # Mutate that fight's value
        df_mutated = df.copy()
        df_mutated.loc[target_idx, "sig_str_landed"] = 9999

        result_mutated = causal_expanding(df_mutated, by="fighter_id", sort_col="event_rank",
                                          value_col="sig_str_landed")

        # The feature AT target_idx must not change (it uses only rows 0..2)
        orig_val = result_orig.loc[target_idx]
        mut_val = result_mutated.loc[target_idx]
        assert abs(orig_val - mut_val) < 1e-9, \
            f"Mutation leaked into current row! Before={orig_val}, After={mut_val}"

        # Downstream rows (4, 5) SHOULD change because they use fight 3 as history
        later_idx = grp.index[4]
        orig_later = result_orig.loc[later_idx]
        mut_later = result_mutated.loc[later_idx]
        assert abs(orig_later - mut_later) > 0.1, \
            "Downstream rows should change after mutation"


class TestCausalRolling:
    def test_window_3_manual(self):
        df = _make_fighter_df(n_fights=8, n_fighters=1)
        result = causal_rolling(df, by="fighter_id", sort_col="event_rank",
                                 value_col="sig_str_landed", window=3)
        result = result.reindex(df.index)

        grp = df.sort_values("event_rank")
        vals = grp["sig_str_landed"].values
        indices = grp.index

        # Fight 0: NaN (no prior)
        assert np.isnan(result.loc[indices[0]])
        # Fight 1: mean(vals[0])
        assert abs(result.loc[indices[1]] - vals[0]) < 1e-9
        # Fight 3: mean(vals[0:3])
        assert abs(result.loc[indices[3]] - np.mean(vals[0:3])) < 1e-9
        # Fight 5: mean(vals[2:5]) — window=3
        assert abs(result.loc[indices[5]] - np.mean(vals[2:5])) < 1e-9


class TestCausalDecay:
    def test_decay_first_row_nan(self):
        df = _make_fighter_df(n_fights=5, n_fighters=1)
        result = causal_decay(df, by="fighter_id", date_col="event_date",
                              value_col="sig_str_landed", halflife_days=180)
        grp = df.sort_values("event_rank")
        first_idx = grp.index[0]
        assert np.isnan(result.loc[first_idx])

    def test_decay_recent_fight_weighted_more(self):
        # Create a fighter with two fights, second much larger value
        rows = [
            {"fighter_id": "X", "event_rank": 0,
             "event_date": pd.Timestamp("2010-01-01"), "sig_str_landed": 10},
            {"fighter_id": "X", "event_rank": 1,
             "event_date": pd.Timestamp("2010-07-01"), "sig_str_landed": 100},
            {"fighter_id": "X", "event_rank": 2,
             "event_date": pd.Timestamp("2011-01-01"), "sig_str_landed": 50},
        ]
        df = pd.DataFrame(rows)
        result = causal_decay(df, by="fighter_id", date_col="event_date",
                              value_col="sig_str_landed", halflife_days=180)
        # At fight 2 (rank 2): should use fights 0 and 1
        # Fight 1 (100) is ~6 months ago, fight 0 (10) is ~12 months ago
        # With 180-day halflife: w1 = 0.5^(6/6)=0.5, w0 = 0.5^(12/6)=0.25
        # decay = (0.5*100 + 0.25*10) / (0.5+0.25) = 52.5/0.75 = 70
        val_at_2 = result.iloc[2]
        assert abs(val_at_2 - 70.0) < 2.0, f"Expected ~70, got {val_at_2}"


class TestNoLeakInAllFlavors:
    def test_all_window_flavors_no_leakage(self):
        from ufc.features.windows import all_window_flavors
        df = _make_fighter_df(n_fights=7, n_fighters=2)
        flavors = all_window_flavors(
            df, by="fighter_id", sort_col="event_rank",
            date_col="event_date", value_col="sig_str_landed",
            halflife_days=180, window_24mo_days=730,
        )
        for name, series in flavors.items():
            series = series.reindex(df.index)
            # For each fighter, first fight must be NaN or 0 (not computed from future)
            for fid, grp in df.groupby("fighter_id"):
                grp_sorted = grp.sort_values("event_rank")
                first_val = series.loc[grp_sorted.index[0]]
                assert np.isnan(first_val) or first_val == 0.0, \
                    f"{name}: first fight of fighter {fid} should be NaN/0, got {first_val}"
