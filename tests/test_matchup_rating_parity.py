"""Pairwise rating features must reflect the requested pairing, not the last opponent."""
import sys
sys.path.insert(0, "src")
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ufc.inference.matchup import build_matchup_features

# build_matchup_features reads the built feature parquet for its column schema,
# and that dataset is not distributed with the repo (see DATA.md). Skips cleanly
# on a fresh clone; runs whenever the data is present.
_FEATURES = Path(__file__).parents[1] / "data" / "processed" / "features_props.parquet"
pytestmark = pytest.mark.skipif(not _FEATURES.exists(),
                                reason="built feature dataset not distributed")


def _pfs():
    """Minimal 2-fighter pre_fight_state with deliberately WRONG stale pairwise
    fields (as if each last fought someone else)."""
    base = dict(
        event_date=pd.Timestamp("2026-01-01"), event_rank=100,
        weight_class="Bantamweight", stance="ORTHO",
        age_years=30.0, reach_in=68.0, height_in=68.0, weight_lbs=135.0,
    )
    return pd.DataFrame([
        dict(fighter_id="A", fight_id="sentinel_A", elo_pre=1600.0,
             opp_elo_pre=np.nan, elo_diff=np.nan,
             glicko_mu_pre=1620.0, glicko_rd_pre=80.0, glicko_z=np.nan,
             ts_mu_pre=27.0, ts_sigma_pre=4.0, ts_z=np.nan, **base),
        dict(fighter_id="B", fight_id="sentinel_B", elo_pre=1500.0,
             opp_elo_pre=np.nan, elo_diff=np.nan,
             glicko_mu_pre=1480.0, glicko_rd_pre=120.0, glicko_z=np.nan,
             ts_mu_pre=24.0, ts_sigma_pre=6.0, ts_z=np.nan, **base),
    ])


def _fighters():
    return pd.DataFrame([
        dict(fighter_id="A", fighter_name="Fighter A", dob=pd.Timestamp("1996-01-01"),
             reach_in=68.0, height_in=68.0, weight_lbs=135.0, stance="ORTHO"),
        dict(fighter_id="B", fighter_name="Fighter B", dob=pd.Timestamp("1994-01-01"),
             reach_in=70.0, height_in=69.0, weight_lbs=135.0, stance="SOUTH"),
    ])


def test_pairwise_ratings_recomputed_for_actual_pairing():
    feat = build_matchup_features(
        "A", "B", date(2026, 7, 11), 3, False,
        pre_fight_state=_pfs(), fighters_df=_fighters(),
    )
    r = feat.iloc[0]
    assert r["opp_elo_pre_a"] == 1500.0
    assert r["opp_elo_pre_b"] == 1600.0
    assert r["elo_diff_a"] == 100.0
    assert r["elo_diff_b"] == -100.0
    gz = (1620.0 - 1480.0) / math.sqrt(80.0**2 + 120.0**2)
    assert abs(r["glicko_z_a"] - gz) < 1e-9
    assert abs(r["glicko_z_b"] + gz) < 1e-9
    tz = (27.0 - 24.0) / math.sqrt(4.0**2 + 6.0**2)
    assert abs(r["ts_z_a"] - tz) < 1e-9
    assert abs(r["ts_z_b"] + tz) < 1e-9
