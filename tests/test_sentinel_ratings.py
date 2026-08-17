"""Single-fighter (sentinel) fight groups must receive pre-ratings, no update."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import pytest

from ufc.features.ratings import compute_elo, compute_glicko2, compute_trueskill


def _ledger():
    """Two fighters, one real fight (f1 beats f2), then one sentinel row each."""
    rows = [
        # real fight — long-form: one row per (fight, fighter)
        dict(fight_id="F1", event_date=pd.Timestamp("2025-01-01"),
             fighter_id="f1", opponent_id="f2", won=1.0, method="KO/TKO"),
        dict(fight_id="F1", event_date=pd.Timestamp("2025-01-01"),
             fighter_id="f2", opponent_id="f1", won=0.0, method="KO/TKO"),
        # sentinels — single-fighter fights, no outcome
        dict(fight_id="sentinel_f1", event_date=pd.Timestamp("2025-01-02"),
             fighter_id="f1", opponent_id=None, won=np.nan, method=None),
        dict(fight_id="sentinel_f2", event_date=pd.Timestamp("2025-01-02"),
             fighter_id="f2", opponent_id=None, won=np.nan, method=None),
    ]
    return pd.DataFrame(rows)


def test_elo_sentinel_gets_post_fight_rating():
    out = compute_elo(_ledger())
    s1 = out[out["fight_id"] == "sentinel_f1"].iloc[0]
    s2 = out[out["fight_id"] == "sentinel_f2"].iloc[0]
    r1 = out[(out["fight_id"] == "F1") & (out["fighter_id"] == "f1")].iloc[0]
    # real fight row: both start at initial rating -> elo_pre equal
    assert s1["elo_pre"] > r1["elo_pre"]          # winner's rating went UP after F1
    assert s2["elo_pre"] < r1["elo_pre"]          # loser's went DOWN
    assert pd.isna(s1["opp_elo_pre"])             # no opponent on a sentinel


def test_glicko_sentinel_gets_current_mu():
    out = compute_glicko2(_ledger())
    s1 = out[out["fight_id"] == "sentinel_f1"].iloc[0]
    s2 = out[out["fight_id"] == "sentinel_f2"].iloc[0]
    assert s1["glicko_mu_pre"] > s2["glicko_mu_pre"]   # winner > loser post-update
    assert pd.notna(s1["glicko_rd_pre"])
    assert pd.isna(s1["glicko_z"])                     # pairwise field NaN


def test_trueskill_sentinel_gets_current_mu():
    trueskill = pytest.importorskip("trueskill")
    out = compute_trueskill(_ledger())
    s1 = out[out["fight_id"] == "sentinel_f1"].iloc[0]
    s2 = out[out["fight_id"] == "sentinel_f2"].iloc[0]
    assert s1["ts_mu_pre"] > s2["ts_mu_pre"]
    assert pd.isna(s1["ts_z"])


def test_real_rows_unchanged_by_sentinels():
    """Adding sentinel rows must not change any real row's ratings."""
    with_s = compute_elo(_ledger())
    without_s = compute_elo(_ledger()[~_ledger()["fight_id"].str.startswith("sentinel")])
    for fid, f in [("F1", "f1"), ("F1", "f2")]:
        a = with_s[(with_s["fight_id"] == fid) & (with_s["fighter_id"] == f)].iloc[0]
        b = without_s[(without_s["fight_id"] == fid) & (without_s["fighter_id"] == f)].iloc[0]
        assert a["elo_pre"] == b["elo_pre"]
        assert a["opp_elo_pre"] == b["opp_elo_pre"]
