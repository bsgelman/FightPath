# tests/test_injury_flags.py
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ufc.ingest.parse_scraper import _injury_freak_flags

CSV = Path(__file__).parents[1] / "data" / "raw" / "manual" / "injury_stoppages.csv"
# Needs the built parquet dataset, which is not distributed with the repo
# (see DATA.md). Skips cleanly on a fresh clone; runs whenever data is present.
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="curation CSV not distributed")

def test_curation_csv_valid():
    df = pd.read_csv(CSV, dtype={"fight_id": str})
    assert list(df.columns) == ["fight_id", "detail_text", "injury_type", "freak", "note"]
    assert df["fight_id"].is_unique and df["fight_id"].notna().all()
    assert df["freak"].isin([0, 1]).all()
    assert df["injury_type"].isin(["arm", "leg", "knee", "eye", "rib", "other"]).all()
    assert (df["note"].astype(str).str.len() > 3).all(), "every row needs a rationale"
    pantoja = df[df["fight_id"] == "dfa692db6d39330c"]
    assert len(pantoja) == 1 and pantoja["freak"].iloc[0] == 1

def test_unsure_convention_machine_enforced():
    # UNSURE rows must pair freak=0 with a literal "UNSURE:" note prefix,
    # in BOTH directions (no freak=1 UNSURE, no lowercase/loose variants).
    df = pd.read_csv(CSV, dtype={"fight_id": str})
    unsure_prefix = df["note"].astype(str).str.startswith("UNSURE:")
    assert (df.loc[unsure_prefix, "freak"] == 0).all()
    loose = df["note"].astype(str).str.lower().str.contains("unsure") & ~unsure_prefix
    assert not loose.any(), df.loc[loose, "fight_id"].tolist()


def _merged(details, fight_ids):
    return pd.DataFrame({"DETAILS": details, "fight_id": fight_ids})


def test_flags_curated_freak_and_combat():
    cur = pd.DataFrame({"fight_id": ["aaa", "bbb"], "freak": [1, 0]})
    m = _merged(["Arm injury to X", "Leg injury", "Punches"], ["aaa", "bbb", "ccc"])
    flags = _injury_freak_flags(m, curation=cur)
    assert flags.tolist() == [True, False, False]


def test_uncurated_keyword_row_defaults_freak(capsys):
    cur = pd.DataFrame({"fight_id": [], "freak": []})
    m = _merged(["Knee Injury", "Punches"], ["new1", "ccc"])
    flags = _injury_freak_flags(m, curation=cur)
    assert flags.tolist() == [True, False]
    assert "injury-keyword rows not in curation CSV: 1" in capsys.readouterr().out


def test_tripwire_prints_zero(capsys):
    cur = pd.DataFrame({"fight_id": ["aaa"], "freak": [1]})
    flags = _injury_freak_flags(_merged(["Arm injury"], ["aaa"]), curation=cur)
    assert "injury-keyword rows not in curation CSV: 0" in capsys.readouterr().out


def test_injury_freak_never_a_model_feature():
    from ufc.models.base import get_feature_cols
    df = pd.DataFrame({"injury_freak": [True, False], "elo_pre_a": [1.0, 2.0]})
    cols = get_feature_cols(df)
    assert "injury_freak" not in cols and "elo_pre_a" in cols


def _toy_ledger():
    # fighter f1: KO win (normal), KO win (freak), then a 3rd row to read ctd features
    n = 3
    return pd.DataFrame({
        "fighter_id": ["f1"] * n, "opponent_id": ["o1", "o2", "o3"],
        "event_rank": [1, 2, 3],
        "event_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        "method": ["KO/TKO", "KO/TKO", "U-DEC"],
        "won": pd.array([1, 1, 1], dtype="Int8"),
        "end_round": pd.array([1, 1, 3], dtype="Int8"),
        "scheduled_rounds": pd.array([3, 3, 3], dtype="Int8"),
        "weight_class": ["Flyweight"] * n,
        "injury_freak": [False, True, False],
    })

def test_freak_ko_win_gets_no_ko_credit():
    from ufc.features.finishes import compute_finish_rates
    out = compute_finish_rates(_toy_ledger())
    row3 = out[out["event_rank"] == 3].iloc[0]
    # 2 prior wins, only 1 real KO -> 0.5 (would be 1.0 if the freak fight counted)
    assert row3["ko_win_rate_ctd"] == 0.5
    # freak fight still in the denominator: finish_rate over 2 prior fights = 0.5
    assert row3["finish_rate_ctd"] == 0.5
    # R1 numerators must not bypass the gate (INJ-1 review catch)
    assert row3["r1_ko_win_rate_ctd"] == 0.5


from ufc.features.ratings import compute_elo, compute_glicko2, compute_trueskill

def _two_isolated_fights(freak_second):
    # (f1 beats f2) normal, (f3 beats f4) freak — fresh fighters, identical setup,
    # plus per-fighter sentinel-style later rows to read the post-fight rating.
    rows = []
    for i, (w, l, fid, inj) in enumerate([("f1", "f2", "A", False),
                                          ("f3", "f4", "B", freak_second)]):
        for me, opp, won in [(w, l, 1), (l, w, 0)]:
            rows.append(dict(fight_id=fid, fighter_id=me, opponent_id=opp,
                             won=won, method="KO/TKO",
                             event_date=pd.Timestamp("2020-01-01"),
                             event_rank=1, injury_freak=inj))
        for me, opp in [(w, l), (l, w)]:
            rows.append(dict(fight_id=f"read_{fid}_{me}", fighter_id=me, opponent_id=opp,
                             won=np.nan, method=None,
                             event_date=pd.Timestamp("2021-01-01"),
                             event_rank=2, injury_freak=False))
    df = pd.DataFrame(rows)
    df["won"] = pd.array(df["won"], dtype="Int8")
    return df

def test_injury_dampens_all_three_rating_systems():
    led = _two_isolated_fights(freak_second=True)
    for fn, col in [(compute_elo, "elo_pre"),
                    (compute_glicko2, "glicko_mu_pre"),
                    (compute_trueskill, "ts_mu_pre")]:
        out = fn(led)
        def gain(f):  # post-fight rating minus initial, read at the later row
            post = out[(out["fighter_id"] == f) & (out["event_rank"] == 2)][col].iloc[0]
            pre = out[(out["fighter_id"] == f) & (out["event_rank"] == 1)][col].iloc[0]
            return post - pre
        assert abs(gain("f3") / gain("f1") - 0.25) < 0.02, f"{col}: {gain('f3')/gain('f1')}"
