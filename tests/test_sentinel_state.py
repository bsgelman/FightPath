"""append_sentinel_rows: structure, dating, nulling, and training-table isolation."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import pytest

from ufc.features.assemble import append_sentinel_rows, SENTINEL_KEEP_COLS


def _ledger():
    df = pd.DataFrame([
        dict(fight_id="F1", event_date=pd.Timestamp("2025-01-01"), event_rank=0,
             fighter_id="f1", opponent_id="f2", won=1.0, method="KO/TKO",
             weight_class="Bantamweight", stance="ORTHO", age_years=28.0,
             reach_in=68.0, height_in=68.0, weight_lbs=135.0,
             sig_str_landed=50.0, end_round=2, scheduled_rounds=3,
             referee="Herb Dean", location="Las Vegas", is_title=False,
             n_common_opps=2),
        dict(fight_id="F1", event_date=pd.Timestamp("2025-01-01"), event_rank=0,
             fighter_id="f2", opponent_id="f1", won=0.0, method="KO/TKO",
             weight_class="Bantamweight", stance="SOUTH", age_years=30.0,
             reach_in=70.0, height_in=69.0, weight_lbs=135.0,
             sig_str_landed=30.0, end_round=2, scheduled_rounds=3,
             referee="Herb Dean", location="Las Vegas", is_title=False,
             n_common_opps=3),
        dict(fight_id="F2", event_date=pd.Timestamp("2025-06-01"), event_rank=1,
             fighter_id="f1", opponent_id="f3", won=0.0, method="U-DEC",
             weight_class="Featherweight", stance="ORTHO", age_years=28.4,
             reach_in=68.0, height_in=68.0, weight_lbs=145.0,
             sig_str_landed=40.0, end_round=3, scheduled_rounds=3,
             referee="Marc Goddard", location="London", is_title=True,
             n_common_opps=1),
        dict(fight_id="F2", event_date=pd.Timestamp("2025-06-01"), event_rank=1,
             fighter_id="f3", opponent_id="f1", won=1.0, method="U-DEC",
             weight_class="Featherweight", stance="ORTHO", age_years=25.0,
             reach_in=72.0, height_in=70.0, weight_lbs=145.0,
             sig_str_landed=60.0, end_round=3, scheduled_rounds=3,
             referee="Marc Goddard", location="London", is_title=True,
             n_common_opps=0),
    ])
    # mirror real ledger dtypes (nullable Int, plain bool) — see data/processed/ledger
    df = df.astype({"end_round": "Int32", "scheduled_rounds": "Int8", "is_title": bool})
    return df


def test_one_sentinel_per_fighter():
    out = append_sentinel_rows(_ledger())
    sent = out[out["is_sentinel"]]
    assert len(sent) == 3                              # f1, f2, f3
    assert sent["fighter_id"].is_unique
    assert (~out["is_sentinel"]).sum() == 4            # real rows untouched


def test_sentinel_dated_after_last_fight_and_ranked_last():
    out = append_sentinel_rows(_ledger())
    s_f1 = out[(out["is_sentinel"]) & (out["fighter_id"] == "f1")].iloc[0]
    s_f2 = out[(out["is_sentinel"]) & (out["fighter_id"] == "f2")].iloc[0]
    assert s_f1["event_date"] == pd.Timestamp("2025-06-02")   # f1's last fight + 1d
    assert s_f2["event_date"] == pd.Timestamp("2025-01-02")   # f2's last fight + 1d
    assert (out[out["is_sentinel"]]["event_rank"] > 1).all()  # sorts after all real fights


def test_sentinel_keeps_identity_nulls_fight_facts():
    out = append_sentinel_rows(_ledger())
    s = out[(out["is_sentinel"]) & (out["fighter_id"] == "f1")].iloc[0]
    assert s["weight_class"] == "Featherweight"        # carried from LAST fight
    assert s["stance"] == "ORTHO"
    assert s["fight_id"] == "sentinel_f1"
    for col in ("won", "sig_str_landed", "end_round", "scheduled_rounds"):
        assert pd.isna(s[col]), col
    assert s["opponent_id"] is None or pd.isna(s["opponent_id"])
    assert s["method"] is None or pd.isna(s["method"])
    assert s["referee"] is None or pd.isna(s["referee"])
    # bool column has no null slot — sentinel gets the False placeholder, not
    # a carried value (f1's last real fight had is_title=True).
    assert s["is_title"] == False and isinstance(s["is_title"], (bool, np.bool_))
    assert out["is_title"].dtype == bool
    # plain-int placeholder: n_common_opps is not nullable, so the sentinel
    # gets an explicit 0 instead of a null (which would promote real rows to
    # float64 via pd.concat).
    assert s["n_common_opps"] == 0
    assert pd.api.types.is_integer_dtype(out["n_common_opps"].dtype)


def test_real_rows_byte_identical():
    led = _ledger()
    out = append_sentinel_rows(led)
    real = out[~out["is_sentinel"]].drop(columns=["is_sentinel"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(real, led.reset_index(drop=True))


def test_plain_int_column_raises():
    led = _ledger()
    led["kd_count"] = 1  # plain int64 — no null slot, would silently promote real rows to float64
    with pytest.raises(TypeError, match="kd_count"):
        append_sentinel_rows(led)
