"""Grappling profile features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from ufc.io import paths
from ufc.features.windows import all_window_flavors, causal_expanding


def _cfg():
    with open(paths.root() / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)


def compute_grappling(ledger: pd.DataFrame) -> pd.DataFrame:
    df = ledger.copy()
    cfg = _cfg()
    hl = cfg["windows"]["decay_halflife_days"]
    w2y = cfg["windows"]["date_months"] * 30

    by = "fighter_id"
    sort_col = "event_rank"
    date_col = "event_date"

    df["fight_min"] = df["total_fight_sec"].fillna(0) / 60.0

    # ── Per-fight grappling rates ─────────────────────────────────────────
    df["_td_per_15_fight"] = np.where(
        df["fight_min"] > 0,
        df["td_landed"] / df["fight_min"] * 15,
        np.nan,
    )
    df["_td_acc_fight"] = np.where(
        df["td_attempted"] > 0,
        df["td_landed"] / df["td_attempted"],
        np.nan,
    )
    df["_td_def_fight"] = np.where(
        df["td_absorbed_attempted"] > 0,
        1.0 - df["td_absorbed_landed"] / df["td_absorbed_attempted"],
        np.nan,
    )
    df["_ctrl_pct_fight"] = np.where(
        df["total_fight_sec"].fillna(0) > 0,
        df["ctrl_sec"] / df["total_fight_sec"],
        np.nan,
    )
    df["_sub_att_per_15_fight"] = np.where(
        df["fight_min"] > 0,
        df["sub_att_for"] / df["fight_min"] * 15,
        np.nan,
    )
    df["_reversal_per_15_fight"] = np.where(
        df["fight_min"] > 0,
        df["rev_for"] / df["fight_min"] * 15,
        np.nan,
    )
    df["_ground_share_grp_fight"] = np.where(
        df["sig_str_landed"] > 0,
        df["ground_landed"] / df["sig_str_landed"],
        np.nan,
    )
    df["_td_attempted_per_15_fight"] = np.where(
        df["fight_min"] > 0,
        df["td_attempted"] / df["fight_min"] * 15,
        np.nan,
    )

    # Sub defense: 1 - P(submitted | opp sub att > 0)
    # We approximate as: 1 - (was submitted) per fight where opp had sub attempts
    df["_sub_def_binary"] = np.where(
        df["sub_att_against"] > 0,
        1.0 - (df["method"] == "SUB").astype(float),
        np.nan,
    )

    # Scramble proxy: reversals + successful TD defenses
    # We approximate TD defenses events as where opp attempted TD and didn't land
    df["_td_def_success_fight"] = np.where(
        df["td_absorbed_attempted"] > 0,
        (df["td_absorbed_attempted"] - df["td_absorbed_landed"]).clip(0) / df["fight_min"].replace(0, np.nan) * 15,
        np.nan,
    )

    raw_metrics = [
        ("_td_per_15_fight", "td_per_15"),
        ("_td_acc_fight", "td_acc"),
        ("_td_def_fight", "td_def"),
        ("_ctrl_pct_fight", "ctrl_pct"),
        ("_sub_att_per_15_fight", "sub_att_per_15"),
        ("_reversal_per_15_fight", "reversal_per_15"),
        ("_ground_share_grp_fight", "ground_share_grp"),
        ("_td_attempted_per_15_fight", "td_attempted_per_15"),
        ("_sub_def_binary", "sub_def"),
        ("_td_def_success_fight", "td_def_success_per_15"),
    ]

    new_cols = {}
    for raw_col, feat_name in raw_metrics:
        if raw_col not in df.columns:
            continue
        flavors = all_window_flavors(
            df, by, sort_col, date_col, raw_col,
            halflife_days=hl,
            window_24mo_days=w2y,
            prefix=feat_name,
        )
        new_cols.update(flavors)

    raw_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_fight")]
    raw_cols += ["_sub_def_binary", "_td_def_success_fight"]
    df = df.drop(columns=raw_cols, errors="ignore")
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df
