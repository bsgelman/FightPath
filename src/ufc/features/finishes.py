"""Finish-rate features — direct success/failure rates per method.

The method classifier and duration P(decision) classifier currently rely on
indirect proxies (kd_per_15, sub_att_per_15) that conflate ATTEMPTS with
FINISHES. These features add explicit rolling rates per fighter, both offensive
(did THIS fighter finish opponents?) and defensive (did THIS fighter get
finished?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from ufc.io import paths
from ufc.features.windows import all_window_flavors


def _cfg():
    with open(paths.root() / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)


def compute_finish_rates(ledger: pd.DataFrame) -> pd.DataFrame:
    """Add finish-rate features to the per-fighter ledger.

    Inputs (ledger must have): fighter_id, event_rank, event_date, method, won.
    Output: ledger + new rolling-rate columns in the standard 5 flavors
    (_ctd, _l3, _l5, _2y, _decay) for each base metric below.
    """
    df = ledger.copy()
    cfg = _cfg()
    hl = cfg["windows"]["decay_halflife_days"]
    w2y = cfg["windows"]["date_months"] * 30

    by, sort_col, date_col = "fighter_id", "event_rank", "event_date"

    m = df["method"].astype("string")
    won = df["won"].fillna(-1).astype(int)  # -1 = unknown outcome
    inj = df.get("injury_freak", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    # Freak-injury stoppages are evidence of nothing except that the fight
    # occurred: no KO/SUB credit (winner) or debit (loser). Every numerator
    # below flows from these two binaries, so this is the single choke point.
    # (R1 numerators included — they must use is_ko/is_sub, never raw m ==; see INJ-1 review)
    is_ko = ((m == "KO/TKO") & ~inj).astype(float)
    is_sub = ((m == "SUB") & ~inj).astype(float)
    is_dec = m.isin(["U-DEC", "S-DEC", "M-DEC"]).astype(float)
    is_finish = (is_ko + is_sub).clip(0, 1)

    # ── Per-fight binaries (current row; will be shifted by all_window_flavors) ─
    df["_ko_win_fight"]   = np.where(won == 1, is_ko, np.nan)
    df["_sub_win_fight"]  = np.where(won == 1, is_sub, np.nan)
    df["_ko_loss_fight"]  = np.where(won == 0, is_ko, np.nan)
    df["_sub_loss_fight"] = np.where(won == 0, is_sub, np.nan)
    # Overall fight outcome rates (don't condition on won — captures pace)
    df["_finish_fight"]   = is_finish.where(m.notna(), np.nan)
    df["_dec_fight"]      = is_dec.where(m.notna(), np.nan)
    # Tactical: did fight end before round 3?
    end_round = df.get("end_round", pd.Series(np.nan, index=df.index))
    df["_early_finish_fight"] = np.where(
        m.notna(),
        ((is_finish == 1) & (end_round.fillna(99) <= 2)).astype(float),
        np.nan,
    )

    # R1-only finish rates — captures first-round predators (e.g. Topuria)
    df["_r1_ko_win_fight"] = np.where(
        (won == 1) & (is_ko == 1) & (end_round.fillna(99) == 1),
        1.0, np.where(m.notna(), 0.0, np.nan),
    )
    df["_r1_sub_win_fight"] = np.where(
        (won == 1) & (is_sub == 1) & (end_round.fillna(99) == 1),
        1.0, np.where(m.notna(), 0.0, np.nan),
    )

    # Prior 5-round decision rate — per-fighter "I go the distance in title fights" signal.
    # NaN for fights where scheduled_rounds != 5; expanding mean over prior 5-round fights only.
    sched_rounds = df.get("scheduled_rounds", pd.Series(3, index=df.index)).fillna(3)
    df["_dec_5rd_only_fight"] = np.where(
        (sched_rounds >= 5) & m.notna(),
        is_dec, np.nan,
    )

    raw_metrics = [
        ("_ko_win_fight",       "ko_win_rate"),
        ("_sub_win_fight",      "sub_win_rate"),
        ("_ko_loss_fight",      "ko_loss_rate"),     # chin durability inverse
        ("_sub_loss_fight",     "sub_loss_rate"),    # sub defense inverse
        ("_finish_fight",       "finish_rate"),       # fights involving a finisher
        ("_dec_fight",          "dec_rate"),
        ("_early_finish_fight", "early_finish_rate"),
        ("_r1_ko_win_fight",    "r1_ko_win_rate"),
        ("_r1_sub_win_fight",   "r1_sub_win_rate"),
        ("_dec_5rd_only_fight", "prior_5rd_dec_rate"),
    ]

    new_cols = {}
    for raw_col, feat_name in raw_metrics:
        flavors = all_window_flavors(
            df, by, sort_col, date_col, raw_col,
            halflife_days=hl,
            window_24mo_days=w2y,
            prefix=feat_name,
        )
        new_cols.update(flavors)

    # Drop raw intermediates
    raw_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_fight")]
    df = df.drop(columns=raw_cols, errors="ignore")
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def fill_sparse_history(df: pd.DataFrame,
                         train_mask: pd.Series | None = None,
                         weight_class_col: str = "weight_class") -> pd.DataFrame:
    """Backfill _l5/_l3/_2y finish-rate features with TRAIN-fold weight-class median.

    Rookies with <3 prior UFC fights have unreliable rolling rates. Replace NaN
    with the weight-class median computed on the training fold only (no lookahead).
    """
    out = df.copy()
    if weight_class_col not in out.columns:
        return out
    if train_mask is None:
        train_mask = pd.Series(True, index=out.index)

    sparse_suffixes = ("_l5", "_l3", "_2y")
    bases = [
        "ko_win_rate", "sub_win_rate", "ko_loss_rate", "sub_loss_rate",
        "finish_rate", "dec_rate", "early_finish_rate",
        "r1_ko_win_rate", "r1_sub_win_rate", "prior_5rd_dec_rate",
    ]
    for base in bases:
        for suf in sparse_suffixes:
            col = f"{base}{suf}"
            if col not in out.columns:
                continue
            tr_med = out.loc[train_mask].groupby(weight_class_col)[col].median()
            global_med = out.loc[train_mask, col].median()
            mapped = out[weight_class_col].map(tr_med).fillna(global_med)
            out[col] = out[col].fillna(mapped)
    return out
