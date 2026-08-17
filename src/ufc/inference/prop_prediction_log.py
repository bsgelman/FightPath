"""Forward real-line prop ledger — the honest per-prop betting track record.

Log pre-fight model probs at REAL operator lines (Power Play/Flat Multi) before each
card; grade realized stat after results land. Mirrors prediction_log.py pattern.

A prop row is locked on first write — re-logging never overwrites an existing entry.
This is the only honest CLV dataset; past fights cannot be added retroactively
(the prod model already trained on them).

Scripts:
  07b_log_prop_lines.py  — log pre-fight at real fetched lines
  08b_grade_props.py     — grade realized stat + hit/loss after results
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ufc.io import paths
from ufc.inference.prediction_log import norm_name, matchup_key  # reuse existing helpers

_COLUMNS = [
    "key",           # matchup_key(event_date, red, blue) + f"|{market}|{side}|{line:.4f}"
    "event_date",
    "red",           # red-corner fighter name on card
    "blue",          # blue-corner fighter name on card
    "fighter",       # fighter whose stat is being measured (or "" for fight-level)
    "corner",        # "red" | "blue" | "fight"
    "market",        # canonical market key e.g. "sig_strikes"
    "side",          # "over" | "under"
    "line_value",    # raw line value in canonical units (seconds for duration/ctrl_time)
    "model_prob",    # P(side wins) from production model
    "breakeven",     # implied per-leg breakeven at payout_type
    "edge_pct",      # model_prob - breakeven
    "platform",      # "powerplay" | "flatmulti"
    "odds_type",     # "standard" | "goblin" | "demon"
    "board_multiplier",  # per-line multiplier (None for standard PP)
    "payout_type",   # e.g. "powerplay_power_2pick"
    "model_sha",     # prod winner model sha
    "logged_at",     # ISO timestamp of pre-fight log
    "status",        # "pending" | "resolved"
    "realized_stat", # actual stat value after fight (None until resolved)
    "hit",           # True/False — did the side win (None until resolved)
    "resolved_at",   # ISO timestamp of grading
    "close_line_value",   # closing board line in canonical units (None until captured)
    "close_captured_at",  # ISO timestamp of the capture run (None until captured)
    "yes_bid",       # Kalshi yes-bid at log time (None for non-exchange rows) —
                     # lets 08c grade the maker lane fill-aware (resting price = bid+1c)
]


def log_path() -> "Path":
    from ufc.io import paths
    p = paths.root() / "data" / "predictions" / "prop_log.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def prop_key(event_date: str, red: str, blue: str,
             market: str, side: str, line_value: float,
             corner: str = "", platform: str = "") -> str:
    """Unique key for one prop row. Order-independent on red/blue.

    corner/platform: added 2026-07-09 (audit remediation A6). Finish props are
    always logged at line_value=0.5 for BOTH fighters, so without `corner` in
    the key one fighter's row silently overwrote the other on first-write-wins
    dedup (verified: 81/81 finish groups in the ledger collapsed to a single
    corner). platform distinguishes the same fighter/market/side/line quoted
    by both Power Play and Flat Multi. Default "" keeps the key shape unchanged
    for callers that don't pass them (e.g. existing tests) — real logging
    (07b_log_prop_lines.py) always passes both. Historical rows logged before
    this change keep their old (collision-prone, pre-fight, unrecoverable)
    keys; this only fixes newly-logged rows going forward."""
    base = matchup_key(event_date, red, blue)
    tail = ""
    if corner:
        tail += f"|{corner}"
    if platform:
        tail += f"|{platform}"
    return f"{base}|{market}|{side}|{line_value:.4f}{tail}"


def kalshi_key(event_date: str, red: str, blue: str, market: str,
               corner: str, odds_type: str, ask: float) -> str:
    """Unique key for one Kalshi ledger row. Extends prop_key's shape with
    corner (so two fighters' win_in_r{N} quotes at the same ask don't collide —
    a real risk since both sides of a coin-flip round market often price near
    the same few cents) and odds_type (so the taker and maker rows for the same
    quote don't collide). prop_key() itself is shared with the DFS lane and
    must not change."""
    base = matchup_key(event_date, red, blue)
    return f"{base}|{market}|{corner}|over|{odds_type}|{ask:.4f}"


def legacy_kalshi_key(event_date: str, red: str, blue: str, market: str,
                       odds_type: str, ask: float) -> str:
    """The pre-corner Kalshi key format (no `corner` segment). Used only to
    dedupe against rows already logged under the old format so they aren't
    re-logged under the new one."""
    base = matchup_key(event_date, red, blue)
    return f"{base}|{market}|over|{odds_type}|{ask:.4f}"


def load_log() -> pd.DataFrame:
    p = log_path()
    if p.exists():
        df = pd.read_parquet(p)
        for c in _COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[_COLUMNS]
    return pd.DataFrame(columns=_COLUMNS)


def save_log(df: pd.DataFrame) -> None:
    df[_COLUMNS].to_parquet(log_path(), index=False)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_prop_record(n_events: int = 10) -> dict:
    """Summarise resolved prop predictions for the History API."""
    df = load_log()
    resolved = df[df["status"] == "resolved"].copy()
    pending  = int((df["status"] == "pending").sum())

    if len(resolved) == 0:
        return {
            "bets": 0, "hits": 0, "misses": 0, "hitRate": 0.0,
            "pending": pending, "since": None,
        }

    resolved["hit"] = resolved["hit"].astype(bool)
    n_hits  = int(resolved["hit"].sum())
    n_total = int(len(resolved))
    since   = str(resolved["event_date"].min())[:10]

    per_market = (
        resolved.groupby("market")["hit"]
        .agg(hits="sum", n="count")
        .assign(hit_rate=lambda x: x["hits"] / x["n"])
        .reset_index()
        .to_dict("records")
    )

    return {
        "bets": n_total, "hits": n_hits, "misses": n_total - n_hits,
        "hitRate": round(n_hits / n_total, 3) if n_total else 0.0,
        "pending": pending, "since": since,
        "perMarket": per_market,
    }
