"""Live forward-prediction log — the honest track record of the SERVED (prod) model.

A fight is only out-of-sample for the prod model in the window between "card
announced" and "model retrains on it". So we LOG the prod model's prediction for
each upcoming card BEFORE it happens (scripts/07_log_predictions.py), then GRADE
it once results land (scripts/08_grade_predictions.py). The pre-fight prediction
is locked in — re-logging never overwrites an existing entry.

This is the only honest accuracy number for the served model. Past fights can NOT
be added retroactively (the prod model already trained on them) — those stay on
the eval model's held-out backtest (see service._load_eval_winner).

Match key is order-independent (corner red/blue may differ between the upcoming
card and the recorded result): (event_date, sorted normalized name pair).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone

import pandas as pd

from ufc.io import paths

_COLUMNS = [
    "key", "event_date", "event_name", "card_id",
    "red", "blue", "p_red", "pred_winner", "model_sha", "logged_at",
    "status", "actual_winner", "correct", "resolved_at",
]

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def log_path():
    p = paths.root() / "data" / "predictions" / "live_log.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def norm_name(s: str) -> str:
    """Lowercase, strip accents/punctuation, drop common suffixes — for matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    toks = [t for t in s.split() if t and t not in _SUFFIXES]
    return " ".join(toks)


def matchup_key(event_date: str, a: str, b: str) -> str:
    d = str(event_date)[:10]
    pair = sorted([norm_name(a), norm_name(b)])
    return f"{d}|{pair[0]}|{pair[1]}"


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


def build_live_record(n_events: int = 10) -> dict:
    """Summarise the resolved live predictions for the History API."""
    df = load_log()
    resolved = df[df["status"] == "resolved"].copy()
    pending = int((df["status"] == "pending").sum())

    if len(resolved) == 0:
        return {
            "fights": 0, "correct": 0, "wrong": 0, "hitRate": 0.0,
            "pending": pending, "since": None, "modelSha": None, "events": [],
        }

    resolved["correct"] = resolved["correct"].astype(bool)
    n_correct = int(resolved["correct"].sum())
    n_fights = int(len(resolved))
    since = str(resolved["event_date"].min())[:10]
    model_sha = next((s for s in resolved["model_sha"] if s), None)

    resolved["_d"] = pd.to_datetime(resolved["event_date"])
    feed = []
    for ev_date in sorted(resolved["_d"].unique(), reverse=True)[:n_events]:
        grp = resolved[resolved["_d"] == ev_date]
        fights = [{
            "red": r["red"], "blue": r["blue"],
            "pRed": round(float(r["p_red"]), 3),
            "predWinner": r["pred_winner"],
            "actualWinner": r["actual_winner"],
            "correct": bool(r["correct"]),
        } for _, r in grp.iterrows()]
        c = int(grp["correct"].sum())
        feed.append({
            "id": str(ev_date)[:10],
            "event": grp["event_name"].iloc[0],
            "date": str(ev_date)[:10],
            "correct": c, "total": len(grp),
            "hitRate": round(c / len(grp), 3) if len(grp) else 0.0,
            "fights": fights,
        })

    return {
        "fights": n_fights, "correct": n_correct, "wrong": n_fights - n_correct,
        "hitRate": round(n_correct / n_fights, 3) if n_fights else 0.0,
        "pending": pending, "since": since, "modelSha": model_sha, "events": feed,
    }
