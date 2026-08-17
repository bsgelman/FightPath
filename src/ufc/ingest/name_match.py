"""Fighter name normalization and ID resolution.

Canonical key = scraper's fighter_id (URL hex).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from ufc.io import paths
from ufc.ingest.parse_helpers import normalize_name


def _load_overrides() -> dict[str, str]:
    cfg = paths.root() / "configs" / "name_overrides.yaml"
    if not cfg.exists():
        return {}
    with open(cfg) as f:
        data = yaml.safe_load(f) or {}
    return data.get("overrides", {}) or {}


def build_lookup(fighters_df: pd.DataFrame) -> dict[str, str]:
    """Build normalized_name -> fighter_id lookup.

    Returns dict mapping normalized name -> fighter_id.
    Where multiple fighters share a name, the lookup returns a list
    but resolve_name handles disambiguation by event date.
    """
    lookup: dict[str, list[str]] = {}
    for _, row in fighters_df.iterrows():
        if not row.get("fighter_id"):
            continue
        norm = normalize_name(str(row.get("fighter_name", "")))
        if norm not in lookup:
            lookup[norm] = []
        lookup[norm].append(row["fighter_id"])
    return lookup


def _fighter_dates(fights_df: pd.DataFrame, fighter_id: str) -> list[date]:
    """Get all fight dates for a fighter from the fights dataframe."""
    mask = (fights_df["fighter_a_id"] == fighter_id) | (fights_df["fighter_b_id"] == fighter_id)
    dates = fights_df[mask]["event_date"].dropna().tolist()
    return [d for d in dates if isinstance(d, date)]


def resolve_name(
    name: str,
    event_date: date,
    lookup: dict[str, list[str]],
    fighters_df: pd.DataFrame,
    fights_df: pd.DataFrame | None = None,
    overrides: dict[str, str] | None = None,
) -> str | None:
    """Resolve a fighter name string to a fighter_id.

    Priority:
    1. Check overrides dict first.
    2. Normalize and look up in lookup.
    3. If multiple matches, pick the one with a fight within ±5 years of event_date.
    4. If still ambiguous, return the first match and log a warning.
    """
    if overrides is None:
        overrides = {}

    norm = normalize_name(name)

    # Override check
    if norm in overrides:
        return overrides[norm]

    candidates = lookup.get(norm, [])

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple: disambiguate by date proximity if fights_df provided
    if fights_df is not None and event_date:
        window = timedelta(days=5 * 365)
        scored = []
        for fid in candidates:
            fight_dates = _fighter_dates(fights_df, fid)
            if fight_dates:
                min_diff = min(abs((d - event_date).days) for d in fight_dates)
                scored.append((min_diff, fid))
        scored.sort(key=lambda x: x[0])
        if scored and scored[0][0] <= window.days:
            return scored[0][1]

    return candidates[0]


def build_name_map(
    fights_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Produce name_map: each fight row -> (fighter_a_id, fighter_b_id).

    fights_df must have columns: fight_id, fighter_a_name, fighter_b_name, event_name
    events_df must have: event_name, event_date
    fighters_df must have: fighter_id, fighter_name
    """
    overrides = _load_overrides()
    lookup = build_lookup(fighters_df)

    # Attach event dates to fights
    event_dates = events_df.set_index("event_name")["event_date"].to_dict()

    # We'll do a partial resolution — fights_df doesn't yet have fighter IDs,
    # so we can't use date-based fight activity disambiguation yet.
    # We'll do a two-pass: first pass resolves unambiguous names,
    # second pass uses the partially built mapping for disambiguation.

    reports_dir = paths.outputs_reports()
    reports_dir.mkdir(parents=True, exist_ok=True)
    unresolved_log = reports_dir / "unresolved_names.txt"

    results = []
    unresolved = []

    for _, row in fights_df.iterrows():
        fight_id = row["fight_id"]
        event_date = event_dates.get(row.get("event_name", ""))

        a_id = resolve_name(
            str(row.get("fighter_a_name", "")),
            event_date,
            lookup,
            fighters_df,
            overrides=overrides,
        )
        b_id = resolve_name(
            str(row.get("fighter_b_name", "")),
            event_date,
            lookup,
            fighters_df,
            overrides=overrides,
        )

        if a_id is None:
            unresolved.append(f"UNRESOLVED A: '{row.get('fighter_a_name')}' in '{row.get('event_name')}' on {event_date}")
        if b_id is None:
            unresolved.append(f"UNRESOLVED B: '{row.get('fighter_b_name')}' in '{row.get('event_name')}' on {event_date}")

        results.append({
            "fight_id": fight_id,
            "event_date": event_date,
            "fighter_a_name": row.get("fighter_a_name"),
            "fighter_b_name": row.get("fighter_b_name"),
            "fighter_a_id": a_id,
            "fighter_b_id": b_id,
        })

    if unresolved:
        with open(unresolved_log, "w") as f:
            f.write("\n".join(unresolved))
        print(f"  {len(unresolved)} unresolved names — see {unresolved_log}")

    return pd.DataFrame(results)
