"""Fetch and parse live prop lines from Power Play and Flat Multi.

These are unofficial, undocumented endpoints — schema and league IDs drift.
Keep endpoints + MARKET_MAP as editable constants and parse defensively.
Cloudflare / geo-block can return 403/429/HTML; errors surface as messages,
never raised, so one platform down does not crash the whole feature.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Endpoint constants (edit here when APIs drift)
# ---------------------------------------------------------------------------

# Power Play JSON:API — league id discovered dynamically; fallback if lookup fails

# ── KILL SWITCH (2026-06, re-enabled 2026-07-01) ────────────────────────────
# The account was rate-limited / blocked by Power Play starting 2026-06-22.
# Single manual GET to /leagues from an alternate (non-home) network on
# 2026-07-01 returned a clean 200 (no Cloudflare challenge) — consistent with
# an IP-level block rather than an account/device ban, so default flipped back
# on. NOT re-verified from the regular home network/IP yet — the scheduled
# line-log's "local primary" leg runs from that network, so if the block was
# IP-based and that IP is still flagged, Power Play calls from THAT leg specifically
# could fail again (Flat Multi is unaffected either way). Disable again by setting
# env FIGHTPATH_ENABLE_POWERPLAY=0 if that happens.

# Flat Multi over/under board

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Market map: (substring in normalized stat) -> (canonical_market, unit_factor)
# unit_factor converts API line value to canonical units:
#   duration/rounds markets: seconds  |  count markets: raw count (factor=1)
# Order matters — more specific patterns first.
# ---------------------------------------------------------------------------

MARKET_MAP: list[tuple[str, tuple[str, float]]] = [
    # Round-1 sig strikes (must come BEFORE generic sig strikes)
    # Matched only when stat contains both a "round 1"/"1st round" token AND a sig-strikes token.
    # Handled specially in _map_market (not via substring alone).
    # Body/leg sig strikes — must come BEFORE generic "significant strikes"
    ("body strikes",                ("body_sig_strikes", 1.0)),
    ("leg strikes",                 ("leg_sig_strikes",  1.0)),
    ("takedown",                    ("takedowns", 1.0)),
    ("significant strikes",         ("sig_strikes", 1.0)),
    ("sig strikes",                 ("sig_strikes", 1.0)),
    ("total strikes",               ("sig_strikes", 1.0)),
    ("submission attempt",          ("sub_attempts", 1.0)),
    ("knockdown",                   ("knockdowns", 1.0)),
    # ctrl_time MUST come before "fight time"/"minutes" so ctrl doesn't map to duration
    ("control time",                ("ctrl_time", 60.0)),   # minutes -> seconds
    ("total rounds",                ("rounds", 300.0)),     # rounds * 300 -> seconds
    ("fight time",                  ("duration", 60.0)),    # minutes -> seconds
    ("fight duration",              ("duration", 60.0)),
    # Generic "rounds" / "minutes" — put after "total rounds" / "fight time"
    ("rounds",                      ("rounds", 300.0)),
    ("minutes",                     ("duration", 60.0)),
]

# Stat labels to explicitly skip (no model for these)
_SKIP_PATTERNS = [
    "fantasy score", "fantasy points",
    "fight result", "method", "moneyline", "winner",
    "win by", "distance", "will fight go", "significant strike accuracy",
]

# Per-market line plausibility bounds (canonical units: seconds for ctrl/duration/rounds,
# raw count otherwise). A line outside these bounds means the API stat was mis-mapped to
# the wrong market (e.g. a 29.5 "significant strikes" line landing on "takedowns") — drop
# it rather than score it against the wrong CDF and report a bogus ~100% probability.
_LINE_BOUNDS: dict[str, tuple[float, float]] = {
    "sig_strikes":       (1.0, 250.0),
    "r1_sig_strikes":    (1.0, 120.0),
    "body_sig_strikes":  (0.5, 120.0),
    "leg_sig_strikes":   (0.5, 120.0),
    "sig_strikes_combo": (1.0, 400.0),
    "takedowns":         (0.5, 20.0),
    "r1_takedowns":      (0.5, 12.0),
    "sub_attempts":      (0.5, 12.0),
    "knockdowns":        (0.5, 8.0),
    "ctrl_time":         (5.0, 1800.0),
    "duration":          (30.0, 2100.0),
    "rounds":            (150.0, 1500.0),
}


def _line_plausible(market: str, line_value: float) -> bool:
    """False when line_value is outside the sane range for market (mis-mapping guard)."""
    bounds = _LINE_BOUNDS.get(market)
    if bounds is None:
        return True
    return bounds[0] <= line_value <= bounds[1]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LiveProp:
    platform: str           # 'powerplay' | 'flatmulti'
    player_name: str        # as returned by the API
    market: str             # canonical: sig_strikes | r1_sig_strikes | takedowns | duration | rounds
    line_value: float       # canonical units: seconds for duration/rounds, raw count otherwise
    raw_stat: str           # original API stat label (for debugging / display)
    odds_type: str = "standard"           # standard | demon | goblin (PP) | boost (UD)
    board_multiplier: Optional[float] = None  # max payout multiplier across sides
    directional: bool = False             # over-only (PP demon/goblin; UD higher-only lines)
    under_only: bool = False              # under-only (UD lower-only adjusted lines)
    over_multiplier: Optional[float] = None   # live payout mult for higher/over side
    under_multiplier: Optional[float] = None  # live payout mult for lower/under side


@dataclass
class ResolvedProp:
    platform: str
    player_name: str        # API name (pre-resolution)
    card_red: str           # canonical card fighter name
    card_blue: str
    fight_idx: int          # index into card_matchups list
    corner: str             # 'red' | 'blue' | 'fight'
    market: str
    line_value: float       # canonical units
    raw_stat: str
    odds_type: str = "standard"
    board_multiplier: Optional[float] = None
    directional: bool = False
    under_only: bool = False
    over_multiplier: Optional[float] = None
    under_multiplier: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



# Name matching lives in name_norm.py so it survives independently of this
# scraping module. Aliased to the old private names to keep call sites here
# unchanged.
from ufc.ingest.name_norm import (  # noqa: E402
    normalize as _normalize,
    token_surname_match as _token_surname_match,
    _first_compatible,
)


_FINISH_MARKET_NAMES = frozenset({
    "ko_finish", "sub_finish", "finish",
    "r1_finish", "r2_finish", "r3_finish", "r4_finish", "r5_finish",
    "r1_ko", "r2_ko", "r3_ko", "r4_ko", "r5_ko",
})

# Tokens used to detect finish props. Normalization removes punctuation including '/',
# so "KO/TKO" → "kotko". Both are in the list for robustness.
_KO_TOKENS  = ("kotko", "knockout", "knock out", " ko ", " tko", "by ko", "by tko",
                "wins by ko", "win by ko", "wins by tko", "win by tko")
_SUB_TOKENS = ("wins by submission", "win by submission", "by sub", "tapout",
                "tap out", "submission finish")
_ANY_FINISH_TOKENS = ("finish", "inside the distance", "not go")
# Round indicator tokens for k=1..5
_ROUND_TOKENS: dict[int, tuple[str, ...]] = {
    1: ("round 1", "1st round", "first round", "rd 1"),
    2: ("round 2", "2nd round", "second round", "rd 2"),
    3: ("round 3", "3rd round", "third round", "rd 3"),
    4: ("round 4", "4th round", "fourth round", "rd 4"),
    5: ("round 5", "5th round", "fifth round", "rd 5"),
}


def _map_market(raw_stat: str) -> tuple[str, float] | None:
    """Map raw API stat label to (canonical_market, unit_factor). Returns None to skip."""
    norm = _normalize(raw_stat)

    # ── Finish-market detection — BEFORE _SKIP_PATTERNS because labels like
    # "Wins by Submission" and "Inside the Distance" contain skip-pattern tokens.
    has_ko  = any(t in norm for t in _KO_TOKENS)
    has_sub = any(t in norm for t in _SUB_TOKENS)
    has_ss  = any(t in norm for t in ("significant strike", "sig strike"))
    has_any_finish = any(t in norm for t in _ANY_FINISH_TOKENS) or has_ko or has_sub

    if has_any_finish and not has_ss:
        # Check for round-specific finish first. KO-specific round markets
        # ("Round N Knockout") are DISTINCT from any-finish round markets
        # ("Round N Finish", which includes submissions) — verified against
        # live snapshots both text variants are quoted for the same fight.
        # Pricing/grading a KO-only market as any-finish overprices it and
        # misgrades a round-N submission as a hit.
        for k, rtokens in _ROUND_TOKENS.items():
            if any(t in norm for t in rtokens):
                if has_ko:
                    return (f"r{k}_ko", 1.0)
                return (f"r{k}_finish", 1.0)
        # No round token → fight-level
        if has_ko:
            return ("ko_finish", 1.0)
        if has_sub:
            return ("sub_finish", 1.0)
        return ("finish", 1.0)

    # Skip explicitly (after finish detection)
    for skip in _SKIP_PATTERNS:
        if skip in norm:
            return None

    r1_tokens = ("round 1", "1st round", "first round", "rd 1", "r1")
    has_r1 = any(t in norm for t in r1_tokens)

    # Sig-strikes combo: "combo" token + sig-strikes indicator
    has_combo = "combo" in norm or (" + " in raw_stat and has_ss)
    if has_combo and has_ss:
        return ("sig_strikes_combo", 1.0)

    # R1 sig strikes: R1 indicator AND sig-strikes indicator
    if has_r1 and has_ss:
        return ("r1_sig_strikes", 1.0)

    # R1 takedowns: R1 indicator AND takedown token (before generic takedown in MARKET_MAP)
    has_td = "takedown" in norm
    if has_r1 and has_td:
        return ("r1_takedowns", 1.0)

    # Body/leg sig strikes: requires has_ss modifier
    if has_ss and "body" in norm:
        return ("body_sig_strikes", 1.0)
    if has_ss and ("leg" in norm or "low kick" in norm):
        return ("leg_sig_strikes", 1.0)

    for pattern, result in MARKET_MAP:
        if pattern in norm:
            return result

    return None






# ---------------------------------------------------------------------------
# Power Play
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Flat Multi
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Combined fetch
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Card resolution
# ---------------------------------------------------------------------------

def resolve_to_card(
    props: list[LiveProp],
    card_matchups: list,
    fighters_df=None,
) -> tuple[list[ResolvedProp], list[str]]:
    """Match API fighter names to fighters on the loaded card (card-restricted).

    card_matchups: list of 8-tuples (red, blue, rounds, is_title, event_date,
                   weight_class, referee, location) — the in-scope variable from the UI.

    Returns (resolved_props, unresolved_names).
    Fighter names not found on any matchup are logged to outputs/reports/unresolved_names.txt
    and returned in the unresolved list for UI display.
    """
    from ufc.io import paths

    # Build lookup: normalized_name -> (fight_idx, corner, card_red, card_blue)
    name_lookup: dict[str, tuple[int, str, str, str]] = {}
    for idx, m in enumerate(card_matchups):
        red, blue = m[0], m[1]
        name_lookup[_normalize(red)] = (idx, "red", red, blue)
        name_lookup[_normalize(blue)] = (idx, "blue", red, blue)
    all_norms = list(name_lookup.keys())

    # Load name overrides (normalized_api_name -> canonical fighter name on card)
    overrides: dict[str, str] = {}
    try:
        import yaml
        from ufc.io import paths as _paths
        override_path = _paths.root() / "configs" / "name_overrides.yaml"
        with open(override_path) as f:
            raw = yaml.safe_load(f) or {}
        overrides = {k: v for k, v in (raw.get("overrides") or {}).items()}
    except Exception:
        pass
    # Build api_norm -> card_norm mapping from overrides
    api_to_card: dict[str, str] = {}
    for api_norm, fighter_id_or_name in overrides.items():
        # The override value is a fighter_id hex — we need the canonical card name.
        # Try to match fighter_id against card fighters by looking up fighters_df.
        # If fighters_df not available, skip override (resolved by fuzzy instead).
        if fighters_df is not None:
            try:
                row = fighters_df[fighters_df["fighter_id"] == fighter_id_or_name]
                if not row.empty:
                    card_norm = _normalize(row.iloc[0]["fighter_name"])
                    if card_norm in name_lookup:
                        api_to_card[api_norm] = card_norm
            except Exception:
                pass

    resolved: list[ResolvedProp] = []
    unresolved: list[str] = []
    seen_unresolved: set[str] = set()

    for prop in props:
        norm = _normalize(prop.player_name)

        # Check manual overrides first
        if norm in api_to_card:
            card_norm = api_to_card[norm]
            hit = name_lookup[card_norm]
        elif norm in name_lookup:
            hit = name_lookup[norm]
        else:
            # Token-aware surname match first (handles first-name abbreviations like
            # "Sharabutdin Magomedov"↔"Shara Magomedov" that fall below the 0.85 fuzzy
            # cutoff). Only accept when it resolves to exactly ONE card fighter, so two
            # brothers sharing a surname stay disambiguated by their first names.
            tok = [cn for cn in all_norms if _token_surname_match(norm, cn)]
            if len(tok) == 1:
                hit = name_lookup[tok[0]]
            else:
                # Fuzzy match fallback
                matches = difflib.get_close_matches(norm, all_norms, n=1, cutoff=0.85)
                if matches:
                    hit = name_lookup[matches[0]]
                else:
                    key = f"{prop.platform}:{prop.player_name}"
                    if key not in seen_unresolved:
                        seen_unresolved.add(key)
                        unresolved.append(prop.player_name)
                    continue

        fight_idx, corner, card_red, card_blue = hit

        # Fight-level markets: corner is 'fight' regardless of which fighter resolved
        if prop.market in ("duration", "duration_sec", "rounds", "sig_strikes_combo"):
            corner = "fight"

        resolved.append(ResolvedProp(
            platform=prop.platform,
            player_name=prop.player_name,
            card_red=card_red,
            card_blue=card_blue,
            fight_idx=fight_idx,
            corner=corner,
            market=prop.market,
            line_value=prop.line_value,
            raw_stat=prop.raw_stat,
            odds_type=prop.odds_type,
            board_multiplier=prop.board_multiplier,
            directional=prop.directional,
            under_only=prop.under_only,
            over_multiplier=prop.over_multiplier,
            under_multiplier=prop.under_multiplier,
        ))

    # Append new unresolved names to the report file (deduped)
    if unresolved:
        try:
            report_path = paths.outputs_reports() / "unresolved_names.txt"
            existing: set[str] = set()
            if report_path.exists():
                existing = set(report_path.read_text(encoding="utf-8").splitlines())
            new_entries = [n for n in unresolved if n not in existing]
            if new_entries:
                with open(report_path, "a", encoding="utf-8") as f:
                    for name in new_entries:
                        f.write(f"{name}\n")
        except Exception:
            pass

    return resolved, unresolved
