"""Shared weight-class label cleaning + poundage lookup.

Single source of truth for both feature-build (mileage.py) and inference
(matchup.py, serialize.py). Previously a correct cleaning function existed
only in serialize.py (API display), while mileage.py and matchup.py did raw
`.get()`/`.map()` lookups against the noisy scraped label directly — 1,328
ledger rows ("UFC Interim Heavyweight Title", "TUF ... Tournament Title",
"Catch Weight", "Open Weight") and 172 pre-fight-state fighters silently
failed the lbs lookup and produced NaN weight_class_change_lbs (audit
finding C-1, 2026-07-08).
"""
from __future__ import annotations

from typing import Any

# Longest-first so "Light Heavyweight" matches before "Heavyweight".
_WC_CANON = [
    "Light Heavyweight", "Heavyweight", "Middleweight", "Welterweight",
    "Lightweight", "Featherweight", "Bantamweight", "Flyweight",
    "Strawweight", "Catchweight",
]


def clean_weight_class(raw: Any) -> str:
    """Reduce noisy labels ('UFC Welterweight Title') to the bare class ('Welterweight').

    Already-clean scraped values pass through unchanged (idempotent). Falls back to the
    raw string if no canonical class is found (e.g. 'Open Weight' -> 'Open Weight',
    deliberately not a _WC_WEIGHT_LBS key — catch/open-weight bouts are handled by an
    explicit is_catch branch in matchup.py, not invented poundage)."""
    if not raw:
        return ""
    s = str(raw).strip()
    low = s.lower()
    womens = "women" in low or "wmma" in low
    for cls in _WC_CANON:
        if cls.lower() in low:
            return f"Women's {cls}" if (womens and not cls.startswith("Women")) else cls
    return s


_WC_WEIGHT_LBS: dict[str, float] = {
    "Strawweight": 115.0, "Women's Strawweight": 115.0,
    "Flyweight": 125.0, "Women's Flyweight": 125.0,
    "Bantamweight": 135.0, "Women's Bantamweight": 135.0,
    "Featherweight": 145.0, "Women's Featherweight": 145.0,
    "Lightweight": 155.0,
    "Welterweight": 170.0,
    "Middleweight": 185.0,
    "Light Heavyweight": 205.0,
    "Heavyweight": 265.0,
    "Super Heavyweight": 265.0,
}


def weight_class_lbs(raw: Any) -> float | None:
    """Clean then look up poundage. None for catch/open-weight/unrecognized labels."""
    return _WC_WEIGHT_LBS.get(clean_weight_class(raw))
