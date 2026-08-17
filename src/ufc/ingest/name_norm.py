"""Fighter-name normalization and token-aware surname matching.

Extracted verbatim from `prop_lines.py` so the shared matchers survive
independently of the sportsbook-scraping module that happened to host them.
Consumers: `ingest.market_lines` (Kalshi contract -> card fighter) and
`inference.matchup`.

The normalization here is deliberately SPACE-PRESERVING and differs from other
normalizers in this codebase — see the note in `inference/matchup.py`. Do not
"unify" it with them.
"""
from __future__ import annotations

import re
import unicodedata


def normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    # Non-alnum -> space (not deleted) so compound surnames split on punctuation
    # ("Saint-Denis") tokenize the same as a source that uses a plain space
    # ("Saint Denis") instead of silently fusing into one unmatchable token.
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _first_compatible(a_tokens: list[str], b_tokens: list[str]) -> bool:
    """First names compatible: equal, shared single initial, or one a prefix of the
    other (handles abbreviations like 'Sharabutdin'↔'Shara'). Single-token names pass."""
    if len(a_tokens) < 2 or len(b_tokens) < 2:
        return True
    a, b = a_tokens[0], b_tokens[0]
    if a == b:
        return True
    if (len(a) == 1 or len(b) == 1) and a[0] == b[0]:
        return True
    return a.startswith(b) or b.startswith(a)


def token_surname_match(api_norm: str, card_norm: str) -> bool:
    """Token-aware surname match. Inputs are `normalize`d (space-separated,
    diacritics/punctuation stripped). Surnames must match as WHOLE tokens — never a
    raw substring — so 'Murtazali Magomedov' does not phantom-match 'Andre Lima'
    (the documented 'lima' substring bug)."""
    a, b = api_norm.split(), card_norm.split()
    if not a or not b:
        return False
    la, lb = a[-1], b[-1]
    if len(la) < 3 or len(lb) < 3:
        return False
    if la == lb:
        return _first_compatible(a, b)
    return lb in a or la in b
