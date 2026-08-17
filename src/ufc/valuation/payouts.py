"""Payout table → implied probability per leg."""
from __future__ import annotations

import yaml
from ufc.io import paths


def _cfg() -> dict:
    with open(paths.root() / "configs" / "valuation.yaml") as f:
        return yaml.safe_load(f)


def get_payout_multiplier(payout_type: str) -> float:
    """Return the per-pick multiplier for a given payout type string.

    payout_type format: '{platform}_{variant}_{npick}' e.g. 'powerplay_power_2pick'
    Raises ValueError on unknown payout (instead of silently defaulting to 2.0×).
    """
    cfg = _cfg()
    parts = payout_type.split("_")
    if not parts:
        raise ValueError("Empty payout_type")
    platform = parts[0]
    sub_key = "_".join(parts[1:])

    platform_cfg = cfg.get(platform)
    if platform_cfg is None:
        raise ValueError(
            f"Unknown payout platform '{platform}'. "
            f"Available: {list(cfg.keys())}"
        )

    mult = platform_cfg.get(sub_key)
    if mult is None:
        raise ValueError(
            f"Unknown payout '{payout_type}'. "
            f"Platform '{platform}' has: {list(platform_cfg.keys())}"
        )
    if isinstance(mult, dict):
        # Flex picks have nested {all_correct: M, one_miss: M, ...} — return all_correct branch
        if "all_correct" in mult:
            return float(mult["all_correct"])
        return float(max(mult.values()))
    return float(mult)


def implied_prob_per_leg(payout_type: str, n_legs: int | None = None,
                         multiplier: float | None = None) -> float:
    """Implied breakeven probability per leg.

    For an N-pick power play at multiplier M: implied = M^(-1/N).
    If `multiplier` is given it overrides the config-derived M (e.g. goblin/demon
    or Flat Multi per-pick modifier adjusted total); N is still inferred from payout_type.
    """
    cfg = _cfg()
    parts = payout_type.split("_")
    platform = parts[0]
    sub_key = "_".join(parts[1:])

    # Infer N from payout_type if not provided
    if n_legs is None:
        import re
        m = re.search(r"(\d+)pick", payout_type)
        n_legs = int(m.group(1)) if m else 2

    mult = get_payout_multiplier(payout_type) if multiplier is None else float(multiplier)
    return mult ** (-1.0 / n_legs)


def get_odds_type_multiplier(platform: str, odds_type: str) -> float | None:
    """Return config fallback total multiplier for a non-standard odds_type.

    Returns None if no explicit config entry exists for this platform+odds_type.
    Used as a fallback when the live API does not supply a per-line multiplier.
    """
    cfg = _cfg()
    platform_cfg = cfg.get(platform, {})
    val = platform_cfg.get(odds_type)
    if val is None:
        return None
    if isinstance(val, dict):
        if "all_correct" in val:
            return float(val["all_correct"])
        return float(max(val.values()))
    return float(val)


def kelly_cap() -> float:
    return float(_cfg().get("kelly_cap", 0.25))
