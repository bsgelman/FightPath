"""Edge calculation: model probability vs implied probability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ufc.valuation.lines import Line
from ufc.valuation.payouts import implied_prob_per_leg, kelly_cap


@dataclass
class Edge:
    market: str
    side: str
    line_value: float
    model_prob: float
    implied_prob: float
    edge_pct: float
    kelly_fraction: float
    confidence_band: tuple[float, float]
    fighter_name: str | None = None
    payout_type: str = ""


def evaluate_line(line: Line, prediction: Any) -> Edge:
    """Compute edge for a single line against model prediction.

    prediction: PropCDF or DurationCDF object with .p_over(line) / .p_under(line) / .uncertainty_band()
                OR a float (for winner market).
    """
    implied = implied_prob_per_leg(line.payout_type, multiplier=line.payout_multiplier)

    if line.market == "winner":
        # prediction is a float P(this fighter wins)
        model_p = float(prediction)
    elif line.side == "over":
        model_p = float(prediction.p_over(line.line_value))
    elif line.side == "under":
        model_p = float(prediction.p_under(line.line_value))
    else:
        model_p = 0.5

    # Per-leg Kelly: f = (p - q) / (1 - q) where q = implied breakeven per leg.
    # This handles multi-pick parlays correctly — implied already encodes the parlay structure.
    kelly = max(0.0, (model_p - implied) / (1.0 - implied)) if implied < 1.0 else 0.0
    kelly = min(kelly, kelly_cap())

    # Uncertainty band
    try:
        band = prediction.uncertainty_band(line.line_value)
    except Exception:
        se = (model_p * (1 - model_p)) ** 0.5 / 30
        band = (max(0, model_p - 1.28 * se), min(1, model_p + 1.28 * se))

    return Edge(
        market=line.market,
        side=line.side,
        line_value=line.line_value,
        model_prob=model_p,
        implied_prob=implied,
        edge_pct=model_p - implied,
        kelly_fraction=kelly,
        confidence_band=band,
        fighter_name=line.fighter_name,
        payout_type=line.payout_type,
    )
