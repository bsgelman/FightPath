"""Settlement predicates for all Kalshi market kinds — the single source of
truth 08b_grade_props.py routes into for grading. Kept separate from the
script so the predicate table is importable and fixture-testable on its own.

Predicates read the RAW features_props method label (e.g. "DQ", "U-DEC"), not
the model's 3-class method_class (KO/TKO|SUB|DEC) — method_class folds DQ and
NC into "DEC" (METHOD_MAP in ufc.models.method), which is correct for model
training but wrong for Kalshi settlement: Kalshi's KXUFCMOF "-KOTKODQ" and
KXUFCVICROUND round-win markets explicitly count a DQ as a KO/TKO-family
result, not a decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

DEC_RAW_METHODS = frozenset({"U-DEC", "S-DEC", "M-DEC"})
_FINISH_RAW = frozenset({"KO/TKO", "SUB", "DQ"})  # DQ ends a fight early; Kalshi buckets it with KO/TKO

_END_BEFORE_RE = re.compile(r"^end_before_r(\d+)$")
_WIN_IN_RE = re.compile(r"^win_in_r(\d+)$")


@dataclass(frozen=True)
class FightOutcome:
    """A resolved fight's outcome, from the NAMED fighter's perspective for the
    per-fighter fields."""
    fighter_won: Optional[bool]     # the NAMED fighter; None = draw or unresolved
    raw_method: str                 # RAW features_props label — NOT method_class
    end_round: Optional[int]
    scheduled_rounds: Optional[int]
    total_fight_sec: Optional[float]
    is_draw: bool                   # won_a NaN AND raw_method in DEC_RAW_METHODS (KO/TKO+NaN
                                     # is a data gap, not a draw — see is_draw callers)


@dataclass(frozen=True)
class KindSpec:
    needs_fighter: bool                                   # True -> 08b must resolve the named fighter first
    nc_hit: Optional[bool]                                 # None = void on NC; True = NC settles YES (vicround_other only)
    predicate: Callable[[FightOutcome], Optional[bool]]    # None return = pending (can't grade yet)


def _pred_winner(o: FightOutcome) -> bool:
    return bool(o.fighter_won)


def _pred_method_ko(o: FightOutcome) -> bool:
    return bool(o.fighter_won) and o.raw_method in ("KO/TKO", "DQ")


def _pred_method_sub(o: FightOutcome) -> bool:
    return bool(o.fighter_won) and o.raw_method == "SUB"


def _pred_method_dec(o: FightOutcome) -> bool:
    return bool(o.fighter_won) and o.raw_method in DEC_RAW_METHODS


def _pred_distance(o: FightOutcome) -> bool:
    return o.raw_method in DEC_RAW_METHODS


def _pred_mof_ko(o: FightOutcome) -> bool:
    return o.raw_method in ("KO/TKO", "DQ")


def _pred_mof_sub(o: FightOutcome) -> bool:
    return o.raw_method == "SUB"


def _pred_mof_dec(o: FightOutcome) -> bool:
    return o.raw_method in DEC_RAW_METHODS and not o.is_draw


def _pred_vicround_other(o: FightOutcome) -> bool:
    return o.raw_method in DEC_RAW_METHODS or o.is_draw


def _end_before_pred(r: int) -> Callable[[FightOutcome], Optional[bool]]:
    def _p(o: FightOutcome) -> Optional[bool]:
        if o.raw_method not in _FINISH_RAW:
            return False               # went the distance-family route -> didn't end early
        if o.end_round is None:
            return None                # finish confirmed but round unknown -> pending
        return o.end_round < r
    return _p


def _win_in_pred(r: int) -> Callable[[FightOutcome], Optional[bool]]:
    def _p(o: FightOutcome) -> Optional[bool]:
        if not bool(o.fighter_won):
            return False               # named fighter didn't win -> can't have won in round r
        if o.raw_method not in _FINISH_RAW:
            return False               # won by decision-family, not a round finish
        if o.end_round is None:
            return None                # finish confirmed but round unknown -> pending
        return o.end_round == r
    return _p


_EXACT_KIND_SPECS: dict[str, KindSpec] = {
    "winner":         KindSpec(True,  None, _pred_winner),
    "method_ko":      KindSpec(True,  None, _pred_method_ko),
    "method_sub":     KindSpec(True,  None, _pred_method_sub),
    "method_dec":     KindSpec(True,  None, _pred_method_dec),
    "distance":       KindSpec(False, None, _pred_distance),
    "mof_ko":         KindSpec(False, None, _pred_mof_ko),
    "mof_sub":        KindSpec(False, None, _pred_mof_sub),
    "mof_dec":        KindSpec(False, None, _pred_mof_dec),
    "vicround_other": KindSpec(False, True, _pred_vicround_other),
}


def kind_spec(market_kind: str) -> Optional[KindSpec]:
    """The KindSpec for one Kalshi market_kind, or None if it isn't a
    recognized Kalshi kind (e.g. a DFS market name, or genuine schema drift)."""
    spec = _EXACT_KIND_SPECS.get(market_kind)
    if spec is not None:
        return spec
    m = _END_BEFORE_RE.match(market_kind)
    if m is not None:
        return KindSpec(False, None, _end_before_pred(int(m.group(1))))
    m = _WIN_IN_RE.match(market_kind)
    if m is not None:
        return KindSpec(True, None, _win_in_pred(int(m.group(1))))
    return None


def settle(market_kind: str, o: FightOutcome) -> tuple[Optional[float], Optional[bool], Optional[str]]:
    """Settle one Kalshi ledger row. Returns (realized_value, hit_bool, status),
    matching 08b_grade_props._resolve_row's contract: status is 'resolved',
    'void', or None (leave pending — result not knowable yet)."""
    spec = kind_spec(market_kind)
    if spec is None:
        return None, None, None

    if o.raw_method == "NC":
        if spec.nc_hit is None:
            return None, None, "void"
        hit = bool(spec.nc_hit)
        return (1.0 if hit else 0.0), hit, "resolved"

    if spec.needs_fighter and o.fighter_won is None and not o.is_draw:
        return None, None, None

    hit = spec.predicate(o)
    if hit is None:
        return None, None, None

    hit = bool(hit)
    return (1.0 if hit else 0.0), hit, "resolved"
