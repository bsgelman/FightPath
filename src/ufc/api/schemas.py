"""Pydantic request/response schemas for the FastAPI service.

Field names use camelCase (alias) to match the React UI design contract.
Python attribute names stay snake_case; `populate_by_name=True` allows
either convention when constructing from dicts.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Camel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


# ── Requests ─────────────────────────────────────────────────────────────────

class PredictRequest(_Camel):
    red:         str = Field(..., min_length=1, max_length=100)
    blue:        str = Field(..., min_length=1, max_length=100)
    rounds:      Literal[3, 5] = 3
    is_title:    bool = Field(False, alias="isTitle")
    event_date:  str  = Field(..., alias="eventDate", max_length=10)
    weight_class: Optional[str] = Field(None, alias="weightClass", max_length=100)
    referee:     str = Field("", max_length=100)
    location:    str = Field("", max_length=100)


class PortfolioLeg(_Camel):
    fight_id: str = Field(..., alias="fightId", max_length=100)
    model_p:  float = Field(..., ge=0.0, le=1.0, alias="modelP")
    label:    str = Field("", max_length=200)
    market:   str = Field("", max_length=50)
    side:     str = Field("over", max_length=10)
    line:     Optional[float] = None
    corner:   str = Field("fight", max_length=10)


class PortfolioRequest(_Camel):
    legs:       list[PortfolioLeg] = Field(..., max_length=20)
    payout_key: str  = Field("pp_power_2", alias="payoutKey", max_length=20)
    mult:       float = Field(3.0, gt=1.0, le=50.0)


# ── Response atoms ────────────────────────────────────────────────────────────

class CurvePoint(BaseModel):
    x: float
    p: float


class QuantileRow(_Camel):
    label: str
    value: Optional[float]


class CountSummary(_Camel):
    mean:  float
    sd:    float
    p0:    float
    q:     dict[str, float]


class CountPropOut(_Camel):
    curve:     list[list[float]]
    hist:      list[dict]
    quantiles: list[QuantileRow]
    summary:   CountSummary
    p_zero:    float = Field(0.0, alias="pZero")


class MethodOut(_Camel):
    ko:  float
    sub: float
    dec: float


class FinishOut(_Camel):
    ko_finish:  Optional[float] = Field(None, alias="koFinish")
    sub_finish: Optional[float] = Field(None, alias="subFinish")
    finish:     Optional[float] = None
    r1_finish:  Optional[float] = Field(None, alias="r1Finish")


class ConfidenceOut(_Camel):
    lo: float
    hi: float


class FighterOut(_Camel):
    name:    str
    id:      str
    p_win:   float = Field(..., alias="pWin")
    method:  MethodOut
    sig:     Optional[dict] = None
    r1sig:   Optional[dict] = None
    td:      Optional[dict] = None
    finish:  Optional[dict] = None


class FightOut(_Camel):
    id:           str
    slot:         str
    rounds:       int
    is_title:     bool = Field(..., alias="isTitle")
    weight_class: str  = Field("", alias="weightClass")
    a:            FighterOut
    b:            FighterOut
    method:       MethodOut
    inside:       float
    p_dec:        float = Field(..., alias="pDec")
    confidence:   ConfidenceOut
    dur_curve:    list  = Field(..., alias="durCurve")
    dur_quantiles: list = Field(..., alias="durQuantiles")
    round_dist:   list  = Field(..., alias="roundDist")
    median_min:   Optional[float] = Field(None, alias="medianMin")
    sched_sec:    float = Field(..., alias="schedSec")
    r1_finish:    Optional[float] = Field(None, alias="r1Finish")


class EventOut(_Camel):
    code:  str
    name:  str
    venue: str
    date:  str


class CardOut(_Camel):
    event:  EventOut
    fights: list[FightOut]


class CardListItem(_Camel):
    id:          str
    label:       str
    event_date:  str = Field(..., alias="eventDate")
    n_fights:    int = Field(..., alias="nFights")


# ── Positions ─────────────────────────────────────────────────────────────────

class BestBetOut(_Camel):
    fight:       str
    fighter:     str
    market:      str
    side:        str
    line:        float
    platform:    str
    model_p:     float = Field(..., alias="modelP")
    implied_p:   float = Field(..., alias="impliedP")
    edge_pct:    float = Field(..., alias="edgePct")
    kelly:       float
    band:        list[float]
    fight_idx:   int   = Field(..., alias="fightIdx")
    corner:      str
    card_red:    str   = Field(..., alias="cardRed")
    card_blue:   str   = Field(..., alias="cardBlue")


# ── Portfolio ──────────────────────────────────────────────────────────────────

class PortfolioOut(_Camel):
    n_legs:              int   = Field(..., alias="nLegs")
    individual_probs:    list[float] = Field(..., alias="individualProbs")
    naive_joint_prob:    float = Field(..., alias="naiveJointProb")
    mc_joint_prob:       float = Field(..., alias="mcJointProb")
    correlation_adj:     float = Field(..., alias="correlationAdj")
    ev:                  float
    breakeven:           float
    kelly:               float
    verdict:             str


# ── History ───────────────────────────────────────────────────────────────────

class HistoryFightRow(_Camel):
    red:          str
    blue:         str
    p_red:        float = Field(..., alias="pRed")
    pred_winner:  str   = Field(..., alias="predWinner")
    actual_winner: str  = Field(..., alias="actualWinner")
    correct:      bool


class HistoryEventOut(_Camel):
    id:       str
    event:    str
    date:     str
    correct:  int
    total:    int
    hit_rate: float = Field(..., alias="hitRate")
    fights:   list[HistoryFightRow]


# ── Meta ──────────────────────────────────────────────────────────────────────

class MetaOut(_Camel):
    version:        str
    last_sync:      str = Field(..., alias="lastSync")
    cards_analyzed: int = Field(..., alias="cardsAnalyzed")
    hit_rate:       float = Field(..., alias="hitRate")
    roi:            float
    units:          float
    calib:          list[dict]
