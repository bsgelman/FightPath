"""Single source of truth for prop count model target specifications.

Each PropTargetSpec describes one trained prop model:
  - target      : artifact name suffix → props_{target}_{gitsha}.joblib
  - raw_col_a   : label column in features_props.parquet (fighter-A side)
  - ceiling     : active_minutes_ceiling in minutes (5.0 = R1 cap; None = full fight)
  - weight      : sample weighting scheme ("recency" | "censor" | None)
  - rate_calib  : whether to fit val-anchored rate_calib_factor after training
  - rate_ceiling: per-rate-draw clip (None = no clip; 60.0 for ctrl_time)

Weighting rationale
-------------------
- recency   : exponential down-weighting of old fights (halflife 730d, floor 0.05).
              Tracks era pace drift (sig_strikes, body/leg, knockdowns, r1 models).
- censor    : right-censoring weight (fight_sec / scheduled_sec, clip [0.1, 1]).
              Short fights provide less exposure → weight reflects information content.
              Used for takedowns, sub_attempts, ctrl_time.
- knockdowns: recency NOT censor — short KO fights are the KD-rich rows; censoring
              would down-weight exactly the positives that price the 0.5 line.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PropTargetSpec:
    target: str
    raw_col_a: str
    ceiling: float | None       # active_minutes_ceiling in minutes
    weight: str | None          # "recency" | "censor" | None
    rate_calib: bool
    rate_ceiling: float | None = field(default=None)
    model_kind: str = field(default="rate_hurdle")  # "rate_hurdle" | "control_share"


PROP_TARGET_SPECS: list[PropTargetSpec] = [
    PropTargetSpec("sig_strikes",      "sig_str_landed_a",    None, "recency", True),
    PropTargetSpec("takedowns",        "td_landed_a",         None, "censor",  False),
    PropTargetSpec("r1_sig_strikes",   "r1_sig_str_landed_a", 5.0,  "recency", False),
    PropTargetSpec("knockdowns",       "kd_for_a",            None, "recency", False),
    PropTargetSpec("sub_attempts",     "sub_att_for_a",       None, "censor",  False),
    PropTargetSpec("r1_takedowns",     "r1_td_landed_a",      5.0,  "recency", False),
    PropTargetSpec("body_sig_strikes", "body_landed_a",       None, "recency", True),
    PropTargetSpec("leg_sig_strikes",  "leg_landed_a",        None, "recency", True),
    PropTargetSpec("ctrl_time",        "ctrl_sec_a",          None, "censor",  False, rate_ceiling=60.0, model_kind="control_share"),
]
