"""Shared prop line menus.

Single source of truth for:
  STANDARD_LINES  — the synthetic line menu evaluated by the API prop-edge endpoint.
  SCORECARD_LINES — characteristic lines evaluated by the edge scorecard (05c).

Both are consumed by:
  src/ufc/api/app.py          (STANDARD_LINES)
  scripts/05c_evaluate_prop_edge.py  (SCORECARD_LINES + STANDARD_LINES)
"""
from __future__ import annotations

# ── API synthetic line menu ────────────────────────────────────────────────────
# Each entry: (line_value, market, side, corner)
# line_value in canonical units (seconds for duration/ctrl_time, raw count otherwise).
# corner: "red" | "blue" | "fight"
STANDARD_LINES: list[tuple] = [
    (19.5, "sig_strikes",       "over",  "red"),
    (19.5, "sig_strikes",       "over",  "blue"),
    (19.5, "sig_strikes",       "under", "red"),
    (19.5, "sig_strikes",       "under", "blue"),
    (29.5, "sig_strikes",       "over",  "red"),
    (29.5, "sig_strikes",       "over",  "blue"),
    (1.5,  "takedowns",         "over",  "red"),
    (1.5,  "takedowns",         "over",  "blue"),
    (0.5,  "takedowns",         "over",  "red"),
    (0.5,  "takedowns",         "over",  "blue"),
    (14.5, "r1_sig_strikes",    "over",  "red"),
    (14.5, "r1_sig_strikes",    "over",  "blue"),
    (0.5,  "knockdowns",        "over",  "red"),
    (0.5,  "knockdowns",        "over",  "blue"),
    (0.5,  "sub_attempts",      "over",  "red"),
    (0.5,  "sub_attempts",      "over",  "blue"),
    (0.5,  "r1_takedowns",      "over",  "red"),
    (0.5,  "r1_takedowns",      "over",  "blue"),
    (4.5,  "body_sig_strikes",  "over",  "red"),
    (4.5,  "body_sig_strikes",  "over",  "blue"),
    (14.5, "body_sig_strikes",  "over",  "red"),
    (14.5, "body_sig_strikes",  "over",  "blue"),
    (9.5,  "leg_sig_strikes",   "over",  "red"),
    (9.5,  "leg_sig_strikes",   "over",  "blue"),
    (105.0, "ctrl_time",        "over",  "red"),
    (105.0, "ctrl_time",        "over",  "blue"),
    (76.5,  "sig_strikes_combo","over",  "fight"),
]

# ── Scorecard evaluation lines ─────────────────────────────────────────────────
# Characteristic lines for resolution/calibration/edge scoring in 05c.
# Keys match canonical prop target names (PropTargetSpec.target).
# Values are lists of line values in canonical units.
# Duration lines in seconds; ctrl_time in seconds; all count props in raw count units.
SCORECARD_LINES: dict[str, list[float]] = {
    "sig_strikes":      [14.5, 19.5, 24.5, 29.5, 39.5],
    "r1_sig_strikes":   [9.5, 14.5, 19.5],
    "takedowns":        [0.5, 1.5, 2.5],
    "knockdowns":       [0.5],
    "sub_attempts":     [0.5, 1.5],
    "r1_takedowns":     [0.5, 1.5, 2.5],
    "body_sig_strikes": [4.5, 7.5, 9.5, 14.5],
    "leg_sig_strikes":  [4.5, 9.5, 14.5],
    "ctrl_time":        [60.0, 105.0, 150.0],    # 1, 1.75, 2.5 min
    "duration":         [450.0, 600.0, 750.0],   # 7.5, 10, 12.5 min
}

# Canonical prop key → frontend market key (used by /api/prop-trust response)
CANONICAL_TO_FRONTEND: dict[str, str] = {
    "sig_strikes":      "sig",
    "r1_sig_strikes":   "r1sig",
    "takedowns":        "td",
    "knockdowns":       "kd",
    "sub_attempts":     "subAtt",
    "r1_takedowns":     "r1td",
    "body_sig_strikes": "bodySig",
    "leg_sig_strikes":  "legSig",
    "ctrl_time":        "ctrl",
    "duration":         "duration",
    "rounds":           "rounds",
    "sig_strikes_combo":"combo",
    "ko_finish":        "ko_finish",
    "sub_finish":       "sub_finish",
    "finish":           "finish",
    "r1_finish":        "r1_finish",
}
