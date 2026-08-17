"""Per-matchup feature attribution for the winner model (read-only).

`evaluation/feature_importance.py` gives *global* importance. This module
answers the per-fight question the UI needs — "what is driving THIS P(win)?" —
by occlusion: for each interpretable feature *group* (reach, striking, grappling
threat, finishing, ratings, experience) it neutralises the group's A-vs-B edge
and measures how far P(red win) moves. Pure read-only use of the already-trained
WinnerModel: no training, no fitting, no model mutation.

Neutralisation rule (uniform for every column type):

    neutral_c = (feat[c] + feat_flip[c]) / 2     applied to BOTH frames

Because feat is the (red, blue) perspective and feat_flip is (blue, red), the
midpoint removes the red-vs-blue tilt of that column regardless of whether it is
an ``_a``/``_b`` pair (→ shared mean), a ``_diff`` (→ 0), an unpaired side
(pruned sibling), or a symmetric matchup constant (→ unchanged, contributes 0).
The driver delta is then  p_full − p_neutral_group  in P(red) units.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Strip trailing corner/diff and the rolling-window token to reach the base concept,
# e.g. "td_def_success_per_15_l5_a" -> "td_def_success_per_15".
_WIN_RE = re.compile(r"_(ctd|l3|l5|2y|decay)(?=_|$)")


def _base(col: str) -> str:
    c = re.sub(r"_(a|b|diff)$", "", col)
    return _WIN_RE.sub("", c)


# (key, label, accent CSS var, substring matchers on the base concept).
# First group whose matcher is a substring of the base wins → ORDER MATTERS
# (specific concepts before the generic group that would also swallow them).
_GROUP_DEFS: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("ratings", "Rating & momentum", "var(--gold-br)", (
        "elo", "glicko", "ts_mu", "ts_sigma", "ts_z", "pre_ufc_win_rate", "opp_elo",
        "def_pct_trend", "volume_trend",
    )),
    ("grappling", "Grappling & control", "var(--m-td)", (
        "td_per_15", "td_acc", "td_def", "td_attempted", "ctrl", "sub_att", "sub_def",
        "reversal", "grappler", "wrestler", "grappling_control_threat", "sub_trap",
        "tdd_vs_wrestler", "r1_td", "r1_sub_att", "r1_ctrl",
    )),
    # KO/finishing PROPENSITY *and* punching power live together — a power-puncher
    # who ends fights should score here, not get split off into a "striking"
    # volume bar. Knockdowns / damage / power-density therefore belong with the
    # finish rates, NOT with strike volume (that split read as a contradiction:
    # "Striking & power → A, but Finishing ability → B").
    ("finishing", "Power & finishing", "var(--m-r1)", (
        "ko_win_rate", "sub_win_rate", "ko_loss_rate", "sub_loss_rate", "finish_rate",
        "dec_rate", "early_finish", "r1_ko_win", "r1_sub_win", "prior_5rd_dec",
        "ko_specialist", "sub_specialist", "sub_threat", "ko_threat", "finish_combined",
        "dec_prone", "combined_finish", "combined_dec", "expected_p_r1_finish",
        "ko_matchup", "chin_threat", "power_vs_chin", "layoff_chin",
        "kd_", "r1_kd", "combined_power_density", "pace_x_power", "damage_index",
    )),
    ("reach", "Reach & size", "var(--m-sig)", (
        "reach", "height", "weight",
    )),
    # Strike OUTPUT only — volume / accuracy / pace / targeting (the "busy
    # striker" signal: how much and how accurately a fighter throws). Power &
    # knockdowns live in "Power & finishing"; strikes ABSORBED + strike defence
    # live in "Striking defense" below. Keeping them apart matters: a high-output
    # but leaky striker (Kape: slpm ~5.4 vs 3.8, but sapm ~3.8 vs 2.5) otherwise
    # reads as *losing* one merged "striking" bar because the opponent's much
    # lower absorbed-rate dominates the net — which is not mentally verifiable.
    ("striking", "Striking output", "var(--m-body)", (
        "slpm", "str_acc", "head_share", "body_share", "leg_share",
        "distance_share", "clinch_share", "ground_share", "head_acc", "body_acc",
        "leg_acc", "vol_attempted", "striker_score", "pressure_score",
        "volume_score", "r1_sig_str", "r2_sig_str", "r3_sig_str", "r4_sig_str",
        "r5_sig_str", "pace", "early_vol_ratio", "combined_slpm",
        "combined_vol", "expected_total_strikes", "cardio",
    )),
    ("striking_def", "Striking defense", "var(--m-sub)", (
        "sapm", "str_def", "opp_hittability",
    )),
    # Age and experience pull in OPPOSITE directions (younger is better, but the
    # younger fighter usually has *fewer* fights) — bundling them into one bar
    # nets out a real tug-of-war and reads backwards. Keep them as two groups so
    # the UI can show "youth/freshness favours X" and "tenure favours Y" separately.
    ("age", "Age & freshness", "var(--m-ctrl)", (
        "age", "layoff",
    )),
    # Accumulated damage is a CUMULATIVE count, so it tracks how much a fighter
    # has fought rather than how durable they are: a 2-fight newcomer has absorbed
    # ~70 career strikes against a 25-round veteran's ~486, reading as "fresh"
    # purely from having barely fought. That is the opposite direction to both
    # neighbours — more mileage is bad, but younger is good and more tenure is
    # good — so inside either bar it nets against the signal it sits with, the
    # same tug-of-war the age/experience split above exists to avoid.
    #
    # Measured, not assumed: splitting it out did NOT move the Erceg vs Temirov
    # bar it was suspected of flipping (that bar stayed -1.19pp; the flip is
    # driven by age, Temirov being 1.5y younger, and that fight's attribution is
    # quantisation-limited anyway). Standalone it cleared the _MIN_DELTA noise
    # floor in 0 of 11 fights on the 2026-07-25 card. This split is therefore
    # structural hygiene, not a fix for an observed number.
    ("mileage", "Mileage & damage", "var(--m-kd)", (
        "total_sig_str_absorbed",
    )),
    ("experience", "Experience", "var(--m-rnd)", (
        "fights_career", "total_rounds_career",
    )),
]

# Minimum |delta| (probability) for a group to be reported as a driver — below
# ~0.3pp it is noise, not signal.
_MIN_DELTA = 0.003


def _assign_group(col: str) -> str | None:
    b = _base(col)
    for key, _label, _accent, matchers in _GROUP_DEFS:
        if any(m in b for m in matchers):
            return key
    return None


def winner_drivers(
    model,
    feat: pd.DataFrame,
    feat_flip: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Top feature-group drivers of P(red win) for one matchup.

    Returns a list (sorted by magnitude, longest first) of:
        {key, label, accent, delta, magnitude, favors}
    where ``delta`` is signed in P(red) units (>0 favours red / corner A) and
    ``favors`` is "a" or "b". Empty list on any failure — never raises.
    """
    cols = list(getattr(model, "feature_cols", []) or [])
    if not cols:
        return []

    base_a = feat.reindex(columns=cols, fill_value=0).fillna(0)
    base_b = feat_flip.reindex(columns=cols, fill_value=0).fillna(0)

    group_cols: dict[str, list[str]] = {}
    for c in cols:
        g = _assign_group(c)
        if g is not None:
            group_cols.setdefault(g, []).append(c)
    order = [k for k, *_ in _GROUP_DEFS if k in group_cols]
    if not order:
        return []

    # One batched predict: [feat, feat_flip, (neutralised a/b per group)...]
    frames = [base_a, base_b]
    for g in order:
        na, nb = base_a.copy(), base_b.copy()
        for c in group_cols[g]:
            neutral = (na[c].to_numpy() + nb[c].to_numpy()) / 2.0
            na[c] = neutral
            nb[c] = neutral
        frames.append(na)
        frames.append(nb)

    batch = pd.concat(frames, ignore_index=True)
    probs = np.asarray(model.predict_proba(batch), dtype=float)

    from ufc.training.symmetrize import inference_average

    p_full = inference_average(float(probs[0]), float(probs[1]))

    drivers: list[dict] = []
    for i, g in enumerate(order):
        pa = float(probs[2 + 2 * i])
        pb = float(probs[2 + 2 * i + 1])
        delta = p_full - inference_average(pa, pb)
        _label, _accent = next((l, a) for k, l, a, _ in _GROUP_DEFS if k == g)
        drivers.append({
            "key": g,
            "label": _label,
            "accent": _accent,
            "delta": round(delta, 4),
            "magnitude": round(abs(delta), 4),
            "favors": "a" if delta >= 0 else "b",
        })

    drivers.sort(key=lambda d: d["magnitude"], reverse=True)
    significant = [d for d in drivers if d["magnitude"] >= _MIN_DELTA]
    return (significant or drivers)[:top_n]
