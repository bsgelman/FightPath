"""Style-counter interaction features for the wide (A vs B) fight DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-linear style-counter cross-terms.

    Expects a wide DataFrame with _a / _b suffix columns for each fighter.
    """
    out = df.copy()

    def _a(col, default=0.0):
        c = f"{col}_a"
        return out[c].fillna(default) if c in out.columns else pd.Series(default, index=out.index)

    def _b(col, default=0.0):
        c = f"{col}_b"
        return out[c].fillna(default) if c in out.columns else pd.Series(default, index=out.index)

    # 1. Submission trap: A wrestles into B's submissions
    out["sub_trap_a"] = _a("wrestler_score") * (
        _b("sub_att_per_15_decay") * _b("sub_def_decay")
    )
    out["sub_trap_b"] = _b("wrestler_score") * (
        _a("sub_att_per_15_decay") * _a("sub_def_decay")
    )

    # 2. Reach punish: B's long-reach kickboxing vs A at reach disadvantage
    reach_adv_b = (_b("reach_in", 70) - _a("reach_in", 70)).clip(lower=0)
    out["reach_punish_a"] = reach_adv_b * _b("distance_share_decay") * _b("str_acc_decay")
    reach_adv_a = (_a("reach_in", 70) - _b("reach_in", 70)).clip(lower=0)
    out["reach_punish_b"] = reach_adv_a * _a("distance_share_decay") * _a("str_acc_decay")

    # 3. Cardio gap in 5-rounders
    if "scheduled_rounds" in out.columns:
        is_5rd = (out["scheduled_rounds"].fillna(3) >= 5).astype(float)
    else:
        is_5rd = pd.Series(0.0, index=out.index)
    cardio_a = _a("cardio_score", 1.0)
    cardio_b = _b("cardio_score", 1.0)
    out["cardio_gap_5rd"] = (cardio_a - cardio_b) * is_5rd

    # 4. Power vs chin
    out["power_vs_chin_a"] = _a("kd_per_15_decay") * _b("chin_proxy", 0.0)
    out["power_vs_chin_b"] = _b("kd_per_15_decay") * _a("chin_proxy", 0.0)

    # 5. TDD vs wrestler
    out["tdd_vs_wrestler_a"] = _b("td_def_decay") * _a("td_per_15_decay")
    out["tdd_vs_wrestler_b"] = _a("td_def_decay") * _b("td_per_15_decay")

    # 6. Pace differential
    out["pace_diff"] = _a("volume_score") - _b("volume_score")

    # 7. Age × heavyweight non-linearity
    is_hw = out.get("weight_class", pd.Series("", index=out.index)).str.contains(
        r"\bHeavyweight\b", case=False, na=False
    ).astype(float)
    out["age_hw_a"] = _a("age_years", 28) * is_hw * (_a("age_years", 28) > 33).astype(float)
    out["age_hw_b"] = _b("age_years", 28) * is_hw * (_b("age_years", 28) > 33).astype(float)

    # 8. Layoff × age
    out["layoff_age_a"] = _a("layoff_days", 0) * (
        (_a("age_years", 28) - 32).clip(lower=0)
    )
    out["layoff_age_b"] = _b("layoff_days", 0) * (
        (_b("age_years", 28) - 32).clip(lower=0)
    )

    # 9. Reach × leg kick share (reach_diff is a single non-suffixed col from physical.py)
    _reach_diff = out["reach_diff"] if "reach_diff" in out.columns else pd.Series(0.0, index=out.index)
    out["reach_kick_a"] = _reach_diff.clip(lower=0) * _a("leg_share_decay")
    out["reach_kick_b"] = (-_reach_diff).clip(lower=0) * _b("leg_share_decay")

    # 9b. Reach offense: own reach advantage × distance fighting × accuracy
    out["reach_offense_a"] = _reach_diff.clip(lower=0) * _a("distance_share_decay") * _a("str_acc_decay")
    out["reach_offense_b"] = (-_reach_diff).clip(lower=0) * _b("distance_share_decay") * _b("str_acc_decay")

    # 9c. Opponent hittability: opponent absorbs hits AND defends poorly
    out["opp_hittability_a"] = _b("sapm_decay") * (1.0 - _b("str_def_decay")).clip(lower=0)
    out["opp_hittability_b"] = _a("sapm_decay") * (1.0 - _a("str_def_decay")).clip(lower=0)

    # 10. Stance history performance (career win rate vs opponent stance)
    out["stance_history_perf_a"] = _a("stance_wr_vs_opp_stance", 0.5)
    out["stance_history_perf_b"] = _b("stance_wr_vs_opp_stance", 0.5)

    # 11. Finish-rate matchup cross-terms (KO offense × chin; SUB offense × sub defense weakness)
    out["ko_threat_a"] = _a("ko_win_rate_decay") * _b("ko_loss_rate_decay")
    out["ko_threat_b"] = _b("ko_win_rate_decay") * _a("ko_loss_rate_decay")
    out["sub_threat_a"] = _a("sub_win_rate_decay") * _b("sub_loss_rate_decay")
    out["sub_threat_b"] = _b("sub_win_rate_decay") * _a("sub_loss_rate_decay")
    # Joint finisher index: P(neither avoids a finish)
    out["finish_combined"] = 1.0 - (1.0 - _a("finish_rate_decay")) * (1.0 - _b("finish_rate_decay"))
    # Dec-prone differential (both sides tend to go to decision)
    out["dec_prone_combined"] = _a("dec_rate_decay") * _b("dec_rate_decay")

    # 12. R1-specific finish threat (first-round predators like Topuria)
    out["r1_ko_threat_a"] = _a("r1_ko_win_rate_decay") * _b("ko_loss_rate_decay")
    out["r1_ko_threat_b"] = _b("r1_ko_win_rate_decay") * _a("ko_loss_rate_decay")
    out["r1_sub_threat_a"] = _a("r1_sub_win_rate_decay") * _b("sub_loss_rate_decay")
    out["r1_sub_threat_b"] = _b("r1_sub_win_rate_decay") * _a("sub_loss_rate_decay")

    # 13. Method ↔ pace coupling (replaces leaky cardio_ratio_fight signal).
    # Combined fight-level pace and power, so the method classifier sees
    # an explicit pace × power product (high tempo + low chin → KO probability).
    slpm_a = _a("slpm_decay")
    slpm_b = _b("slpm_decay")
    vol_a = _a("vol_attempted_pm_decay")
    vol_b = _b("vol_attempted_pm_decay")
    kdps_a = _a("kd_per_sig_decay")
    kdps_b = _b("kd_per_sig_decay")
    sched = out.get("scheduled_rounds", pd.Series(3.0, index=out.index)).fillna(3.0).astype(float)

    out["combined_slpm"] = slpm_a + slpm_b
    out["combined_vol_attempted_pm"] = vol_a + vol_b
    out["expected_total_strikes"] = out["combined_slpm"] * sched * 5.0  # rough fight-volume forecast
    out["combined_power_density"] = kdps_a + kdps_b
    out["pace_x_power_a"] = vol_a * kdps_b  # A's pace exposes B's chin → KO risk for B
    out["pace_x_power_b"] = vol_b * kdps_a
    out["combined_finish_rate"] = _a("finish_rate_decay") + _b("finish_rate_decay")
    out["combined_dec_rate"] = _a("dec_rate_decay") + _b("dec_rate_decay")

    # 14. Specialist amplifiers — non-linear "this fighter wins by X" signals.
    # Replaces flat mean-based grappler_score that flattens true specialists.
    # log1p amplification: 1→3 sub_att/15 matters more than 7→9 (diminishing returns).
    sub_win_a = _a("sub_win_rate_ctd")
    sub_win_b = _b("sub_win_rate_ctd")
    ko_win_a = _a("ko_win_rate_ctd")
    ko_win_b = _b("ko_win_rate_ctd")
    sub_att_a = _a("sub_att_per_15_decay")
    sub_att_b = _b("sub_att_per_15_decay")
    kd_per_a = _a("kd_per_15_decay")
    kd_per_b = _b("kd_per_15_decay")
    sub_loss_a = _a("sub_loss_rate_decay")
    sub_loss_b = _b("sub_loss_rate_decay")
    ko_loss_a = _a("ko_loss_rate_decay")
    ko_loss_b = _b("ko_loss_rate_decay")

    out["sub_specialist_idx_a"] = sub_win_a * np.log1p(sub_att_a * 15.0)
    out["sub_specialist_idx_b"] = sub_win_b * np.log1p(sub_att_b * 15.0)
    out["sub_specialist_x_weakness_a"] = out["sub_specialist_idx_a"] * (sub_loss_b + 0.1)
    out["sub_specialist_x_weakness_b"] = out["sub_specialist_idx_b"] * (sub_loss_a + 0.1)

    out["ko_specialist_idx_a"] = ko_win_a * np.log1p(kd_per_a * 15.0)
    out["ko_specialist_idx_b"] = ko_win_b * np.log1p(kd_per_b * 15.0)
    out["ko_specialist_x_weakness_a"] = out["ko_specialist_idx_a"] * (ko_loss_b + 0.1)
    out["ko_specialist_x_weakness_b"] = out["ko_specialist_idx_b"] * (ko_loss_a + 0.1)

    # 15. Finish-share — fraction of decided outcomes ending in a finish.
    # Captures "this fighter never goes to decision" (Ngannou, Khabib early career, Oliveira).
    dec_a = _a("dec_rate_ctd")
    dec_b = _b("dec_rate_ctd")
    out["finish_share_a"] = (ko_win_a + sub_win_a) / (ko_win_a + sub_win_a + dec_a + 0.1)
    out["finish_share_b"] = (ko_win_b + sub_win_b) / (ko_win_b + sub_win_b + dec_b + 0.1)

    # 16. Grappling control threat — captures TOP-POSITION accumulation, not just submission attempts.
    # This is the Oliveira-style signal: he gets top, holds it, then finishes.
    # (1 - opp_td_def) flips td_def into vulnerability.
    td_per_a = _a("td_per_15_decay")
    td_per_b = _b("td_per_15_decay")
    ctrl_a = _a("ctrl_pct_decay")
    ctrl_b = _b("ctrl_pct_decay")
    td_def_a = _a("td_def_decay")
    td_def_b = _b("td_def_decay")
    out["grappling_control_threat_a"] = td_per_a * ctrl_a * (1.0 - td_def_b).clip(lower=0)
    out["grappling_control_threat_b"] = td_per_b * ctrl_b * (1.0 - td_def_a).clip(lower=0)

    # 18. KO matchup discrimination (v8.2) — targeted at closing the resolution gap
    # in the KO/TKO Brier.  Brier decomposition showed RES=0.007 (barely above base-rate)
    # with era features dominating.  These conjunctive features (striker power × opponent
    # chin) directly predict KO outcomes beyond what either marginal captures.
    str_acc_a = _a("str_acc_decay")
    str_acc_b = _b("str_acc_decay")
    slpm_a_v2 = _a("slpm_decay")
    slpm_b_v2 = _b("slpm_decay")
    age_a2 = _a("age_years", 28.0)
    age_b2 = _b("age_years", 28.0)
    layoff_a2 = _a("layoff_days", 0.0) / 365.0   # in years
    layoff_b2 = _b("layoff_days", 0.0) / 365.0

    # Accuracy-weighted volume × opponent chin: more precise striker hits a fragile chin
    out["ko_matchup_a"] = slpm_a_v2 * str_acc_a * ko_loss_b
    out["ko_matchup_b"] = slpm_b_v2 * str_acc_b * ko_loss_a

    # Aging chin: old fighters who've been stopped have deteriorating durability
    out["chin_decay_threat_a"] = ko_loss_b * ((age_b2 - 30.0).clip(lower=0.0) / 10.0)
    out["chin_decay_threat_b"] = ko_loss_a * ((age_a2 - 30.0).clip(lower=0.0) / 10.0)

    # Layoff + chin damage: rusty, previously-stopped opponent is extra KO-vulnerable
    out["layoff_chin_a"] = ko_loss_b * layoff_b2
    out["layoff_chin_b"] = ko_loss_a * layoff_a2

    # Strongest discriminator from AUC analysis: specialist finish rate × fragility
    out["ko_specialist_x_chin_a"] = out["ko_specialist_idx_a"] * ko_loss_b
    out["ko_specialist_x_chin_b"] = out["ko_specialist_idx_b"] * ko_loss_a

    # 17. Expected P(fight ends in R1) — fight-level feature for r1_sig_strikes.
    # Captures early-stoppage archetypes (Topuria, Ngannou) so the count model
    # can down-weight expected strike volume in high-R1-finish-probability matchups.
    r1_ko_a = _a("r1_ko_win_rate_decay")
    r1_ko_b = _b("r1_ko_win_rate_decay")
    r1_sub_a = _a("r1_sub_win_rate_decay")
    r1_sub_b = _b("r1_sub_win_rate_decay")
    out["expected_p_r1_finish"] = 1.0 - (
        (1 - r1_ko_a - r1_sub_a).clip(0, 1)
        * (1 - r1_ko_b - r1_sub_b).clip(0, 1)
    )

    return out


def compute_opponent_quality(df: pd.DataFrame,
                              train_mask: pd.Series | None = None) -> pd.DataFrame:
    """Bucket opponent ELO into tiers and per-WC percentile using TRAIN stats only."""
    out = df.copy()
    if train_mask is None:
        train_mask = pd.Series(True, index=out.index)

    for side in ["a", "b"]:
        col = f"opp_elo_pre_{side}"
        if col not in out.columns:
            continue

        tr_vals = out.loc[train_mask, col].dropna()
        if len(tr_vals) < 4:
            out[f"opp_tier_{side}"] = 0
            out[f"opp_elo_pct_{side}"] = 0.5
            continue

        edges = np.quantile(tr_vals, [0.0, 0.25, 0.5, 0.75, 1.0])
        edges[0] = -np.inf
        edges[-1] = np.inf
        out[f"opp_tier_{side}"] = pd.cut(
            out[col], bins=edges, labels=False, include_lowest=True,
        ).astype("Int8")

        pct = pd.Series(np.nan, index=out.index, dtype=float)
        for wc, idx in out.groupby("weight_class").groups.items():
            tr_idx = idx[train_mask.reindex(idx, fill_value=False).values]
            tr_wc_vals = out.loc[tr_idx, col].dropna().values
            if len(tr_wc_vals) == 0:
                pct.loc[idx] = 0.5
                continue
            sorted_tr = np.sort(tr_wc_vals)
            pct.loc[idx] = np.searchsorted(sorted_tr, out.loc[idx, col].values) / len(sorted_tr)
        out[f"opp_elo_pct_{side}"] = pct.fillna(0.5)
    return out
