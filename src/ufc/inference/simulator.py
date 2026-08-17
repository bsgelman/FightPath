"""Monte Carlo joint simulator for correlated prop evaluation."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml

from ufc import SEED
from ufc.io import paths

logger = logging.getLogger(__name__)


def _cfg():
    with open(paths.root() / "configs" / "model_props.yaml") as f:
        return yaml.safe_load(f)


def simulate(
    winner_prob: float,
    method_probs: dict[str, float],
    duration_cdf,
    sig_str_cdf_a,
    sig_str_cdf_b,
    td_cdf_a,
    td_cdf_b,
    scheduled_rounds: int = 3,
    n_samples: int = 50000,
    fighter_id_a: str = "a",
    fighter_id_b: str = "b",
    seed: int = SEED,
    duration_cdfs_by_method: dict | None = None,
    sig_str_method_adj: dict[str, float] | None = None,
    td_method_adj: dict[str, float] | None = None,
) -> dict:
    """Joint Monte Carlo simulation.

    Returns dict of arrays, each of length n_samples.
    """
    rng = np.random.default_rng(seed)
    cfg = _cfg()

    # ── 1. Sample method first (needed for method-conditional winner) ────
    # (moved method sampling before winner so we can condition winner on method)
    p_ko = method_probs.get("KO/TKO", 0.33)
    p_sub = method_probs.get("SUB", 0.17)
    p_dec = max(0, 1.0 - p_ko - p_sub)
    # Re-normalize
    total = p_ko + p_sub + p_dec
    p_ko /= total; p_sub /= total; p_dec /= total

    method_draws = rng.choice(
        ["KO/TKO", "SUB", "DEC"],
        size=n_samples,
        p=[p_ko, p_sub, p_dec],
    )
    is_finish = method_draws != "DEC"

    # ── 2. Sample winner (method-conditional) ────────────────────────────
    # Method-conditional winner adjustment (empirical priors).
    # KO/SUB slightly strengthen the lean toward the favored fighter; DEC weakens it.
    # Invariant: Σ_m p_m * w_m == winner_prob — the per-method shift below must
    # never change the simulated winner marginal (unrescaled: input 0.65 drifted
    # to sim mean 0.6174). Rescale multiplicatively, re-clip, and rescale once
    # more in case clipping perturbed the mix.
    method_shift = {"KO/TKO": 0.05, "SUB": 0.05, "DEC": -0.10}
    direction = float(np.sign(winner_prob - 0.5))
    p_method = {"KO/TKO": p_ko, "SUB": p_sub, "DEC": p_dec}
    w_method = {
        m: float(np.clip(winner_prob + direction * s, 0.02, 0.98))
        for m, s in method_shift.items()
    }
    for _ in range(2):
        mix = sum(p_method[m] * w_method[m] for m in w_method)
        if mix <= 0 or abs(mix - winner_prob) < 1e-9:
            break
        scale = winner_prob / mix
        w_method = {m: float(np.clip(v * scale, 0.02, 0.98)) for m, v in w_method.items()}
    else:
        # Loop exhausted without converging — happens only when the 0.98 clip
        # binds on 2+ method arms at once (winner_prob near the extremes),
        # which the fixed 2-iteration rescale can't fully correct. Degraded
        # marginal accuracy, not a crash; log so it's visible if it starts
        # firing (it shouldn't at today's ~0.75 practical winner_prob ceiling).
        residual = sum(p_method[m] * w_method[m] for m in w_method) - winner_prob
        if abs(residual) > 1e-3:
            logger.warning(
                "simulate(): winner-marginal rescale did not converge "
                "(winner_prob=%.4f, residual=%.4f)", winner_prob, residual,
            )

    winner_prob_per_sample = np.array([w_method[m] for m in method_draws])
    winner_a = rng.random(n_samples) < winner_prob_per_sample

    # ── 3. Sample duration ────────────────────────────────────────────────
    max_sec = scheduled_rounds * 300
    duration_sec = np.full(n_samples, float(max_sec))

    u_dur = rng.random(n_samples)
    if duration_cdfs_by_method is not None:
        # Method-conditional duration: KO/SUB draws use their own CDF; DEC = max_sec.
        for m in ["KO/TKO", "SUB"]:
            cdf_m = duration_cdfs_by_method.get(m)
            if cdf_m is None:
                continue
            m_mask = method_draws == m
            if not m_mask.any():
                continue
            grid_t_m = np.linspace(1.0, float(max_sec), 512)
            grid_cdf_m = np.array([cdf_m.cdf(t) for t in grid_t_m])
            grid_cdf_m = np.clip(np.maximum.accumulate(grid_cdf_m), 0.0, 1.0)
            duration_sec[m_mask] = np.interp(u_dur[m_mask], grid_cdf_m, grid_t_m)
    elif duration_cdf is not None:
        grid_t = np.linspace(1.0, float(max_sec), 512)
        grid_cdf = np.array([duration_cdf.cdf(t) for t in grid_t])
        sampled = np.interp(u_dur, grid_cdf, grid_t)
        duration_sec = np.where(is_finish, sampled, float(max_sec))

    # ── 4. Sample per-fighter counts ─────────────────────────────────────
    fight_fraction = duration_sec / max_sec

    def _sample_counts(cdf, fighter_label) -> np.ndarray:
        if cdf is None:
            return np.zeros(n_samples, dtype=int)
        u = rng.random(n_samples)

        # RateXDurationCDF already integrates duration into its MC samples —
        # applying fight_fraction scaling would double-discount duration.
        from ufc.models.props_count import RateXDurationCDF  # noqa: PLC0415
        duration_already_encoded = isinstance(cdf, RateXDurationCDF)

        if duration_already_encoded:
            # Inverse-CDF via empirical sorted samples (binary search)
            vals = np.interp(u, np.linspace(0.0, 1.0, len(cdf._samples)), cdf._samples)
        elif hasattr(cdf, "qv") and hasattr(cdf, "qs"):
            vals = np.interp(u, cdf.qs, cdf.qv)
        else:
            # Fallback for HurdlePropCDF: build a small grid
            grid_x = np.linspace(0.0, 300.0, 512)
            grid_cdf = np.array([cdf.cdf(x) for x in grid_x])
            vals = np.interp(u, grid_cdf, grid_x)

        if not duration_already_encoded:
            # Sub-linear scaling for short fights (preserves original heuristic)
            vals = vals * (fight_fraction ** 0.85)

        return np.round(vals).clip(min=0).astype(int)

    sig_str_a = _sample_counts(sig_str_cdf_a, "a")
    sig_str_b = _sample_counts(sig_str_cdf_b, "b")
    td_a = _sample_counts(td_cdf_a, "a")
    td_b = _sample_counts(td_cdf_b, "b")

    # Method-conditional count adjustment: use learned log-rate residuals when available,
    # fall back to log of the original v5 heuristic multipliers otherwise.
    _FALLBACK_SS_LOG = {"KO/TKO": np.log(0.85), "SUB": np.log(0.75), "DEC": np.log(1.05)}
    _FALLBACK_TD_LOG = {"KO/TKO": np.log(0.90), "SUB": np.log(1.10), "DEC": np.log(1.00)}
    _ss_log = sig_str_method_adj if sig_str_method_adj is not None else _FALLBACK_SS_LOG
    _td_log = td_method_adj if td_method_adj is not None else _FALLBACK_TD_LOG
    ss_mult = np.exp(np.array([_ss_log.get(m, 0.0) for m in method_draws]))
    td_mult = np.exp(np.array([_td_log.get(m, 0.0) for m in method_draws]))
    sig_str_a = (sig_str_a * ss_mult).round().astype(int)
    sig_str_b = (sig_str_b * ss_mult).round().astype(int)
    td_a = (td_a * td_mult).round().astype(int)
    td_b = (td_b * td_mult).round().astype(int)

    return {
        "winner_a": winner_a,
        "method": method_draws,
        "duration_sec": duration_sec,
        "sig_str_a": sig_str_a,
        "sig_str_b": sig_str_b,
        "td_a": td_a,
        "td_b": td_b,
        "fight_fraction": fight_fraction,
        "fighter_id_a": fighter_id_a,
        "fighter_id_b": fighter_id_b,
        "n_samples": n_samples,
    }
