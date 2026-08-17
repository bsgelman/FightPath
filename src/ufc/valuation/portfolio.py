"""Portfolio-level joint hit-rate using Monte Carlo simulation samples."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from ufc.valuation.lines import Line
from ufc.valuation.edge import Edge


@dataclass
class PortfolioResult:
    n_legs: int
    individual_probs: list[float]
    naive_joint_prob: float    # product of marginals
    mc_joint_prob: float       # from joint samples
    correlation_adjustment: float  # mc - naive
    edges: list[Edge]


def evaluate_portfolio(
    lines: list[Line],
    edges: list[Edge],
    sim_samples: dict,  # output of simulator.simulate()
    n_samples: int = 50000,
) -> PortfolioResult:
    """Compute joint hit probability using Monte Carlo samples.

    sim_samples keys expected:
      winner_a      : bool array (did fighter A win?)
      sig_str_a, sig_str_b : int arrays
      td_a, td_b    : int arrays
      duration_sec  : float array
      method        : str array ('KO/TKO', 'SUB', 'DEC')
    """
    if not lines:
        return PortfolioResult(0, [], 1.0, 1.0, 0.0, [])

    individual_probs = [e.model_prob for e in edges]
    naive_joint = float(np.prod(individual_probs))

    # Evaluate each leg on MC samples
    hit_matrix = np.ones(n_samples, dtype=bool)

    for line, edge in zip(lines, edges):
        hits = _evaluate_line_on_samples(line, sim_samples, n_samples, model_prob=edge.model_prob)
        hit_matrix &= hits

    mc_joint = float(hit_matrix.mean())

    return PortfolioResult(
        n_legs=len(lines),
        individual_probs=individual_probs,
        naive_joint_prob=naive_joint,
        mc_joint_prob=mc_joint,
        correlation_adjustment=mc_joint - naive_joint,
        edges=edges,
    )


def _evaluate_line_on_samples(line: Line, samples: dict, n: int, model_prob: float = 0.5) -> np.ndarray:
    """Return boolean array of shape (n,) indicating if this leg hits."""
    m = line.market
    side = line.side
    lv = line.line_value
    fid = line.fighter_id

    # Determine if fighter is _a or _b perspective
    is_a = fid == samples.get("fighter_id_a")

    if m == "winner":
        if is_a:
            return samples.get("winner_a", np.zeros(n, bool)).astype(bool)
        else:
            return ~samples.get("winner_a", np.ones(n, bool)).astype(bool)

    if m == "sig_strikes":
        key = "sig_str_a" if is_a else "sig_str_b"
        vals = samples.get(key, np.zeros(n))
    elif m == "takedowns":
        key = "td_a" if is_a else "td_b"
        vals = samples.get(key, np.zeros(n))
    elif m in ("duration_sec", "duration", "rounds"):
        vals = samples.get("duration_sec", np.zeros(n))
    elif m == "method":
        method_arr = samples.get("method", np.array(["DEC"] * n))
        from ufc.models.method import METHOD_MAP
        return method_arr == side.upper()
    else:
        # Unknown market: independent Bernoulli at model_prob (no joint-MC signal).
        # Documented approximation — count props not in the joint simulator.
        return np.random.default_rng(abs(hash(m)) & 0xFFFFFFFF).random(n) < model_prob

    if side == "over":
        return vals > lv
    else:  # under
        return vals < lv
