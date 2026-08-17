"""Common-opponent transitivity feature (v2).

For each fight A vs B at event_rank R:
  1. Find all fighters C where A fought C (event_rank < R)
     AND B fought C (event_rank < R).
  2. Compute method-weighted margin: A's result vs C minus B's result vs C.
       Finish win/loss = ±1.0,  Decision = ±0.6,  Split-dec = ±0.3
     Recency-weighted: fights closer to rank R count more (halflife 50 ranks).
  3. Aggregate: common_opp_advantage = weighted mean margin.
  4. Optional 2-step chains: if A beat X and X beat B (not common, but path exists),
     add a damped margin (0.5 × finish_weight) capped at 20 paths.

v1 (original): binary margins, no weights, no 2-step chains.
v2 changes: recency + method weighting, 2-step chains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Method weight for margin calculation
_METHOD_WEIGHT: dict[str, float] = {
    "KO/TKO": 1.0,
    "SUB":    1.0,
    "U-DEC":  0.6,
    "S-DEC":  0.3,
    "M-DEC":  0.4,
}
_DEFAULT_WEIGHT = 0.6  # unknown method

_RECENCY_HALFLIFE = 50.0   # in event_rank units
_TWO_STEP_DAMPING = 0.5    # chain confidence damping
_TWO_STEP_CAP     = 20     # max 2-step paths to avoid O(n^2) explosion


def _method_weight(method: str) -> float:
    if not isinstance(method, str):
        return _DEFAULT_WEIGHT
    return _METHOD_WEIGHT.get(method.strip(), _DEFAULT_WEIGHT)


def compute_transitivity(ledger: pd.DataFrame) -> pd.DataFrame:
    """Add common_opp_advantage and n_common_opps columns to the ledger (v2).

    Vectorized per-fighter history with recency + method-weighted margins
    and optional 2-step transitivity chains.
    """
    df = ledger.copy()

    # Build per-fighter sorted history (rank, opponent, won, method)
    valid = df[df["won"].notna()].sort_values(["fighter_id", "event_rank"])
    history: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for fid, grp in valid.groupby("fighter_id", sort=False):
        history[fid] = (
            grp["event_rank"].values.astype(np.int64),
            grp["opponent_id"].astype(str).values,
            grp["won"].values.astype(float),
            grp["method"].fillna("").tolist(),
        )

    def _prior_results(fighter_id: str, rank_cutoff: int) -> dict[str, tuple[float, float, str]]:
        """Return {opponent_id: (won, rank, method)} for all prior fights."""
        rec = history.get(fighter_id)
        if rec is None:
            return {}
        ranks, opps, wons, methods = rec
        i = int(np.searchsorted(ranks, rank_cutoff, side="left"))
        # Latest result vs each opponent wins (overwrite semantics)
        out: dict[str, tuple[float, float, str]] = {}
        for r, opp, w, m in zip(ranks[:i], opps[:i], wons[:i], methods[:i]):
            out[opp] = (w, float(r), m)
        return out

    common_adv = np.full(len(df), np.nan)
    n_common = np.zeros(len(df), dtype=int)

    for i, row in enumerate(df.itertuples(index=False)):
        fid = str(row.fighter_id)
        oid = str(row.opponent_id)
        r = int(row.event_rank)
        a_prior = _prior_results(fid, r)
        b_prior = _prior_results(oid, r)

        common = (set(a_prior) & set(b_prior)) - {fid, oid}

        weighted_margins: list[float] = []
        total_weight: float = 0.0

        if common:
            for c in common:
                wa, ra_rank, ma = a_prior[c]
                wb, rb_rank, mb = b_prior[c]
                # Method weight: use winner's method (the decisive result)
                mw_a = _method_weight(ma) if wa == 1 else _method_weight(ma) * 0.8
                mw_b = _method_weight(mb) if wb == 1 else _method_weight(mb) * 0.8
                mw = (mw_a + mw_b) / 2.0
                # Recency weight: based on how long ago the common fight happened
                recency_a = np.exp(-np.log(2) * (r - ra_rank) / _RECENCY_HALFLIFE)
                recency_b = np.exp(-np.log(2) * (r - rb_rank) / _RECENCY_HALFLIFE)
                rec_w = (recency_a + recency_b) / 2.0
                w_total = mw * rec_w
                margin = (wa - wb)  # +1 A beat C while B lost; -1 vice versa; 0 same
                weighted_margins.append(margin * w_total)
                total_weight += w_total

        # 2-step chains: A beat X, X beat B (or B beat Y, Y beat A)
        n_two_step = 0
        if len(common) < _TWO_STEP_CAP:  # skip expensive path if lots of common opps
            # A's wins -> check if B lost to any of A's victims
            a_wins = {opp for opp, (w, _, _) in a_prior.items() if w == 1}
            b_losses = {opp for opp, (w, _, _) in b_prior.items() if w == 0}
            chain_fwd = (a_wins & b_losses) - common - {fid, oid}
            for x in list(chain_fwd)[:_TWO_STEP_CAP]:
                _, _, mx_a = a_prior[x]
                _, _, mx_b = b_prior[x]
                mw = _method_weight(mx_a) * _method_weight(mx_b)
                weighted_margins.append(1.0 * mw * _TWO_STEP_DAMPING)
                total_weight += mw * _TWO_STEP_DAMPING
                n_two_step += 1

            # B's wins -> check if A lost to any of B's victims
            b_wins = {opp for opp, (w, _, _) in b_prior.items() if w == 1}
            a_losses = {opp for opp, (w, _, _) in a_prior.items() if w == 0}
            chain_bwd = (b_wins & a_losses) - common - {fid, oid}
            for x in list(chain_bwd)[:_TWO_STEP_CAP]:
                _, _, mx_a = a_prior[x]
                _, _, mx_b = b_prior[x]
                mw = _method_weight(mx_a) * _method_weight(mx_b)
                weighted_margins.append(-1.0 * mw * _TWO_STEP_DAMPING)
                total_weight += mw * _TWO_STEP_DAMPING
                n_two_step += 1

        if weighted_margins and total_weight > 0:
            common_adv[i] = sum(weighted_margins) / total_weight
        n_common[i] = len(common) + n_two_step

    df["common_opp_advantage"] = np.where(np.isnan(common_adv), 0.0, common_adv)
    df["n_common_opps"] = n_common
    return df
