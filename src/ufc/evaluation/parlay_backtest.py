"""Walk-forward parlay ROI simulation — Step 11.

Replaces the structurally-inflated roi_vs_line single-leg metric with an
honest event-level walk-forward that forms real 2-pick and 3-pick parlays.

For each event in the test set:
1. Select candidate fights where model_prob > implied_per_leg + threshold.
2. Form all 2-pick combinations (resp. 3-pick) from the event candidates.
3. A parlay hits iff all legs win; pays `mult * stake` (net: (mult-1)*stake), else -stake.
4. Aggregate n_parlays, hit_rate, ROI across all events.

This is the correct way to measure whether the model's edge survives the
parlay compounding structure — not a per-leg multiplied approximation.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ufc.valuation.payouts import implied_prob_per_leg, get_payout_multiplier


def walk_forward_parlay(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    y_true: np.ndarray,
    payout_type_2pick: str = "powerplay_power_2pick",
    payout_type_3pick: str = "powerplay_power_3pick",
    threshold: float = 0.05,
    stake: float = 1.0,
) -> dict:
    """Walk-forward parlay simulation grouped by event.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test-set rows with an ``event_date`` column.  Index must align with
        ``probs`` and ``y_true`` (reset index before calling).
    probs : np.ndarray
        Model P(A wins) for each row in test_df (symmetric-averaged).
    y_true : np.ndarray
        1.0 / 0.0 indicator (A wins) for each row.
    payout_type_2pick / payout_type_3pick : str
        Payout keys from valuation.yaml (e.g. 'powerplay_power_2pick').
    threshold : float
        Edge above per-leg implied breakeven required to include a leg.
        0.05 = bet when model_prob > implied + 5%.
    stake : float
        Stake per parlay in arbitrary units.

    Returns
    -------
    dict with keys '2pick' and '3pick', each containing:
      n_parlays, hits, hit_rate, profit, stake_total, roi,
      implied (per-leg breakeven), multiplier, payout_type.
    Also includes 'event_summary': list of per-event dicts.
    """
    implied_2 = implied_prob_per_leg(payout_type_2pick, n_legs=2)
    implied_3 = implied_prob_per_leg(payout_type_3pick, n_legs=3)
    mult_2 = get_payout_multiplier(payout_type_2pick)   # 3.0
    mult_3 = get_payout_multiplier(payout_type_3pick)   # 5.0

    df = test_df.reset_index(drop=True).copy()
    probs_arr = np.asarray(probs, dtype=float)
    y_arr = np.asarray(y_true, dtype=float)

    acc_2 = {"n_parlays": 0, "hits": 0, "profit": 0.0, "stake_total": 0.0}
    acc_3 = {"n_parlays": 0, "hits": 0, "profit": 0.0, "stake_total": 0.0}
    event_summary: list[dict] = []

    events = pd.to_datetime(df["event_date"]).dt.date
    for event_date in sorted(events.unique()):
        mask = (events == event_date).values
        ep = probs_arr[mask]
        ey = y_arr[mask]

        # Candidates for each pick-size (need edge above that size's implied)
        cands_2 = np.where(ep > (implied_2 + threshold))[0]
        cands_3 = np.where(ep > (implied_3 + threshold))[0]

        ev_2: dict = {"n_parlays": 0, "hits": 0, "candidates": int(len(cands_2))}
        ev_3: dict = {"n_parlays": 0, "hits": 0, "candidates": int(len(cands_3))}

        # 2-pick parlays: all pairs from candidates
        for (i, j) in combinations(cands_2, 2):
            all_win = bool(ey[i]) and bool(ey[j])
            acc_2["n_parlays"] += 1
            acc_2["stake_total"] += stake
            ev_2["n_parlays"] += 1
            if all_win:
                acc_2["hits"] += 1
                acc_2["profit"] += stake * (mult_2 - 1.0)
                ev_2["hits"] += 1
            else:
                acc_2["profit"] -= stake

        # 3-pick parlays: all triples from candidates
        for (i, j, k) in combinations(cands_3, 3):
            all_win = bool(ey[i]) and bool(ey[j]) and bool(ey[k])
            acc_3["n_parlays"] += 1
            acc_3["stake_total"] += stake
            ev_3["n_parlays"] += 1
            if all_win:
                acc_3["hits"] += 1
                acc_3["profit"] += stake * (mult_3 - 1.0)
                ev_3["hits"] += 1
            else:
                acc_3["profit"] -= stake

        event_summary.append({
            "event_date": str(event_date),
            "n_fights": int(mask.sum()),
            "2pick": ev_2,
            "3pick": ev_3,
        })

    def _finalize(acc: dict, implied: float, mult: float, payout_type: str) -> dict:
        n = acc["n_parlays"]
        hits = acc["hits"]
        st = acc["stake_total"]
        return {
            "n_parlays": n,
            "hits": hits,
            "hit_rate": float(hits / n) if n > 0 else 0.0,
            "profit": float(acc["profit"]),
            "stake_total": float(st),
            "roi": float(acc["profit"] / st) if st > 0 else 0.0,
            "implied": float(implied),
            "multiplier": float(mult),
            "payout_type": payout_type,
        }

    return {
        "2pick": _finalize(acc_2, implied_2, mult_2, payout_type_2pick),
        "3pick": _finalize(acc_3, implied_3, mult_3, payout_type_3pick),
        "event_summary": event_summary,
        "threshold": threshold,
    }
