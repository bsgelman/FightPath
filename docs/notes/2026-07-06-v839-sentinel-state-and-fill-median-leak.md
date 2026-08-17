---
type: research
date: 2026-07-06
model_version: v8.39
status: final
related_code_files:
  - src/ufc/features/assemble.py
  - src/ufc/features/ratings.py
  - src/ufc/features/finishes.py
  - src/ufc/inference/matchup.py
  - src/ufc/inference/predict_core.py
  - src/ufc/api/app.py
  - frontend/src/exchange.jsx
related_runs_entry: "v8.39"
tags: [serving, leakage, calibration, gate-b, kalshi, trust-tiers]
---

# v8.39 — Sentinel pre_fight_state, the fill-median leak, and the Garbrandt case

## Summary

Investigating a 39-point model/market divergence (Garbrandt 63.6% vs Kalshi 25¢, UFC 329)
exposed that `pre_fight_state` lagged one fight behind — it was each fighter's *pre-fight*
row of their most recent bout, so the bout's own result never reached serving. Fixing it
(sentinel rows) exposed a second, worse bug: `fill_sparse_history`'s "train-fold" medians
were computed over label-misaligned rows containing **1,926 post-train_end fights** — a
temporal leak that had been mislabeled as "structural Gate B drift" since v8.32.

Outcomes:
- **Gate B 7/11 → 10/11.** duration/body/combo/ctrl/takedowns were never structural; the
  leak was suppressing them. Only `r1_sig_strikes` remains (accepted).
- **Gate A improved** on the same eval models: acc 0.6567 → 0.6606, ECE 0.046 → 0.037.
- **P(Garbrandt) 0.636 → 0.427** once his March no-show (outstruck 59-28 by a 1-3
  opponent, "won" via 2-pt groin-strike deduction) entered the state. The market was right.
- Serve-time pairwise rating parity fixed (`opp_elo_pre`/`elo_diff`/`glicko_z`/`ts_z` were
  vs each fighter's *last* opponent, now vs the actual pairing).
- `⚠ CHECK TAPE` divergence chip (≥20pp model/market gap) in the exchange view.
- Prod retrained (2ef9a07) on leak-fixed features; trust tiers: takedowns + ctrl_time
  WATCH→TRUST (scorecard 2026-07-06).

## Evidence

- **One-fight lag:** `assemble.py` built `pre_fight_state` via `groupby(fighter).last()`
  over causal per-fight rows — by construction the last fight's result is excluded. Fix:
  `append_sentinel_rows` appends a no-outcome row per fighter (last fight +1d, event_rank
  max+1, fight facts nulled dtype-preservingly); causal windows then cover the full history.
  **Sentinel-neutrality proof:** sentinel-on vs sentinel-off builds produce byte-identical
  `features_winner`/`features_props` (0 of 688 columns differ).
- **The leak:** `context.compute_era_baselines` returns the ledger sorted by
  (weight_class, event_date); the rating passes reset the index via `pd.merge`; the
  label-aligned boolean `train_mask` in `fill_sparse_history` then selected arbitrary rows —
  measured `max(event_date)` in the "train fold" = 2026-06-27 with 1,926 rows past
  train_end (2023-12-31). Fix: re-sort chronologically + `reset_index` after era baselines.
  Detected only because the plan's byte-identity gate refused to pass — the sentinel concat's
  `ignore_index=True` had *accidentally* fixed the alignment, changing fill values.
- **Rating parity:** training computes `opp_elo_pre` etc. against the actual opponent;
  serving copied stale state values from the last historical fight. Same bug family as
  v8.11/v8.13 production/gate parity fixes.
- Live verification: HF Space serves identical numbers to local (Garbrandt 0.4274 eval-tier
  pre-retrain; 0.473 after prod retrain), `nFights` counts corrected (the `_n_fights` +1
  correction encoded the old state semantics — caught in final whole-branch review).
- Leakage audit (read-only agent): clean; empirically re-derived — 276 feature columns
  diffed on the real ledger with/without sentinels, 0 mismatches on real rows.

## Lessons

1. **"Structural drift" conclusions need a leak audit first.** v8.38's ablation "proved"
   duration drift was structural; the real cause was a masked index-misalignment leak.
2. **Byte-identity gates earn their keep** — the mismatch that "failed" the plan was the
   discovery mechanism for the leak.
3. **Big model/market divergence = check the fighter's last-fight stat lines**, not just
   W/L. A "W" hid a domination loss. Now partially automated by the divergence chip.
4. Boolean-mask `.loc` after any merge/sort is a silent foot-gun: order + labels must be
   kept in lockstep before masks are computed.

## Related

- RUNS.md entry: v8.39
- Plan: `docs/superpowers/plans/2026-07-05-pre-fight-state-sentinel.md`
- Reports: `outputs/reports/backtest_2026-07-05.md`,
  `outputs/reports/prop_edge_scorecard_2026-07-06.md`,
  `outputs/reports/prod_calibration_2026-07-05.md`
- [[2026-07-03-kalshi-prediction-market-pivot]] — the advising lane this protects
- [[2026-07-05-prop-ledger-clv-pipeline-map]] — where graded outcomes accumulate
