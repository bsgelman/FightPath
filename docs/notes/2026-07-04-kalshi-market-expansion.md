---
type: research
date: 2026-07-04
model_version: n/a
status: active
related_code_files: [src/ufc/ingest/market_lines.py, src/ufc/valuation/market_edge.py, src/ufc/valuation/kalshi_grading.py, src/ufc/io/prop_prediction_log.py, scripts/07b_log_prop_lines.py, scripts/08b_grade_props.py, frontend/src/exchange.jsx]
related_runs_entry: ""
tags: [kalshi, exchange, grading, market-kinds]
---

# Kalshi market expansion: 2 → 6 market families

## Summary

Extended the Kalshi lane from winner+method to six market families by adding
`KXUFCDISTANCE` (goes-distance), `KXUFCMOF` (fight-level method), `KXUFCROUNDS`
(ends-before-round-r), and `KXUFCVICROUND` (fighter×round win + OTHER bucket). All
priced from existing artifacts (`method_probs`, `display_dur_cdf`, `sim_samples`) —
zero retrain, gate-neutral (diff verified: no touches to models/features/training/eval).
Commits `28f29e1`, `e2fe94f`, `ad9b44a` (2026-07-04).

## Extension pattern (for a 7th kind)

The pipeline is fully `market_kind`-string-driven; a new fight-level or
per-fighter-round kind needs exactly four touches:
1. Parser in `market_lines.py` — prefer structured `custom_strike` fields
   (`Method`, `Participant`, `Round`) over ticker/title regex.
2. Pricing branch in `market_edge.model_prob_for_quote`.
3. Predicate entry in `kalshi_grading.py::kind_spec()`.
4. `kindMeta()` entry in `frontend/src/exchange.jsx`.

## Latent bugs found and fixed (pre-existing in the winner/method lane)

1. **DQ/draw grading fold** — `08b_grade_props.py` graded Kalshi method markets in the
   model's 3-class space (`METHOD_MAP` folds DQ into DEC), so a DQ win scored as
   `method_dec`; draws left winner/method rows pending forever. Fix: `kalshi_grading.py`
   predicates read `raw_method` directly. First post-deploy `08b --regrade`
   retro-corrects the 44 existing rows (no realized-hit impact — none were draws/DQs).
2. **Cornerless key collision** — `_kalshi_key` omitted `corner`, so two fighters
   quoted at the same ask (common for `win_in_r{N}`) silently collided and one row was
   dropped. Fix: `kalshi_key()` in `prop_prediction_log.py` now includes corner, with
   `legacy_kalshi_key()` dedupe guard so pending rows aren't re-logged.

## Evidence / verification method worth reusing

`/api/market-lines/{card}` is slow cold (~8s/fight pre-cache) — for UI verification,
mocked `fetch_all_markets` in a throwaway bootstrap, ran a second uvicorn on a spare
port with a second Vite dev server via `VITE_API_URL`, letting the REAL
prediction/pricing pipeline run against synthetic quotes covering all 6 kinds.
Playwright-confirmed fight grouping, DEC-stack collapse, round-strip fills, no 375px
overflow. UI reused the existing `--m-rnd` violet accent for round-indexed kinds
rather than minting a near-duplicate token.

## Related

- Builds on: [2026-07-03 Kalshi pivot](2026-07-03-kalshi-prediction-market-pivot.md)
- The slow cold-load found here motivated:
  [2026-07-04 prediction cache](2026-07-04-service-predict-cache.md)
