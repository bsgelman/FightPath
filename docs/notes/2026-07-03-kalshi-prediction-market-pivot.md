---
type: decision
date: 2026-07-03
model_version: n/a
status: active
related_code_files: [src/ufc/ingest/market_lines.py, src/ufc/valuation/market_edge.py, src/ufc/api/app.py, src/ufc/api/service.py, scripts/07b_log_prop_lines.py, scripts/08b_grade_props.py, frontend/src/exchange.jsx, configs/market_advice.yaml]
related_runs_entry: ""
tags: [kalshi, exchange, market-edge, clv, ledger]
---

# Kalshi prediction-market pivot (winner + method advising)

## Summary

Extended the model from DFS-prop advising to pricing Kalshi prediction-market quotes
(winner `KXUFCFIGHT`, method `KXUFCMOV`), ledger-first with a PAPER gate: advice stays
paper-only until ≥50 graded rows with positive CLV (`configs/market_advice.yaml`).
Gate-neutral — no model, feature, or training change; all pricing reads existing
prediction artifacts. Commits `de2579c`, `4543a82`, `469eb19` (2026-07-03).

## Key design decisions

- **Joint method probability via MC samples.** A method quote needs P(fighter wins AND
  by method), which `method_probs` cannot give (method-only, no fighter attribution).
  `market_edge.model_prob_for_quote` reads it off Monte Carlo `sim_samples["winner_a"]`
  × `sim_samples["method"]` instead. Shared by `07b_log_prop_lines.py` and the API
  endpoint, so ledger and UI can never disagree.
- **`side="over"` for Kalshi rows** (not "yes"): deliberately reuses the existing CLV
  sign convention (`close − open` favorable) for a price/probability quote without
  touching `_clv_delta_fav`.
- **Structured `custom_strike` parsing** (`Method`, `Participant`) beat the planned
  title-regex approach — no fragile string parsing.
- **Fees are the edge killer**: `kalshi_taker_fee` / `effective_cost_taker` /
  `effective_cost_maker` + liquidity tiers (DEEP/OK/THIN) live in `market_edge.py`.
- **DFS lane stays pure**: `service._build_exchange_summary` filters
  `platform != "kalshi"` out of the DFS prop-edge lane.

## Evidence

TDD throughout (tests 179 → 185; `tests/test_market_lines.py`,
`tests/test_market_edge.py`). Live-verified end-to-end on the July 11 card: 22 winner
quotes priced; `/api/market-lines/{card_id}` 60s cache (33ms hit vs 50s cold);
Playwright UI pass with zero console errors and no 375px overflow.

Known friction (pre-existing, shared name matcher): `_token_surname_match` misses
hyphenated-vs-spaced surnames ("Saint-Denis" vs "Saint Denis") and ring names
("Bobby Green" vs "King Green") — logged to `unresolved_names.txt`, never blocks the lane.

## Related

- Follow-up: [2026-07-04 Kalshi market expansion](2026-07-04-kalshi-market-expansion.md)
- Plan of record: `~/.claude/plans` Kalshi pivot plan (6 phases; Phase 6 Polymarket deferred)
- Not in RUNS.md: serving/valuation layer only — Gates A–D untouched.
