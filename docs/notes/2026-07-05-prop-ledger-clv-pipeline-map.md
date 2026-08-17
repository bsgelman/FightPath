---
type: reference
date: 2026-07-05
model_version: n/a
status: active
related_code_files: [scripts/07b_log_prop_lines.py, scripts/07c_capture_closing_lines.py, scripts/08b_grade_props.py, scripts/08c_report_prop_ledger.py, scripts/05c_evaluate_prop_edge.py, scripts/ledger_log.ps1, scripts/closing_log.ps1, configs/market_advice.yaml, configs/prop_trust.yaml]
related_runs_entry: "v8.34"
tags: [ledger, clv, ops, cron, kalshi]
---

# Prop ledger / CLV pipeline — where to look

## Summary

Navigation map for the forward-ledger system (DFS props + Kalshi exchange lanes).
Introduced v8.34 (RUNS.md); extended for Kalshi in the
[2026-07-03 pivot](2026-07-03-kalshi-prediction-market-pivot.md).

## Weekly cycle

| When | What | Script / automation |
|---|---|---|
| Thursday 20:00 local | Log open prop + Kalshi lines vs model preds | `ledger_log.ps1` → `07b_log_prop_lines.py` (GH fallback: `.github/workflows/prop_ledger.yml`, cron `0 1 * * 5` UTC) |
| Saturday (pre-card) | Capture closing lines for CLV | `closing_log.ps1` → `07c_capture_closing_lines.py` (GH: `closing_lines.yml`) |
| Monday refresh | Grade results (all markets, by fighter NAME not corner; NC voids; `--regrade` re-runs) | `refresh_history.ps1` folds in `08b_grade_props.py` |
| On demand | Ledger report: hit rates, CLV beat-close%, Exchange section | `08c_report_prop_ledger.py` |
| Advisory (eval-tier) | Prop-edge scorecard + trust tiers | `05c_evaluate_prop_edge.py` → `configs/prop_trust.yaml` |

## Key invariants

- CLV match key = `(matchup_key, market, corner, platform)` — never line_value.
- Kalshi rows use `side="over"` so the shared CLV sign convention holds; key includes
  `corner` (`kalshi_key()` in `src/ufc/io/prop_prediction_log.py`).
- Kalshi advice is gated PAPER → LIVE by `configs/market_advice.yaml`
  (≥50 graded rows + positive CLV, per market kind).
- Ledger data lives in `data/external/lines/` (DFS) and `data/external/market_lines/`
  (Kalshi); crons commit with `[skip ci]` and push with `--autostash`.

## Related

- [2026-07-03 Kalshi pivot](2026-07-03-kalshi-prediction-market-pivot.md)
- [2026-07-04 market expansion](2026-07-04-kalshi-market-expansion.md)
- RUNS.md entries: v8.34 (scorecard + ledger), v8.33 (History live record)
