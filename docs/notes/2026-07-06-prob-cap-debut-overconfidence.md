---
type: research
date: 2026-07-06
model_version: v8.39
status: final
related_code_files:
  - src/ufc/models/winner.py
  - src/ufc/inference/predict_core.py
  - src/ufc/api/serialize.py
  - frontend/src/the picks surface
  - frontend/src/exchange.jsx
  - src/ufc/features/pre_ufc.py
  - src/ufc/features/ratings.py
related_runs_entry: "v8.32"
tags: [calibration, prob-cap, debutants, thin-data, gate-a, winner-model]
---

# The 75% prob-cap: gate-load-bearing, and the debut-opponent overconfidence it masks

## Summary

Question investigated: is the `max_prob` soft-cap (0.75, `winner.py:_soft_cap`) hurting the
model, and should it be removed? **No — removing it fails Gate A** (read-only ablation,
eval tier, held-out test 2025-01→2026-06, 769 fights). But the investigation exposed *why*
the cap is needed, and it is one specific failure mode: **overconfidence against opponents
with ≤1 prior UFC fights.** A segmented-calibration fix targeting that bucket could both
close the gap and let the ECE search earn a looser cap for veteran-vs-veteran fights.

## Evidence

### 1. Cap ablation (accuracy identical — the cap is monotone and never flips a pick)

| Variant | acc | Brier | log-loss | ECE |
|---|---|---|---|---|
| capped @ 0.75 (current) | 0.6476 | 0.2225 ✓ | 0.6367 | 0.0366 ✓ |
| uncapped (`max_prob=1.0`) | 0.6476 | 0.2305 ✗ | 0.6950 | 0.0513 ✗ |
| hard clip @ 0.75 | 0.6476 | 0.2226 | 0.6368 | 0.0355 |

The cap is re-fit every retrain by ECE grid search (0.65→1.00 step 0.05,
`winner.py:486-492`); cap=1.0 is in the grid and loses. Eval AND prod tiers (2ef9a07)
independently converged on 0.75 from different calibration windows.

### 2. The raw >90%-confidence tail is a debutant-opponent artifact

Test-set fights with raw (pre-cap) confidence >0.90: n=53, hit rate **60.4%**
(vs 96.9% mean claimed confidence). Autopsy of the 21 misses:

- **100% had opponents with ≤3 prior UFC fights; 90% ≤1.** Hits' opponents averaged 3.3.
- Repeat upsetters are elite crossovers: Yaroslav Amosov (ex-Bellator champ; beat Magny at
  raw 98.5% and Alvarez at 98.1%), Bia Mesquita ×3 (BJJ world champ), Luke Riley ×2,
  Ethyn Ewing ×2, Melissa Croden ×2.
- Misses were real dominations: KO/TKO 9, SUB 6, U-DEC 6.
- Raw confidences collapse to repeated identical values (0.981/0.985) — isotonic flat top
  segment ⇒ zero resolution in the tail.
- v9 seeded Elo / pre-UFC priors often rank the debutant HIGHER (Mesquita out-rated all
  three victims) but get outvoted by hundreds of thin/filled UFC-stat features.

### 3. The capped/served model still carries the bias — the cap only blunts it

| Opponent UFC fights | n | mean conf (served) | hit rate | gap |
|---|---|---|---|---|
| 0–1 | 152 | 0.654 | 0.579 | **+7.5pp over** |
| 2–3 | 118 | 0.632 | 0.661 | −2.9pp under |
| 4+ | 499 | 0.638 | 0.665 | −2.7pp under |

High-confidence (>0.70) slice: vs opp ≤1 → hits **55.7%** (n=61, coin flip);
vs opp ≥4 → hits **79.0%** (n=143, genuinely *under*-confident). The single global cap
averages these two populations: it strangles debut-fight disasters at the cost of
under-crediting legitimate favorites over proven opponents.

### 4. Era note

Tail hit rate 72.2% in 2025 vs 54.3% in 2026 — consistent with recent-era debutant
quality drift (Contender Series / crossover signings), worse for the stale eval model.

## Step 0 pre-check results (same day) — seg-τ plan CLOSED, no action

A segmented-calibration fix (seg-τ, plan `~/.claude/plans/2026-07-06-debut-segmented-calibration.md`)
was planned, then killed by its own Step-0 pre-check:

1. **Sub-buckets flip sign between eras.** Splitting the bucket by which side is the
   newcomer: fav-vs-newcomer gap is +2.5pp on val-2024 but +7.6pp on test-2025-26;
   newcomer-is-pick is +7.5pp (over) on val but **−11.6pp (under)** on test — when the
   model backs a newcomer in the current era it wins 75.5% vs 64% claimed. This is
   newcomer-quality **regime drift** (Contender Series pipeline), not a stable bias.
2. **Eval tier fits τ=1.000 exactly** (no val-2024 signal) → Gates A–D could certify
   safety but never effectiveness.
3. **The segment is structurally unpriceable in the current era:** test sharp-segment
   NLL at τ=1 is 0.9855 (coin-flip = 0.693); the unbounded NLL-optimum is τ=0.095,
   i.e. "call every fav-vs-newcomer fight ~53.5%".

**Resolution (Benja, 2026-07-06): abstain + inform — and it's already implemented.**
The raw-model overconfidence measured in this note never reaches served advice:

- `predict_core.py:252-273` ("Fix B" data-volume shrinkage): served P(win) is linearly
  shrunk toward 0.5 when either fighter has <4 UFC fights — a true debutant pins the
  fight to exactly 0.5 (stronger than the τ=0.095 the data asked for); opp with 1
  fight → factor 0.25 (max served conf ~62.5%). Sets `low_data=True` → `lowData` in
  the API payload (`serialize.py:406`).
- Frontend hides `lowData` picks **by default** in both lanes — DFS prop-edge
  (`the picks surface:80`) and Kalshi exchange (`exchange.jsx:213`), single toggle default
  `false` (`panels.jsx:526`) — with a named-fighter LOW DATA badge when shown.

**Gate/serving parity observation (safe direction):** Gates A and this note's numbers
evaluate `WinnerModel.predict_proba` directly — WITHOUT the serving shrinkage — so the
gate view *overstates* the served system's newcomer overconfidence. The served path is
more conservative than the yardstick that grades it. (Corollary: the History page's
eval-backtest rows show unshrunk model probs, e.g. "Magny 75% over Amosov"; the live
served prediction for that fight would have been 0.5 — Amosov had 0 UFC fights.)

Seg-τ / Elo-blend repricing stays parked unless the forward live record shows the
*served* (shrunk) path still overconfident on this segment.

## Related

- [[2026-07-06-v839-sentinel-state-and-fill-median-leak]] — same-day eval baseline. (Ablation acc
  0.6476 vs RUNS.md v8.39 Gate A 0.6606: different eval path — this note's script scores the raw
  test split via `symmetrize`+mean, not `04_backtest.py`. The ablation is internally consistent:
  same model, same rows, only the cap toggled — the with/without deltas are what matter.)
- RUNS.md v8.32 (temporal-OOF calibration; prod max_prob context), v9 (prob-clip introduction, seeded Elo, pre-UFC priors)
