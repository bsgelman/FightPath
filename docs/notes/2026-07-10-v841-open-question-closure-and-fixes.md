---
type: research
date: 2026-07-10
model_version: eval 1821503 / prod e35f901
status: final
related_code_files:
  - scripts/_prod_calibration_report.py
  - src/ufc/inference/matchup.py
  - src/ufc/models/winner.py
  - src/ufc/training/feature_pruning.py
  - outputs/models/.gitignore
related_runs_entry: "RUNS.md 2026-07-10 B4 (REVERTED) + open-anomaly closure"
tags: [investigation, calibration, pca-parity, gate-b, survivorship, dead-features, b3, clv]
---

# 2026-07-10 v8.41: open-question closure + targeted fixes

## Summary

Diagnosis-first session closing the six v8.40 open questions, followed by surgical fixes.
Two standing hypotheses refuted, one live serving bug found and shipped same-day, one
negative result (B4) properly reverted and recorded. Full evidence:
session_report_2026-07-10_investigation.md (session scratchpad) + RUNS.md B4 entry.

## Verdicts

1. **Prod in-dist winner ECE ~0.27 = arithmetic artifact, closed.** split_prod val ⊂ train →
   memorized acc 0.9488; soft-cap max_prob=0.75 → mean conf 0.6753; ECE ≈ difference (0.2657).
   Measurement path is the true served path — no wiring bug. The report's ece≤0.07 tolerance was
   unsatisfiable by construction; fixed in `_prod_calibration_report.py` (ECE now info-only).
   Honest calibration number remains eval ECE 0.0473.
2. **pca_style parity bug (real, shipped):** HF Space served `pca_style_87b198e.joblib` with
   style_pc2 sign-flipped (corr −0.9999) vs prod-model training values. Impact small (P(win)
   |Δ| mean 0.0007, max 0.026). Fixed: whitelist swap to `pca_style_1821503.joblib` (bit-identical
   to training) + HF commit b8fe5b7; Space verified RUNNING with only the correct artifact.
3. **Gate B closure test (4 EVAL retrains, control + 3 row-shuffle seeds):** duration (p≈0.003),
   sig_strikes_combo (≈0.002), leg_sig (0.039–0.045), r1 (0.0000) fail EVERY draw → structural
   under the current feature build. Row-order jitter ±0.01–0.04 in p, never flips the fail-set →
   LightGBM order-sensitivity refuted as the historic flip mechanism; cross-run flips come from
   feature-BUILD state. v8.39 "10/11" confirmed as survivorship across builds. Residue: why the
   leak-era build passes PIT (failures concentrate in [finish]/[decision] buckets).
4. **B4 dead-list fossil re-trial (REVERTED):** 19 healthy-but-dead method features
   (finish_rate/ko_win_rate/r1_kd windows, dec_prone_combined) retrained clean but regressed
   Gate C KO BSS 0.0559→0.0346. Healthy data ≠ useful inputs; monotone prune vindicated.
   If retried: one window family at a time. RUNS.md B4; commit 004c242.
5. **B3 hub hypothesis refuted:** fixed-model swap shows the un-merged data IMPROVES ECE
   (0.0473→0.0464; ratings-only 0.0449); bootstrap ECE σ=0.0135 on n=769 makes the recorded
   0.0031 "regression" 0.23σ noise. Fix preserved on branch `fix/b3-bruno-silva-unmerge`
   (9520d42, stash also intact). Re-land pending a user decision on the ECE gate floor.
6. **Automation readiness (2026-07-11 card):** Thursday ledger job ran clean (UFC 329: 14 fights,
   171 lines, 833 prop rows, 65 pending winner preds), pushed + HF-deployed. Cloud fallbacks all
   green (prop_ledger success incl. 2026-07-10; closing_lines success 2026-07-04). Quirk found:
   on off-weeks the Saturday closing job stamps next week's card early (94 UFC 329 rows closed
   2026-07-04) — harmless because 07c is last-write-wins on pending rows, so the true closes
   overwrite on fight day. Power Play 403 standing; Flat Multi + Kalshi lanes healthy. Closing
   capture is CLOUD-ONLY (no local scheduled task).

## Lessons

- ECE deltas <~0.013 on the 769-fight test set are below sampling noise — the mandatory-revert
  rule can false-positive there; check σ before reverting on ECE alone.
- After any eval whitelist swap, sweep ALL model-key artifacts including pca_style — the alphabetical
  serving glob (matchup.py:464) makes a stale survivor silently win on a fresh checkout.
- "Healthy feature" (nonnull/variance/univariate corr) does not predict positive model value;
  gate-verified re-trial is the only arbiter (B4).

## Related

- [[2026-07-10-audit-remediation-review-and-ship]] — prior session, v8.40 ship
- [[2026-07-06-v839-sentinel-state-and-fill-median-leak]] — leak-fix provenance
- RUNS.md B4 entry — run data source of truth
