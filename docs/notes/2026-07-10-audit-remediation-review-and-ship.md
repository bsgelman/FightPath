---
type: research
date: 2026-07-10
model_version: eval 1821503 / prod e35f901
status: final
related_code_files:
  - src/ufc/inference/predict_core.py
  - src/ufc/api/serialize.py
  - src/ufc/inference/simulator.py
  - src/ufc/models/props_duration.py
  - src/ufc/models/props_count.py
  - src/ufc/models/base.py
  - src/ufc/features/weight_class.py
  - src/ufc/api/ratelimit.py
  - src/ufc/training/feature_pruning.py
  - configs/prop_trust.yaml
related_runs_entry: "RUNS.md 2026-07-09/10 audit remediation (B1/B2/B3/Phase C/Final)"
tags: [audit, remediation, review, gate-b, survivorship, trust-tiers, deploy]
---

# 2026-07-08 audit remediation: independent review + ship (v8.40)

## Summary

Two-session effort, independently double-checked. Session 1 (separate CLI, plan
`jolly-painting-clover.md`): 8 serving/money-path hotfixes (A1-A8), weight-class label
cleaning (B1), winner/method feature visibility (B2), Bruno Silva un-merge attempted
and REVERTED on a confirmed gate regression (B3), prod temporal_oof guards (Phase C),
combined prod retrain e35f901 (Final). Session 2 (this note): every claim re-verified
against the live repo — commits, gate artifacts, model internals, probe re-runs, and
live Gate B/D re-executions. **Nothing contradicted the reports.** Shipped 2026-07-10.

Key numbers: Gate A 0.6580/0.2219/0.0473 PASS; Gate C KO BSS 0.0502→0.0559 (the one
clear accuracy gain, from B2's weight-class-delta visibility); Gate B honest state
7/11+exempt (duration/leg/combo PIT-FAIL + r1 structural); Gate D PASS re-verified live.

## Evidence

- **Survivorship insight (the big one):** the recorded Gate B "10/11" (v8.39) was old
  frozen models re-scored on rebuilt features — models retained *because* they passed,
  across weeks of accept/revert ratcheting = implicit multiple-comparison selection on a
  fixed test set. Fresh honest retrains score 7-8/11. Eval/gate runs are bit-for-bit
  deterministic (three independent reproductions this session, KS stats identical to the
  digit), so cross-retrain flips come from data/feature-build state differences, NOT
  training stochasticity. Proposed closure test (not yet run): 3-5 control retrains
  varying SEED/row-order deliberately; props failing every draw are structural.
- **PIT-fail ≠ edge-fail:** duration (KS p=0.0026) and leg_sig (p=0.0057) fail Gate B
  PIT on the honest retrain, yet their edge-pick hit-rates on the SAME test set are
  0.997 (n=1154) and 0.752 (n=149) vs 0.577 breakeven. Distribution shape is
  miscalibrated; pick-level edge survives. Trust tiers therefore unchanged (see below).
- **B3 (Bruno Silva):** data bug 100% real — one fighter_id carries two people
  (23 fights: 11 FLW + 11 MW + 1 BW; two distinct UFCstats profiles). The correct
  un-merge REGRESSES Gate A ECE 0.0473→0.0504 (isolation-tested: zero-change control
  reproduced baseline bit-for-bit, so causation confirmed). Reverted per gate rule.
  Fix preserved in `git stash` ("B3 WIP for control test"). Untested hypothesis: the
  merged record acts as a rating-graph hub whose incoherent blend accidentally
  calibrates neighbors. Cheap probe before retry: diff Elo/Glicko of his ~20 opponents
  between the two builds.
- **dead-features monotone lists:** `dead_features_{winner,method}.txt` never
  re-evaluate ("once dead, always dead" — feature_pruning.py). B2 required manually
  clearing method's stale entries. Winner's `weight_class_change_lbs_a` was honestly
  re-pruned post-B1 (twin `_b` survived at 1 LGBM split, rank 474/475 — redundant
  anti-symmetric pair, ~noise for winner). Lesson: after populating a formerly-NaN
  feature, check BOTH dead lists.
- **Eval whitelist drift caught at ship time:** `outputs/models/.gitignore` still
  tracked b9db43c/87b198e/2bc425d-era joblibs while the gate-validated 1821503 set sat
  untracked — a fresh HF checkout would have loaded pre-fix models. Swapped (commit
  8e727a7). The whitelist invariant is only true if every retrain-and-validate session
  ends by updating it.
- **Trust tiers (configs/prop_trust.yaml, scorecard 2026-07-10):** zero tier moves;
  numbers + caveats refreshed. Overrode two script suggestions: knockdowns "TRUST" was
  a degenerate n=1 pick (CI [1.0,1.0]) → stays CUT; sub_attempts 0 picks → stays CUT.
- **Open items:** (1) A8 rightmost-XFF assumes exactly one trusted proxy on HF Spaces —
  live tests can't discriminate; close by hitting the rate limit from two networks
  post-deploy. (2) body_sig_strikes prod rate_calib_factor=1.1524, outside [0.90,1.10]
  — shipped accept-and-monitor; check body picks after first graded card. (3) B1
  anomaly closure test (seed-varied control retrains) not run. (4) prod in-dist winner
  ECE ~0.27 predates this work (0.2871→0.2657, slightly improved) — standing question,
  not a regression.

## Related

- RUNS.md entries 2026-07-09/10 (B1/B2/B3/Phase C/Final) — source of truth for run data
- [[2026-07-06-v839-sentinel-state-and-fill-median-leak]] — the leak-fix + "10/11" provenance
- audit_checkpoint.md (2026-07-08) — original ranked findings; A-F1 + C-T4-sub refuted
- Session reports: other CLI's scratchpad `session_report_2026-07-09.md` / `_2026-07-10_final.md`
