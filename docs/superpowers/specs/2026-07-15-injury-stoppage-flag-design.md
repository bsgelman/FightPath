# Injury-Stoppage Flag — Design Spec

**Date:** 2026-07-15
**Status:** Approved design, reviewed by party-mode panel (8 deltas incorporated)
**Motivation:** UFC 323 (Pantoja vs Van) ended at 0:26 R1 by freak arm injury. The pipeline
codes it as a genuine KO: Van gains phantom KO-power credit and full rating transfer,
Pantoja is debited a KO-loss (chin). Measured total effect on the Van–Pantoja rematch
prediction: P(Van) 0.701 with the fight vs 0.669 with it fully removed (~3.2pp).
This spec makes the pipeline *stop lying to itself* about such outcomes. It is a
correctness/robustness fix, not an accuracy unlock.

## Scope decision (made by Ben, 2026-07-15)

Model label-hygiene, full treatment. NOT a learned categorical input feature
(86 injury fights / 8,772 total — trees cannot learn from that support). NOT UI-only.

## 1. Data + curation

- New file `data/raw/manual/injury_stoppages.csv`:
  `fight_id, detail_text, injury_type (arm|leg|knee|eye|rib|other), freak (0|1), note`
- `note` is mandatory: one-line rationale per row so future disagreement argues with a
  reason, not an oracle.
- The CSV is a **union**: seeded from the keyword rule (`injur`, case-insensitive, in
  DETAILS — 86 hits today) PLUS hand-added fight_ids the keyword missed (e.g. details
  reading "Leg Kick" for what was actually a freak break). Keyword is an aid, never
  the gatekeeper.
- Curation workflow: Claude drafts all rows with rationale; ambiguous rows marked
  UNSURE for Ben to rule on.
- Defaults (split deliberately):
  - New scrape rows hitting the keyword but absent from the CSV → treated as `freak=1`
    (tripwire: aggressive until reviewed).
  - Human-reviewed rows left UNSURE → `freak=0` (combat) until Ben rules. UNSURE is
    encoded as `freak=0` + `note` beginning with the literal string `UNSURE:` (no
    extra schema column; greppable).
- Freak boundary: freak = non-combat-induced (awkward landing, twist, spontaneous
  dislocation). Combat damage (checked-kick break, cut/doctor stoppage from strikes,
  damage-induced injury) = `freak=0`.

## 2. Ingest

- `parse_scraper.py`: ONE persisted ledger column, `injury_freak` (bool). The keyword
  flag is ingest-time only (seeds/warns the CSV; not persisted — Yui delta).
- Attribution: the injured party is always the **loser** (fight ends because they cannot
  continue). No name parsing of detail text.
- NC guard: fight-level flag always set; fighter-level consumers apply effects only
  where `won.notna()` (NCs have no loser — Boundary delta).
- Fail-open tripwire: scraped DETAILS is untrusted text that can silently change format.
  Weekly refresh ALWAYS logs `injury-keyword rows not in curation CSV: N` — including
  N=0, so the counter disappearing is itself a signal (Vex delta).
- Sentinel safety: bool column defaults False in `append_sentinel_rows` (existing
  bool branch handles it; no schema exception needed).

## 3. Consumers — one change per commit, full Gate A–D run each, REVERT on regression

**Gate framing (pre-registered):** 86 changed rows cannot move aggregate gate metrics
over the full test fold. Gates here are a **no-harm check, not a benefit detector**.
A flat gate result is a PASS. Do not noise-revert on sub-sigma wobbles (see the v8.40
B3 lesson: a 0.23-sigma "regression" was noise).

- **Step 0 — plumbing no-op:** columns added, nothing consumes them. Gate metrics must
  be identical (metric-identical, not artifact-byte-identical — the parquet grows a
  column). Catches wiring mistakes free.
- **Step 1 — finish rates:** governing principle — a freak-injury fight contributes to
  **experience denominators only**; it is evidence of nothing except that the fight
  occurred. No KO, SUB, finish, early-finish, R1-finish, or decision numerator credit
  on either side. The implementation plan's first task: grep `finishes.py` (and
  specialist scores built on it) and enumerate EVERY numerator against this principle,
  including a coherence check for any share computed residually (e.g. dec = 1−ko−sub).
  The W/L itself is untouched everywhere.
- **Step 2 — ratings:** dampen by **interpolation toward the pre-fight rating**:
  `post' = pre + INJURY_K_FACTOR × (post − pre)` applied to the rating mean for each
  system (Elo rating, Glicko-2 mu, TrueSkill mu); deviation/sigma updates untouched.
  `INJURY_K_FACTOR = 0.25`, named config constant next to existing rating constants.
  Winner still gains — Van did force the scramble — just 4× less. The constant is
  unfittable at n=86; it is validated by check A below, not fit.
- **Step 3 — method labels:** `injury_freak` fights excluded from method-classifier
  training rows (~51 fake-KO + ~9 fake-SUB labels removed). Gate C is the watch.
- **Step 4 — OPTIONAL, LAST, default-dead:** `layoff_after_injury` =
  `days_since_last_fight × last_fight_ended_by_own_injury`. Pre-registered expectation:
  ~86 nonzero rows, trees ignore it. Run ONLY after Steps 0–3 land AND the band check
  shows residual room; drop on ANY Gate A movement. If Steps 0–3 eat the full 3.2pp,
  Step 4 dies unrun (Dana compromise).

Then: prod retrain (`scripts/03_train_prod.py --auto`) + `scripts/_prod_calibration_report.py`
(rate_calib factors ∈ [0.90, 1.10]) before shipping, per project workflow.

## 4. Verification anchors (beyond gates)

The band check is split in two because retraining introduces per-matchup seed noise
LARGER than the 3.2pp effect being verified (v8.38 lesson: a no-op retrain moved two
gate props). A single post-retrain band check would measure noise.

- **Check A — frozen-model band check (the hard gate):** rebuild the hygiene-corrected
  `pre_fight_state` and score the 2026-09-19 rematch under the CURRENT frozen prod
  weights (e35f901), exactly as this session's counterfactual script did (no retrain,
  no seed). P(Van) must land INSIDE (0.669, 0.701):
  - at/above 0.701 → flag isn't reaching the features;
  - at/below 0.669 → dampening exceeds full-removal, K too strong.
- **Check B — post-retrain directional sanity (soft):** after each retraining step,
  P(Van) should sit below the pre-fix 0.701 baseline; no hard band (seed noise).
- **Row-count invariant:** winner-model training row count is IDENTICAL through
  Steps 1–2 (features change, rows never). Step 3 removes exactly the curated
  freak-injury rows from method training only.

## 5. Out of scope (explicit)

- Winner label (a win is a win — records, grading, W/L untouched).
- NC / eye-poke fouls (already excluded from winner training; prop grading voids NC).
- Short-fight per-15 rate pollution (26s fights annualize ×34 into sapm/td_def rates —
  real, adjacent, SEPARATE experiment).
- Transitivity method weights (effect too small to chase).
- PROD-tier-only serving tricks; both tiers get the same feature code per architecture.

## Panel review deltas — round 2 (2026-07-15, spec-text review)

9. Band check split: Check A frozen-model (hard) / Check B post-retrain directional
   (soft) — retrain seed noise exceeds the 3.2pp effect, so only the frozen-model
   check can be a hard gate.
10. Step 1 restated as a principle (freak fights feed experience denominators ONLY)
    + mandatory numerator enumeration of finishes.py incl. residual-share coherence.
11. Rating dampening specified as interpolation toward pre-fight mean (Elo/Glicko-2
    mu/TrueSkill mu); deviation/sigma untouched — "multiply the update" is Elo-shaped
    and ambiguous for the other two systems.
12. UNSURE encoding convention (freak=0 + note prefix "UNSURE:"); Step 0 wording fixed
    to metric-identical.

## Panel review deltas (2026-07-15, code-review-crew)

1. Single persisted column `injury_freak`; keyword flag ingest-time only.
2. Curation CSV is a union (keyword + manual adds), rationale mandatory per row.
3. NC guard on fighter-level attribution.
4. Loud always-on tripwire counter in weekly refresh, including N=0.
5. Split defaults: new-uncurated freak=1, reviewed-UNSURE freak=0.
6. Gates pre-registered as no-harm check; anti-noise-revert note.
7. Step 4 optional/last/default-dead.
8. K=0.25 validated by band check, not fit; landing at the band floor = constant too strong.
