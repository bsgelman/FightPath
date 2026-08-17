# UFC Prediction Pipeline — Architecture Specification

_Current version: v8.30 (2026-06-16)_

## Overview

This pipeline produces calibrated winner probabilities and distributional prop predictions for UFC matchups. It is structured to be strictly leakage-free: no feature computed for a fight at date D may use any information from fights at date ≥ D.

---

## 1. Ingestion Strategy

### Source-of-Truth Decision

**`scrape_ufc_stats-main/` is the canonical source.** The archive (`data/raw/archive/`) is used only for reach back-fills and DOB disambiguation.

| Source | Role | Why |
|--------|------|-----|
| `scrape_ufc_stats-main/` | Primary | Round-by-round granularity; fight URL IDs; daily-updating pipeline |
| `data/raw/archive/` | Supplement | Pre-2015 reach back-fill; DOB cross-check for name disambiguation |

**What the archive must NOT contribute:** `r_SLPM`, `r_STR_ACC`, `r_SAPM`, `r_STR_DEF`, `r_TD_AVG`, `r_TD_DEF`, `r_SUB_AVG` — these are current-snapshot career averages that would silently leak future information into historical predictions.

### Data Files

**Scraper outputs used:**
- `ufc_event_details.csv` → `data/interim/events.parquet`
- `ufc_fight_details.csv` + `ufc_fight_results.csv` → `data/interim/fights.parquet`
- `ufc_fight_stats.csv` → `data/interim/fight_rounds.parquet` (round-by-round)
- `ufc_fighter_tott.csv` + `ufc_fighter_details.csv` → `data/interim/fighters.parquet`

**Canonical fighter ID:** URL hex from `http://ufcstats.com/fighter-details/<hex>`. All joins use this key.

### Name Resolution

Fighter names in `BOUT` strings are normalized:
1. Lowercase + ASCII-fold via `unicodedata`
2. Strip punctuation, collapse whitespace
3. Hard-mapped first via `configs/name_overrides.yaml`
4. Looked up against the fighter table; collisions disambiguated by event-date proximity

Unresolved names logged to `outputs/reports/unresolved_names.txt`.

---

## 2. The Rolling-Horizon (Leakage Prevention) Mechanism

### The Fundamental Problem

A model trained on "career average strike accuracy" computed from a fighter's full career will have future fights embedded in the "historical" feature for older fights. This is temporal data leakage and will produce optimistic in-sample metrics that collapse out-of-sample.

### The Solution: `shift(1)` Causal Rolling

Every feature for fighter X going into fight at date D is computed from the strict prefix of their career (fights with date < D). The mechanism:

```
Fighter X's fight history (chronological):
  Fight 1 (Jan 2019) → Fight 2 (Aug 2019) → Fight 3 (Mar 2020) → Fight 4 (Oct 2020)

Feature at Fight 3:
  Value = f(Fight 1, Fight 2)   ← only prior fights

shift(1) applied before any expanding/rolling aggregation:
  s.shift(1).expanding().mean()
  = [NaN, fight1_val, mean(fight1,fight2), mean(fight1,fight2,fight3), ...]
  
Feature at index i uses only values {0..i-1}.
```

Implementation in `src/ufc/features/windows.py`:
```python
def causal_expanding(df, by, sort_col, value_col, agg="mean"):
    return (df.sort_values([by, sort_col])
              .groupby(by)[value_col]
              .apply(lambda s: s.shift(1).expanding().agg(agg))
              .reset_index(level=0, drop=True))
```

### ELO Chronology

ELO is updated in event order in a single pass. The **pre-fight rating** is stored on the ledger row before the update is applied:

```
for fight in sorted(fights, by event_date):
    store(fighter_A.elo_pre)   ← this becomes the feature
    store(fighter_B.elo_pre)
    update(fighter_A.elo, outcome)   ← this happens AFTER storing
    update(fighter_B.elo, outcome)
```

### Window Flavors

Every numeric feature is emitted in 5 time-window variants:

| Suffix | Description |
|--------|-------------|
| `_ctd` | Career-to-date expanding mean (excluding current fight) |
| `_l3` | Last 3 fights rolling mean |
| `_l5` | Last 5 fights rolling mean |
| `_2y` | Last 24 months (date-based) |
| `_decay` | Exponentially time-decayed, 18-month half-life |

---

## 3. Ledger Schema

`data/processed/ledger.parquet` — one row per (fight_id, fighter_id) = 2 rows per fight.

| Column | Type | Description |
|--------|------|-------------|
| `fight_id` | str | URL hex from ufcstats.com |
| `event_id` | str | Event URL hex |
| `event_date` | date | Fight date |
| `event_rank` | int | Monotonic index across all events by date |
| `fighter_id` | str | This fighter's URL hex |
| `opponent_id` | str | Opponent's URL hex |
| `weight_class` | str | e.g. "Featherweight" |
| `scheduled_rounds` | int | 3 or 5 |
| `is_title` | bool | Title fight indicator |
| `is_main_event` | bool | Main event indicator |
| `won` | Int8 | 1=win, 0=loss, NaN=NC/DQ |
| `method` | str | KO/TKO, SUB, U-DEC, S-DEC, M-DEC, DQ, NC |
| `end_round` | Int8 | Round the fight ended |
| `end_time_sec` | Int16 | Time within end_round in seconds |
| `total_fight_sec` | Int16 | Total elapsed seconds |
| `sig_str_landed` | int | Significant strikes landed |
| `sig_str_attempted` | int | Significant strikes attempted |
| `sig_str_absorbed_landed` | int | Opponent's sig strikes landed on this fighter |
| `sig_str_absorbed_attempted` | int | Opponent's sig strikes attempted on this fighter |
| `td_landed` | int | Takedowns landed |
| `td_attempted` | int | Takedowns attempted |
| `td_absorbed_landed` | int | Opponent's takedowns on this fighter |
| `td_absorbed_attempted` | int | Opponent's takedown attempts on this fighter |
| `ctrl_sec` | int | Control time in seconds |
| `ctrl_sec_absorbed` | int | Time controlled by opponent |
| `kd_for` | int | Knockdowns scored |
| `kd_against` | int | Knockdowns absorbed |
| `sub_att_for` | int | Submission attempts |
| `sub_att_against` | int | Opponent submission attempts |
| `rev_for` | int | Reversals |
| `rev_against` | int | Opponent reversals |
| `head_landed` | int | Head strikes landed |
| `body_landed` | int | Body strikes landed |
| `leg_landed` | int | Leg strikes landed |
| `distance_landed` | int | Distance strikes landed |
| `clinch_landed` | int | Clinch strikes landed |
| `ground_landed` | int | Ground strikes landed |
| `head_absorbed` | int | Head strikes absorbed |
| `body_absorbed` | int | Body strikes absorbed |
| `leg_absorbed` | int | Leg strikes absorbed |
| `distance_absorbed` | int | Distance strikes absorbed |
| `clinch_absorbed` | int | Clinch strikes absorbed |
| `ground_absorbed` | int | Ground strikes absorbed |
| `age_years` | float | Age at fight date |
| `reach_in` | float | Reach in inches |
| `height_in` | float | Height in inches |
| `stance` | str | ORTHO, SOUTH, SWITCH, OPEN |
| `weight_lbs` | float | Weight class in pounds |
| `referee` | str | Referee name |
| `location` | str | Event location |

---

## 4. Feature Catalog

| Group | Module | Key Features | Leakage-safe? |
|-------|--------|-------------|---------------|
| Ratings | `ratings.py` | `elo_pre`, `elo_diff`, `glicko_mu_pre`, `glicko_rd_pre` | Yes — chronological pass, pre-fight stored before update |
| Striking | `striking.py` | `slpm`, `sapm`, `str_acc`, `str_def`, by-location shares | Yes — `causal_expanding/rolling` |
| Grappling | `grappling.py` | `td_per_15`, `td_acc`, `td_def`, `ctrl_pct`, `sub_att_per_15` | Yes |
| Style | `style.py` | `striker_score`, `wrestler_score`, `grappler_score` (z-score), PCA components | Yes — PCA fit on train fold only |
| Interactions | `interactions.py` | `sub_trap`, `reach_punish`, `cardio_gap_5rd`, `power_vs_chin`, `tdd_vs_wrestler` | Yes — built from pre-fight features |
| Mileage | `mileage.py` | `age_years`, `damage_index`, `layoff_days`, `def_pct_trend_l3`, `recent_finish_loss_12mo`, `weight_class_change_lbs`, `layoff_age_interaction` | Yes |
| Physical | `physical.py` | `reach_diff`, `height_diff`, `age_diff`, `stance_pair` | Yes — bio snapshot at fight date |
| Context | `context.py` | `is_title`, `scheduled_rounds`, `referee_stoppage_threshold`, `altitude_meters` | Yes |

**Note:** `weight_class_change_lbs` and `layoff_age_interaction` were added in v8.4. `short_notice_flag` was removed as a dead placeholder.

---

## 5. Model Architecture

Full determinism is enforced via `SEED=42`, `num_threads=1`, `deterministic=True`, and all five LGBM sub-seeds pinned. Models are stamped with the git short-sha at save time (e.g. `winner_ensemble_788b9b6.joblib`); loaders resolve the newest artifact by mtime.

### Winner Model (`src/ufc/models/winner.py`)

- **Algorithm:** Diverse ensemble — 3×LGBM (seeds 42/43/44) + 2×CatBoost + 2×XGB + 1×Logistic (ratings-only). SLSQP log-loss blend weight optimization on OOF predictions.
- **Calibration pipeline:** OOF blend → isotonic regression → Platt scaling (Nelder-Mead) → ECE-optimal symmetric prob cap (`max_prob`)
- **OOF scheme (tier-aware):**
  - *Eval tier* (`temporal_oof=False`): `StratifiedKFold(n_splits=5, shuffle=True)` + val-B for Platt/cap. Gates byte-identical across retrains.
  - *Prod tier* (`temporal_oof=True`): `TimeSeriesSplit(n_splits=5)` expanding window; Platt/cap fit on most-recent 18-month OOF slice. No in-sample calibration overlap.
- **Symmetry:** Each fight generates two training rows (A-perspective, B-perspective). At inference, both orderings are predicted and logit-averaged via `inference_average`
- **Per-WC dampening (v8.5):** At inference only, logit is dampened toward 0.5 by a per-weight-class temperature (`wc_temperature.py`). HW=1.20, LHW=1.15, WStraw=1.10 — widens the distribution for high-variance weight classes
- **Feature count:** ~474 columns after zero-importance pruning

### Method Classifier (`src/ufc/models/method.py`)

- **Algorithm:** Single LightGBM multinomial → raw logits → temperature scaling (1 DOF, fit on val set via `_fit_temperature`) → rolling-36mo era prior blend
- **Rolling era prior (v8.2):** Last 36 months of training data used for the prior rather than full-history base rates, capturing temporal drift in finish rates. Prior weight shrinks toward data as n increases.
- **Output:** `predict_proba_dict(df)` returns `{class: array}` in `METHOD_CLASSES = ["KO/TKO", "SUB", "DEC"]` order

### Duration Model (`src/ufc/models/props_duration.py`)

Two-stage hurdle architecture:
1. **Decision classifier:** `CalibratedClassifierCV(StratifiedKFold(n_splits=5, shuffle=False))` around an LGBM base → P(decision)
2. **Finish quantile regression:** 11 LGBM quantile regressors (q=0.05…0.95) on finish-only rows → finish time CDF
3. **Method dummies:** `method_ko`, `method_sub` appended to the feature matrix so the duration model learns method-conditional distributions. At inference, `method_override` injects the dummy explicitly.
4. **Boundary mass:** Redistributive probability mass placed at intermediate round ends (calibrated from training data); used for duration KS evaluation but disabled (`use_boundary_mass=False`) for count-model MC integration.
5. **Output:** `DurationCDF` object with `.cdf(t)`, `.survival(t)`, `.median_sec`, `.p_over_rounds(r)`.

**Method-conditional duration CDFs** are built at inference for KO/TKO, SUB, and DEC separately, then mixed by predicted method probabilities. This is required for count model calibration (see §5.4).

### Count Models — `RateHurdleCountModel` (`src/ufc/models/props_count.py`)

Used for sig_strikes, takedowns, r1_sig_strikes. Three-stage architecture:

**Stage 1 — Hurdle classifier:**
LightGBM binary classifier → P(count > 0). For takedowns, a method/duration-conditional hurdle (`pos_clf_cond`, CalibratedClassifierCV) captures the zero-mass difference between KO and SUB fights.

**Stage 2 — Log-rate quantile regressors:**
11 LightGBM quantile regressors on positive-count rows predict the log-rate per active minute. The quantile grid spans [0.05, 0.95] and gives the full shape of the per-fight rate distribution. At inference: `rate = exp(log_rate_pred + log(rate_calib_factor))`.

**Stage 3 — Monte Carlo integration:**
For each row, samples N=512 (duration_sec, method) draws from the joint distribution, then draws count from Poisson(rate × duration_sec). This produces a full `RateXDurationCDF` (a sample-backed empirical CDF) per fighter per fight.

**Method adjustments (`fit_method_adjustments`):**
Learned post-fit on training residuals:
- `method_log_rate_adj`: per-method flat rate scalar (KO/TKO, SUB, DEC) from residual regression. **Zeroed at inference** for sig_strikes and takedowns — method signal is already carried by the duration CDFs; the flat adj double-counts.
- `_sw_sub` (takedowns only): SUB-conditional count head with recency weighting (730d halflife, anchored 2023-12-31), capturing the era trend toward higher TD counts in submission fights.
- `finish_head_disp_factor`: dispersion-widening factor (~2.0) for the r1_sig_strikes finish head; applied only in the diagnostic `force_r1_end` path, not the marginal.

**Rate calibration factor (sig_strikes only, v8.13):**
```
rate_calib_factor = mean(actual_val) / mean(predicted_val_marginal)
```
Computed on the 2023 validation set using the **method-marginal** forecast (method-conditional durations mixed by predicted method probs, rate-adj zeroed). Value ≈ 0.97. Expected range [0.90, 1.10]. Applied as `log_rate += log(factor)` in `predict_cdf`.

**Critical invariant — method-marginal parity (v8.13):**  
All three count models (sig_strikes, takedowns, r1) must use method-conditional duration CDFs at both gate evaluation and production inference. Passing a method-blind duration CDF causes the duration model to default to DEC-mode for every fight → ~40% duration inflation → severely miscalibrated count predictions (production PIT-KS 0.243 without this fix). The production calls in `predict.py` and the gate calls in `05_evaluate_props.py` are kept in sync.

### Monte Carlo Simulator (`src/ufc/inference/simulator.py`)

Given a matchup, draws N joint samples of (winner, method, duration, per-fighter counts) coherently:
1. Sample method from `method_probs`
2. Sample duration from the method-conditional `DurationCDF`
3. Sample counts from each fighter's `RateXDurationCDF` conditioned on the sampled duration
4. Sample winner from `winner_prob`

Used for correlated prop leg evaluation and portfolio construction.

---

## 6. Inference Path (`predict.py`)

For an upcoming fight (no realized `method`), the full inference sequence:

1. Build `feat` / `feat_flip` via `build_matchup_features` (pre-fight rolling state; excludes outcome columns including `method`)
2. **Winner:** `WinnerModel.predict_proba(feat)` + symmetry average + WC temperature dampening
3. **Method:** `MethodClassifier.predict_proba_dict(feat)` averaged over both perspectives → `method_probs = {KO/TKO, SUB, DEC}`
4. **Duration:** `DurationModel.predict_cdf(feat)` (single method-blind CDF used internally only) + three method-conditional CDFs via `method_override`. A **method-marginal display CDF** is synthesized by blending `_lgbm_qv` and `_p_dec` across methods weighted by `method_probs` — this is what `print_duration_cdf` and `print_rounds_distribution` consume, ensuring the displayed P(ends R5) is coherent with the method distribution.
5. **Sig strikes:** `RateHurdleCountModel.predict_cdf(feat, method_proba=..., duration_cdfs_by_method=..., method_log_rate_adj=None)` — method-marginal, rate-adj zeroed.
6. **Takedowns:** Same method-marginal call (v8.11 fix); `use_sub_count_head=True`, `use_cond_hurdle=False`.
7. **R1 sig strikes:** Method-marginal call with `active_minutes_ceiling=5.0`, `use_finish_head=True`, `apply_burst=False`.
8. **Simulation:** `simulate(...)` for joint MC samples used in portfolio evaluation.

---

## 7. Time-Series Split

Two **model tiers** share one codebase and architecture. The active split is selected via
`os.environ["UFC_SPLIT_CONFIG"]` (default `"split.yaml"`).

### Eval tier — `configs/split.yaml` (locked)

| Split | Date Range | Purpose |
|-------|-----------|---------|
| Train | 2010-01-01 → 2023-12-31 | Model fitting |
| Validation | 2024-01-01 → 2024-12-31 | Calibration, rate factor, method adjustments |
| Test (locked) | 2025-01-01 → 2026-12-31 | Final gate evaluation only |

Artifacts land in `outputs/models/`. Gates A–D always evaluate **eval-tier** artifacts.

### Prod tier — `configs/split_prod.yaml` (rolling)

| Split | Date Range | Purpose |
|-------|-----------|---------|
| Train | 2010-01-01 → 2024-12-31 | Model fitting |
| Validation | 2025-01-01 → 2025-12-31 | Calibration + rate factor |
| Holdout | 2026-01-01 → 2026-12-31 | Sanity check only (not a gate) |

Artifacts land in `outputs/models/prod/`. The serving loader (`predict_core._find_latest_model`)
looks in `prod/` first and falls back to `outputs/models/` — so the prod tier serves production
when present, else the eval tier is the fallback.

**Ship rule:** eval Gates A–D PASS **and** prod holdout (winner Brier ≤ 0.235, ECE ≤ 0.07;
no prop flips from PASS to FAIL) → ship prod artifacts.

Run prod retrain: `python scripts/03_train_prod.py --auto`
Run prod sanity: `python scripts/_prod_calibration_report.py`

Pre-2010 data excluded: stats coverage is too sparse for reliable features.

**Recency anchors** in `train_all.py` halflife searches and the SUB-head recency weight in
`props_count.py` read `train_end` from the active split config, so prod retrains automatically
anchor at 2024-12-31 instead of 2023-12-31. No hardcoded date changes are needed when
advancing the prod split with `--auto`.

---

## 8. Calibration Gates

Every full retrain must pass all three gates before shipping. Run from repo root.

### Gate A — Winner (`scripts/04_backtest.py`)

| Metric | Threshold | v8.13 |
|--------|-----------|-------|
| Accuracy | ≥ 0.62 | 0.635 |
| Brier score | ≤ 0.225 | 0.224 |
| ECE | ≤ 0.04 | 0.027 |

### Gate B — Props (`scripts/05_evaluate_props.py`)

All four props evaluated via **randomized PIT-KS** against the **method-marginal** forecast (the production path). KS statistic < critical value at p=0.05.

| Prop | Threshold | v8.13 |
|------|-----------|-------|
| Duration KS | p > 0.05 | 0.034 (p=0.26) |
| Sig strikes KS | p > 0.05 | 0.033 (p=0.28) |
| Takedowns KS | p > 0.05 | 0.034 (p=0.26) |
| R1 sig strikes KS | p > 0.05 | 0.029 (p=0.44) |
| Sig strikes CRPS skill | > 0 | 0.095 |

**Non-gating diagnostics:** sig_strikes [finish] and [decision] segments, takedowns [finish], r1 [r1_end] — annotated 🔬 in the report. These are valid conditional-null failures for a method-marginal model conditioned on realized-method slices and do not indicate miscalibration.

### Gate C — Method (`scripts/05b_evaluate_method.py`)

| Metric | Threshold | v8.13 |
|--------|-----------|-------|
| KO/TKO BSS | ≥ 0.02 | 0.030 |
| All-class ECE (noise-aware) | CI lower bound ≤ 0.05 | DEC 0.055, CI [0.033, 0.090] → PASS |

---

## 9. Training Entry Points

| Command | What it does |
|---------|-------------|
| `python scripts/03_train.py` | Full eval-tier retrain (split.yaml): winner → method → count props → duration → rate factor |
| `python scripts/03_train_prod.py` | Full prod-tier retrain (split_prod.yaml) → `outputs/models/prod/` |
| `python scripts/03_train_prod.py --auto` | Same but auto-derives split dates from `max(event_date)` |
| `python scripts/_prod_calibration_report.py` | Score prod model on 2026-H1 holdout → `outputs/reports/prod_calibration_<date>.md` |
| `python scripts/_retrain_count_only.py` | Count props only (loads newest duration + method models from disk) |
| `python scripts/_retrain_duration_only.py` | Duration model only |
| `python scripts/01_ingest.py` | Raw CSVs → interim parquets → ledger |
| `python scripts/02_build_features.py` | Ledger → features_props / features_winner / pre_fight_state |

**Rate calibration factor check:** During `03_train.py` or `03_train_prod.py`, the line `[v8.13] sig_strikes rate_calib_factor = X.XXXX` must print a value in [0.90, 1.10]. A value outside this range (especially near 0.684) indicates the method-marginal helper is receiving a method-blind frame — investigate before shipping.

---

## 10. Model Artifacts

All artifacts are gitignored (`.joblib`) except those explicitly whitelisted in `.gitignore`.
Named `<model>_<gitsha>.joblib`. Loaders resolve newest by **mtime** (not alphabetical).

**Loader priority:** `predict_core._find_latest_model` checks `outputs/models/prod/` first,
then falls back to `outputs/models/`. Gate scripts (`04_backtest.py`, `05_evaluate_props.py`,
`05b_evaluate_method.py`, `_joint_coherence_check.py`) glob only the root `outputs/models/`
directory and are unaffected by prod artifacts.

**Known loader exceptions (non-production):** `05_evaluate_props.py:_td_hurdle_diagnostic`
and `06_sanity_sweep.py` use alphabetical sort — safe to fix opportunistically.

### Eval-tier artifacts (`outputs/models/`) — whitelisted in `outputs/models/.gitignore`

Frozen per the locked eval split. Update the whitelist only on a full eval-tier retrain.

### Prod-tier artifacts (`outputs/models/prod/`) — whitelisted in `outputs/models/prod/.gitignore`

Regenerated weekly / post-scrape via `03_train_prod.py --auto`. After each prod retrain,
update `outputs/models/prod/.gitignore` with the new sha (remove old, add new) before
pushing to HuggingFace.
