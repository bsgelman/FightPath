# Injury-Stoppage Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the pipeline from treating freak-injury stoppages (86/8,772 fights) as genuine KO/SUB outcomes in finish-rate features, rating updates, and method-classifier labels.

**Architecture:** A curated CSV (`data/raw/manual/injury_stoppages.csv`) + one boolean ledger column (`injury_freak`) set at ingest. Three consumers read it, each landing as its own commit + full Gate A–D run: finish-rate numerators (`finishes.py`), rating-update dampening (`ratings.py`, factor 0.25), method-label exclusion (`train_all.py`). Spec: `docs/superpowers/specs/2026-07-15-injury-stoppage-flag-design.md`.

**Tech Stack:** pandas, pytest, existing gate scripts (`scripts/04_backtest.py`, `05_evaluate_props.py`, `05b_evaluate_method.py`, `_joint_coherence_check.py`).

## Global Constraints

- **One consumer change per commit; full Gate A–D run after each; REVERT on real regression** (RUNS.md rule). Gates are pre-registered as a **no-harm check** — 86 rows cannot move aggregates; a flat result is a PASS; do not revert on sub-sigma wobble (v8.40 B3 lesson).
- Gate pass floors: A = acc ≥0.64, Brier ≤0.225, ECE ≤0.05; B = per-prop PIT-KS p>0.05 (accepted structural fails exempt: r1_sig_strikes, duration, leg, combo); C = KO BSS ≥0.02, ECE CI-lower ≤0.05; D = zero violations.
- `INJURY_K_FACTOR = 0.25`, config key `elo.injury_k_factor` in `configs/features.yaml`. Not fit — validated by the band check.
- **Check A (hard):** after Step-2 feature rebuild, frozen prod winner model (`outputs/models/prod/winner_ensemble_e35f901.joblib`) scoring Joshua Van vs Alexandre Pantoja (2026-09-19, 5rd title, LA) must give P(Van) strictly inside **(0.669, 0.701)**.
- **Check B (soft):** post-retrain P(Van) below 0.701; directional only, never a hard gate.
- Winner W/L labels, records, grading: UNTOUCHED. Winner-model training row count IDENTICAL through Tasks 4–5.
- The UFC 323 reference fight_id is `dfa692db6d39330c` (Pantoja vs Van, freak=1).
- Never push to origin or HuggingFace without Ben's explicit go (repo rule).
- Run the `leakage-auditor` agent on the diff before every retrain (project convention for `src/ufc/features/` + `src/ufc/training/` diffs).
- The eval-tier retrain command is `python scripts/03_train.py` (env: repo `.venv`, `PYTHONPATH=src`); prod tier is `python scripts/03_train_prod.py --auto`. Retrains take a long time — run with `run_in_background`.
- Append a RUNS.md entry per task (follow the existing entry format at the top of RUNS.md).

---

### Task 1: Curation CSV

**Files:**
- Create: `data/raw/manual/injury_stoppages.csv`
- Test: `tests/test_injury_flags.py` (first test only)

**Interfaces:**
- Produces: CSV with columns `fight_id,detail_text,injury_type,freak,note` — `fight_id` str (ufcstats hex), `injury_type` ∈ {arm,leg,knee,eye,rib,other}, `freak` ∈ {0,1}, `note` non-empty rationale. UNSURE rows: `freak=0`, `note` starts with literal `UNSURE:`. Consumed by Task 2's `_injury_freak_flags`.

- [ ] **Step 1: Generate the skeleton from the raw scrape**

Run this to emit all keyword-hit rows with fight_id + detail text (curator fills `injury_type`, `freak`, `note`):

```python
# scratchpad one-off — do not commit the script, only the CSV
import pandas as pd, re
df = pd.read_csv(r"data\raw\scraper\ufc_fight_results.csv")
hex_re = re.compile(r"fight-details/([0-9a-f]+)")
m = df["DETAILS"].astype(str).str.contains("injur", case=False, na=False)
out = pd.DataFrame({
    "fight_id": df.loc[m, "URL"].str.extract(hex_re)[0],
    "detail_text": df.loc[m, "DETAILS"].str.strip(),
    "injury_type": "", "freak": "", "note": "",
})
out.to_csv(r"data\raw\manual\injury_stoppages.csv", index=False)
print(len(out), "rows")  # expect 86
```

- [ ] **Step 2: Curate every row** — fill `injury_type`, `freak` (freak = non-combat-induced: awkward landing, twist, spontaneous dislocation; combat damage like checked-kick breaks or strike-induced doctor stoppages = 0), one-line `note` rationale per row. Rows the curator cannot decide: `freak=0`, note `UNSURE: <why>`. Row `dfa692db6d39330c` must be `injury_type=arm, freak=1`. Present the UNSURE list to Ben.

- [ ] **Step 3: Write the failing validation test**

```python
# tests/test_injury_flags.py
from pathlib import Path
import pandas as pd

CSV = Path(__file__).parents[1] / "data" / "raw" / "manual" / "injury_stoppages.csv"

def test_curation_csv_valid():
    df = pd.read_csv(CSV, dtype={"fight_id": str})
    assert list(df.columns) == ["fight_id", "detail_text", "injury_type", "freak", "note"]
    assert df["fight_id"].is_unique and df["fight_id"].notna().all()
    assert df["freak"].isin([0, 1]).all()
    assert df["injury_type"].isin(["arm", "leg", "knee", "eye", "rib", "other"]).all()
    assert (df["note"].astype(str).str.len() > 3).all(), "every row needs a rationale"
    pantoja = df[df["fight_id"] == "dfa692db6d39330c"]
    assert len(pantoja) == 1 and pantoja["freak"].iloc[0] == 1
```

- [ ] **Step 4: Run test** — `pytest tests/test_injury_flags.py -v` → PASS (fails until curation is complete; fix the CSV, not the test).

- [ ] **Step 5: Commit** — `git add data/raw/manual/injury_stoppages.csv tests/test_injury_flags.py && git commit -m "data: curated injury-stoppage CSV (86 fights, freak/combat + rationale)"`

---

### Task 2: Ingest — `injury_freak` ledger column + tripwire

**Files:**
- Modify: `src/ufc/ingest/parse_scraper.py` (parse_fights, ~line 93 and out_cols ~line 99)
- Modify: `src/ufc/ingest/build_ledger.py` (record dict ~line 257)
- Test: `tests/test_injury_flags.py`

**Interfaces:**
- Produces: `_injury_freak_flags(merged: pd.DataFrame, curation: pd.DataFrame | None = None) -> pd.Series` (bool, index-aligned; loads the CSV when `curation is None`). `parse_fights()` output + interim `fights` parquet + ledger gain bool column `injury_freak` (fight-level, identical on both fighter rows).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_injury_flags.py
from ufc.ingest.parse_scraper import _injury_freak_flags

def _merged(details, fight_ids):
    return pd.DataFrame({"DETAILS": details, "fight_id": fight_ids})

def test_flags_curated_freak_and_combat():
    cur = pd.DataFrame({"fight_id": ["aaa", "bbb"], "freak": [1, 0]})
    m = _merged(["Arm injury to X", "Leg injury", "Punches"], ["aaa", "bbb", "ccc"])
    flags = _injury_freak_flags(m, curation=cur)
    assert flags.tolist() == [True, False, False]

def test_uncurated_keyword_row_defaults_freak(capsys):
    cur = pd.DataFrame({"fight_id": [], "freak": []})
    m = _merged(["Knee Injury", "Punches"], ["new1", "ccc"])
    flags = _injury_freak_flags(m, curation=cur)
    assert flags.tolist() == [True, False]
    assert "injury-keyword rows not in curation CSV: 1" in capsys.readouterr().out

def test_tripwire_prints_zero(capsys):
    cur = pd.DataFrame({"fight_id": ["aaa"], "freak": [1]})
    flags = _injury_freak_flags(_merged(["Arm injury"], ["aaa"]), curation=cur)
    assert "injury-keyword rows not in curation CSV: 0" in capsys.readouterr().out
```

- [ ] **Step 2: Run** — `pytest tests/test_injury_flags.py -v` → FAIL (`_injury_freak_flags` not defined).

- [ ] **Step 3: Implement in `parse_scraper.py`**

```python
def _injury_freak_flags(merged: pd.DataFrame,
                        curation: pd.DataFrame | None = None) -> pd.Series:
    """Freak-injury stoppage flag (spec: docs/superpowers/specs/2026-07-15-injury-stoppage-flag-design.md).

    Curation CSV wins; keyword-hit rows absent from it default to freak=1
    (tripwire) and are counted loudly EVERY run, including zero — the DETAILS
    text is untrusted and this rule fails open if its format drifts.
    """
    if curation is None:
        cur_path = paths.root() / "data" / "raw" / "manual" / "injury_stoppages.csv"
        curation = (pd.read_csv(cur_path, dtype={"fight_id": str})
                    if cur_path.exists()
                    else pd.DataFrame({"fight_id": [], "freak": []}))
    detail = merged.get("DETAILS", pd.Series("", index=merged.index)).fillna("")
    keyword = detail.str.contains("injur", case=False)
    freak_map = dict(zip(curation["fight_id"].astype(str),
                         curation["freak"].astype(int)))
    curated_flag = merged["fight_id"].map(freak_map)
    uncurated_kw = keyword & curated_flag.isna()
    print(f"  [injury tripwire] injury-keyword rows not in curation CSV: {int(uncurated_kw.sum())}")
    return (curated_flag.fillna(0).astype(bool) | uncurated_kw)
```

(`paths` needs importing at the top of `parse_scraper.py` if not already: `from ufc.io import paths`.)

In `parse_fights()`, after the `is_title` block (~line 97): `merged["injury_freak"] = _injury_freak_flags(merged)` and add `"injury_freak"` to `out_cols`.

In `build_ledger.py`, next to `is_title` (~line 153): `injury_freak = bool(fight.get("injury_freak", False))`, and in the `record` dict next to `"is_title"`: `"injury_freak": injury_freak,`.

- [ ] **Step 4: Run** — `pytest tests/test_injury_flags.py tests/test_parse_helpers.py -v` → all PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(ingest): injury_freak ledger flag from curation CSV + fail-open tripwire"`

---

### Task 3: Plumbing no-op (spec Step 0) — rebuild, leak guard, metric-identical gates

**Files:**
- Modify: `src/ufc/features/assemble.py:181-185` (fight_cols list)
- Modify: `src/ufc/models/base.py:26-45` (default_exclude list)
- Test: `tests/test_injury_flags.py`

**Interfaces:**
- Produces: `injury_freak` present in ledger, `features_winner`, `features_props`, `pre_fight_state` parquets; NEVER in any model's `feature_cols`.

- [ ] **Step 1: Capture the pre-change gate baseline** — run all four gate scripts on the CURRENT artifacts and save the printed metrics (Gate A acc/Brier/ECE, per-prop KS table, Gate C BSS/ECE, Gate D violations) into the RUNS.md entry as "baseline before injury-flag work":

```
python scripts/04_backtest.py
python scripts/05_evaluate_props.py
python scripts/05b_evaluate_method.py
python scripts/_joint_coherence_check.py
```

- [ ] **Step 2: Add the column to `fight_cols`** in `assemble.py` (line ~184, after `"referee", "location", ...`): add `"injury_freak"` to the `fight_cols` list. Add `"injury_freak"` to `default_exclude` in `base.py` (bool dtype is already skipped by the numeric filter, but the explicit entry survives a future dtype change and documents intent).

- [ ] **Step 3: Write the failing leak-guard test**

```python
# append to tests/test_injury_flags.py
def test_injury_freak_never_a_model_feature():
    from ufc.models.base import get_feature_cols
    df = pd.DataFrame({"injury_freak": [True, False], "elo_pre_a": [1.0, 2.0]})
    cols = get_feature_cols(df)
    assert "injury_freak" not in cols and "elo_pre_a" in cols
```

- [ ] **Step 4: Run** — `pytest tests/test_injury_flags.py -v` → PASS.

- [ ] **Step 5: Rebuild data** — `python scripts/01_ingest.py` then `python scripts/02_build_features.py` (background; ~minutes). Confirm the tripwire line prints `: 0` and `pre_fight_state` contains `injury_freak`.

- [ ] **Step 6: Metric-identical gate check** — re-run the four gate scripts; every metric must equal Step 1's baseline EXACTLY (models unchanged, feature values unchanged, only an excluded column added). Any difference = wiring bug; stop and fix.

- [ ] **Step 7: Full test suite** — `pytest tests/ -x -q` → PASS (sentinel tests cover the new bool column's sentinel handling).

- [ ] **Step 8: Commit + RUNS.md** — `git commit -am "feat(features): plumb injury_freak through assemble; excluded from model features (Step 0, metric-identical)"`

---

### Task 4: Finish-rate hygiene (spec Step 1)

**Files:**
- Modify: `src/ufc/features/finishes.py:40-43`
- Test: `tests/test_injury_flags.py`

**Interfaces:**
- Consumes: ledger `injury_freak` (Task 2/3).
- Produces: all ten `finishes.py` numerators (`ko_win_rate`, `sub_win_rate`, `ko_loss_rate`, `sub_loss_rate`, `finish_rate`, `dec_rate`, `early_finish_rate`, `r1_ko_win_rate`, `r1_sub_win_rate`, `prior_5rd_dec_rate`) treat freak fights as experience-only. Numerator enumeration: every one flows from `is_ko`/`is_sub` (verified — no residual-share computation exists in finishes.py; `dec_rate` is computed independently from `is_dec` and correctly stays 0 for a freak fight, which is factually true: it was not a decision).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_injury_flags.py
import numpy as np
from ufc.features.finishes import compute_finish_rates

def _toy_ledger():
    # fighter f1: KO win (normal), KO win (freak), then a 3rd row to read ctd features
    n = 3
    return pd.DataFrame({
        "fighter_id": ["f1"] * n, "opponent_id": ["o1", "o2", "o3"],
        "event_rank": [1, 2, 3],
        "event_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
        "method": ["KO/TKO", "KO/TKO", "U-DEC"],
        "won": pd.array([1, 1, 1], dtype="Int8"),
        "end_round": pd.array([1, 1, 3], dtype="Int8"),
        "scheduled_rounds": pd.array([3, 3, 3], dtype="Int8"),
        "weight_class": ["Flyweight"] * n,
        "injury_freak": [False, True, False],
    })

def test_freak_ko_win_gets_no_ko_credit():
    out = compute_finish_rates(_toy_ledger())
    row3 = out[out["event_rank"] == 3].iloc[0]
    # 2 prior wins, only 1 real KO -> 0.5 (would be 1.0 if the freak fight counted)
    assert row3["ko_win_rate_ctd"] == 0.5
    # freak fight still in the denominator: finish_rate over 2 prior fights = 0.5
    assert row3["finish_rate_ctd"] == 0.5
```

- [ ] **Step 2: Run** — `pytest tests/test_injury_flags.py::test_freak_ko_win_gets_no_ko_credit -v` → FAIL (currently 1.0).

- [ ] **Step 3: Implement** — in `compute_finish_rates`, replace lines 40-41:

```python
    inj = df.get("injury_freak", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    # Freak-injury stoppages are evidence of nothing except that the fight
    # occurred: no KO/SUB credit (winner) or debit (loser). Every numerator
    # below flows from these two binaries, so this is the single choke point.
    is_ko = ((m == "KO/TKO") & ~inj).astype(float)
    is_sub = ((m == "SUB") & ~inj).astype(float)
```

- [ ] **Step 4: Run** — `pytest tests/test_injury_flags.py tests/test_windows_noleak.py -v` → PASS.

- [ ] **Step 5: Rebuild + audit + retrain** — `python scripts/02_build_features.py`; run the `leakage-auditor` agent on the diff; then `python scripts/03_train.py` (background, long).

- [ ] **Step 6: Gates + checks** — run all four gate scripts. Gate A/C must PASS, Gate D zero violations, Gate B judged against the accepted-exempt list. Run the band probe (Task 6's script works from Task 6 onward; before it exists, reuse the session pattern): expect P(Van) between ~0.68 and 0.701 (directional — the hard band applies after Task 5). Winner training row count must equal baseline.

- [ ] **Step 7: Commit + RUNS.md** — `git commit -am "feat(features): freak-injury fights excluded from all finish-rate numerators (Step 1)"`

---

### Task 5: Rating dampening (spec Step 2)

**Files:**
- Modify: `configs/features.yaml` (elo section)
- Modify: `src/ufc/features/ratings.py` (compute_elo ~57-131, compute_glicko2 ~182-263, compute_trueskill ~298-347)
- Test: `tests/test_injury_flags.py`

**Interfaces:**
- Consumes: ledger `injury_freak`.
- Produces: config key `elo.injury_k_factor: 0.25`; helper `_injury_k() -> float` in ratings.py; all three systems dampen the rating-mean movement of freak-decided fights by that factor (interpolation toward the pre-fight mean; RD/sigma updates untouched).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_injury_flags.py
from ufc.features.ratings import compute_elo, compute_glicko2, compute_trueskill

def _two_isolated_fights(freak_second):
    # (f1 beats f2) normal, (f3 beats f4) freak — fresh fighters, identical setup,
    # plus per-fighter sentinel-style later rows to read the post-fight rating.
    rows = []
    for i, (w, l, fid, inj) in enumerate([("f1", "f2", "A", False),
                                          ("f3", "f4", "B", freak_second)]):
        for me, opp, won in [(w, l, 1), (l, w, 0)]:
            rows.append(dict(fight_id=fid, fighter_id=me, opponent_id=opp,
                             won=won, method="KO/TKO",
                             event_date=pd.Timestamp("2020-01-01"),
                             event_rank=1, injury_freak=inj))
        for me, opp in [(w, l), (l, w)]:
            rows.append(dict(fight_id=f"read_{fid}_{me}", fighter_id=me, opponent_id=opp,
                             won=np.nan, method=None,
                             event_date=pd.Timestamp("2021-01-01"),
                             event_rank=2, injury_freak=False))
    df = pd.DataFrame(rows)
    df["won"] = pd.array(df["won"], dtype="Int8")
    return df

def test_injury_dampens_all_three_rating_systems():
    led = _two_isolated_fights(freak_second=True)
    for fn, col in [(compute_elo, "elo_pre"),
                    (compute_glicko2, "glicko_mu_pre"),
                    (compute_trueskill, "ts_mu_pre")]:
        out = fn(led)
        def gain(f):  # post-fight rating minus initial, read at the later row
            post = out[(out["fighter_id"] == f) & (out["event_rank"] == 2)][col].iloc[0]
            pre = out[(out["fighter_id"] == f) & (out["event_rank"] == 1)][col].iloc[0]
            return post - pre
        assert abs(gain("f3") / gain("f1") - 0.25) < 0.02, f"{col}: {gain('f3')/gain('f1')}"
```

- [ ] **Step 2: Run** — `pytest tests/test_injury_flags.py::test_injury_dampens_all_three_rating_systems -v` → FAIL (ratio 1.0).

- [ ] **Step 3: Implement**

`configs/features.yaml`, under `elo:` (after `k_base: 24`): `injury_k_factor: 0.25   # freak-injury outcomes transfer 4x less rating (spec 2026-07-15)`

`ratings.py` module level:

```python
def _injury_k() -> float:
    return float(_load_cfg().get("injury_k_factor", 0.25))
```

All three functions: add `"injury_freak"` to the ledger column select list, and in the fight_pairs loop store `"injury": bool(row.get("injury_freak", False))` on dict creation.

- `compute_elo` (after line 128): `if info.get("injury"): K_a *= _injury_k(); K_b *= _injury_k()` (K-scaling ≡ interpolation for Elo).
- `compute_glicko2` (line 263): `scale = _injury_k() if info.get("injury") else 1.0` then `mu[f_i] = mu[f_i] + scale * phi[f_i]**2 * g_j * (s_ij - E_ij)` — phi/sigma paths untouched.
- `compute_trueskill` (after line 345):

```python
        if info.get("injury"):
            f = _injury_k()
            new_ra = env.create_rating(mu=ra.mu + f * (new_ra.mu - ra.mu), sigma=new_ra.sigma)
            new_rb = env.create_rating(mu=rb.mu + f * (new_rb.mu - rb.mu), sigma=new_rb.sigma)
```

(TrueSkill's select list currently lacks `method`; it only needs `injury_freak` added.)

- [ ] **Step 4: Run** — `pytest tests/test_injury_flags.py tests/test_elo_chronology.py tests/test_sentinel_ratings.py tests/test_matchup_rating_parity.py -v` → PASS.

- [ ] **Step 5: Rebuild + audit + retrain** — `python scripts/02_build_features.py`; leakage-auditor on the diff; `python scripts/03_train.py` (background).

- [ ] **Step 6: HARD band check (Check A) + gates** — run Task 6's band script (create it first if executing out of order — it is self-contained): frozen prod model + rebuilt `pre_fight_state` → P(Van) must be strictly inside (0.669, 0.701). At/above 0.701 → flag not reaching features; at/below 0.669 → K too strong. Then all four gate scripts (same pass rules as Task 4), winner row count unchanged.

- [ ] **Step 7: Commit + RUNS.md** — `git commit -am "feat(features): dampen Elo/Glicko/TrueSkill mean updates 4x for freak-injury fights (Step 2)"`

---

### Task 6: Band-check script (Check A tooling)

**Files:**
- Create: `scripts/_injury_band_check.py`

**Interfaces:**
- Consumes: `outputs/models/prod/winner_ensemble_*.joblib` via `_find_latest_model` (prefers prod/ — eval retrains never move it), rebuilt reference parquets via `load_reference_data()`.
- Produces: printed P(Van) + PASS/FAIL verdict against (0.669, 0.701). Exit code 1 on FAIL.

- [ ] **Step 1: Write the script**

```python
"""Check A (spec 2026-07-15): frozen-prod-model band probe for the injury flag.

Scores Van vs Pantoja (2026-09-19 rematch) under the CURRENT prod winner
weights against the freshly rebuilt pre_fight_state. Hard gate after the
Step-2 (ratings) rebuild: P(Van) strictly inside (0.669, 0.701).
Run: PYTHONPATH=src python scripts/_injury_band_check.py
"""
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.inference.predict_core import _find_latest_model, load_reference_data, predict_fight
from ufc.models.winner import WinnerModel

LO, HI = 0.669, 0.701

def main() -> int:
    wp = _find_latest_model("winner_ensemble_*.joblib")
    print(f"winner model: {wp}")
    models = {"winner": WinnerModel.load(wp)}
    fighters_df, state, ref_history_df = load_reference_data()
    pred = predict_fight(
        red_name="Joshua Van", blue_name="Alexandre Pantoja",
        rounds=5, is_title=True, event_date=date(2026, 9, 19),
        models=models, fighters_df=fighters_df, pre_fight_state=state,
        location="Los Angeles, California, USA",
        ref_history_df=ref_history_df, run_simulation=False,
    )
    p = pred.prob_red
    inside = LO < p < HI
    print(f"P(Van) = {p:.4f}  band ({LO}, {HI})  ->  {'PASS' if inside else 'FAIL'}")
    if not inside:
        print("  at/above HI: flag not reaching features | at/below LO: dampening too strong")
    return 0 if inside else 1

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it** — `python scripts/_injury_band_check.py`. Against the pre-Task-4 artifacts it prints ~0.7009 FAIL (expected — that IS the pre-fix baseline); after Task 5's rebuild it must PASS.

- [ ] **Step 3: Commit** — `git add scripts/_injury_band_check.py && git commit -m "feat(scripts): frozen-model band probe for injury-flag verification (Check A)"`

*(Note: Tasks 4–5 reference this script; creating it before Task 4 is fine and recommended.)*

---

### Task 7: Method-label exclusion (spec Step 3)

**Files:**
- Modify: `src/ufc/training/train_all.py:188-191`

**Interfaces:**
- Consumes: `features_props` parquet column `injury_freak` (Task 3).
- Produces: method classifier trained and calibrated without freak-injury rows (calibration/halflife search counts as training — val rows excluded too).

- [ ] **Step 1: Implement** — replace the valid-methods block:

```python
    # DQ/NC are corrupt labels for method prediction — drop them
    valid_methods = ["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC"]
    p_train = p_train[p_train["method"].isin(valid_methods)].copy()
    p_val = p_val[p_val["method"].isin(valid_methods)].copy()
    # Freak-injury outcomes are corrupt method labels too (a 26s arm dislocation
    # is not evidence of KO ability) — exclude from fit AND calibration folds.
    if "injury_freak" in p_train.columns:
        n0 = len(p_train) + len(p_val)
        p_train = p_train[~p_train["injury_freak"].fillna(False).astype(bool)].copy()
        p_val = p_val[~p_val["injury_freak"].fillna(False).astype(bool)].copy()
        print(f"  Dropped {n0 - len(p_train) - len(p_val)} freak-injury method rows")
```

- [ ] **Step 2: Verify the drop count** — the printed number must equal the count of freak=1 CSV fights inside the train+val date windows (compute the expected value from the CSV + `configs/split.yaml` dates before running). No feature rebuild needed (labels only).

- [ ] **Step 3: Retrain + gates** — leakage-auditor on the diff; `python scripts/03_train.py` (background); all four gate scripts — **Gate C is the watch** (KO BSS ≥0.02, ECE CI-lower ≤0.05); winner rows unchanged; band probe = Check B directional only (post-retrain).

- [ ] **Step 4: Commit + RUNS.md** — `git commit -am "feat(training): exclude freak-injury fights from method-classifier labels (Step 3)"`

---

### Task 8: Prod retrain + ship gate

**Files:** none new (runs scripts; RUNS.md entry).

- [ ] **Step 1: Prod retrain** — `python scripts/03_train_prod.py --auto` (background, large).
- [ ] **Step 2: Prod calibration sanity** — `python scripts/_prod_calibration_report.py`; rate_calib factors must be ∈ [0.90, 1.10].
- [ ] **Step 3: Check B on the new prod model** — `python scripts/_injury_band_check.py` now loads the NEW prod weights: directional expectation only (P(Van) < 0.701); note the value in RUNS.md.
- [ ] **Step 4: STOP and ask Ben** before `git push origin main` (Vercel) or any HuggingFace deploy — never auto-push. Present: gate table (baseline vs final), band-check values after each step, method-row drop count, UNSURE curation list status.

---

## Spec-Step-4 (layoff × injury feature): NOT in this plan

Pre-registered optional/last/default-dead. Only plannable after Tasks 1–8 land AND the band check shows residual room. If Ben wants it then, it gets its own one-task plan.

## Self-review notes

- Spec coverage: §1→Task 1, §2→Task 2, Step 0→Task 3, Step 1→Task 4, Step 2→Task 5, Step 3→Task 7, Check A/B→Tasks 6+5.6+8.3, prod-retrain workflow→Task 8, out-of-scope list honored (no winner-label, NC, rate-window, transitivity changes).
- Type consistency: `injury_freak` is bool end-to-end; `_injury_freak_flags` returns bool Series; `freak` CSV column int {0,1}.
- The `dec_rate`/residual-share coherence requirement resolved by inspection: finishes.py computes `is_dec` independently (line 42); no share is derived residually; a freak fight correctly contributes 0 to dec numerators.
