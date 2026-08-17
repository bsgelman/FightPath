# Pre-Fight-State Sentinel Rows + Rating Parity + Divergence Flag — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serving state (`pre_fight_state.parquet`) must include each fighter's most recent completed fight (today it lags one fight), pairwise rating features must be computed against the actual opponent at serve time, and the UI must flag large model/market divergences.

**Architecture:** Three independent fixes to the serving path, zero change to training tables. (1) Append one no-outcome "sentinel" row per fighter to the ledger before the per-fighter feature build; causal rolling features computed for that row include the fighter's entire history; `pre_fight_state` becomes the sentinel rows. (2) The three rating passes get a single-fighter branch so sentinels receive current ratings. (3) `matchup.py` recomputes `opp_elo_pre`/`elo_diff`/`glicko_z`/`ts_z` for the requested pairing (training parity — today these are served stale, vs each fighter's LAST opponent). Plus a `divergence` boolean in the `/api/market-lines` rows and a badge in the exchange table.

**Tech Stack:** Python/pandas feature pipeline (`src/ufc/features/`), FastAPI serving (`src/ufc/api/`), React/Vite frontend (`frontend/src/`), pytest.

## Why this is safe (context for the implementer)

- Every rolling-feature helper in `src/ufc/features/windows.py` is causal via `shift(1)` per `(fighter, sort_col)` group: a row's features use only *prior* rows. A sentinel row sorted last per fighter therefore gets features covering ALL completed fights, and its own NaN stats can never leak into any other row (no rows come after it).
- Training tables are protected twice: sentinels are explicitly filtered out before the wide build, and `build_wide_fight_features` already skips any `fight_id` group with `len(grp) != 2` (a sentinel group has 1 row).
- Gates A–D read `features_winner.parquet` / `features_props.parquet` + eval models. Those tables must be **byte-identical** after this change (verified in Task 5). No retrain of any tier is needed — feature *semantics* are unchanged ("state as of before the fighter's next fight"); the state just stops being one fight stale.

## Global Constraints

- `features_winner.parquet` and `features_props.parquet` must be **content-identical** before/after (Task 5 verifies; if they differ, STOP and investigate — do not proceed to deploy).
- Gate A & C must PASS; Gate D zero violations; Gate B: ignore the accepted structural fails (`r1_sig_strikes`, `ctrl_time`, `takedowns`, duration/body/combo per RUNS.md) — only a NEW regression in a passing prop blocks.
- One complexity at a time: Tasks are ordered so each is independently verifiable; do not merge them into one commit.
- Never push `main` or HuggingFace without the user's explicit go (CLAUDE.md deploy rule). Task 8 is gated on the user.
- Run the `leakage-auditor` agent on the diff before the feature rebuild (Task 5) — the diff touches `src/ufc/features/`.
- End every commit message with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Windows environment; run Python via the repo venv (plain `python` resolves to it in this project dir). Pytest: `python -m pytest`.

## Known accepted side effects (do NOT "fix" these; they are deliberate)

- Sentinel rows get `stance_wr_vs_opp_stance = 0.5` (neutral) because they have no opponent. Previously this feature was served *stale vs the last opponent* — equally wrong, differently. Neutral is acceptable.
- Sentinel context features that depend on the fight itself (location/home-advantage, era columns) are NaN; `matchup.py` already overrides `altitude_meters`, `era_avg_sig_str_l12mo`, `wc_finish_share_l2y`, `wc_5rd_dec_rate` at request time (lines 315–384).
- `layoff_days` in the state is measured to the sentinel date (last fight + 1 day); `matchup.py:236-245` recomputes layoff to the actual event date at request time anyway, off the max `event_date` in `pre_fight_state` — post-change that max is the sentinel date, so serving layoff is 1 day short. Negligible; do not add correction logic.

---

### Task 1: Baseline snapshot (before ANY code change)

**Files:**
- Create: `C:\Users\benja\AppData\Local\Temp\claude\C--Users-benja-Desktop-UFC-Prediction-Model\9865fe1d-4237-4462-bb85-5827c95823fa\scratchpad\baseline_hashes.py` (scratchpad — not committed)

**Interfaces:**
- Produces: `baseline_features_winner.sha`, `baseline_features_props.sha`, `baseline_preds.json` in the scratchpad dir; consumed by Task 5's comparison.

- [ ] **Step 1: Confirm clean-ish git state on `main`** (untracked data files are fine; no modified `src/` files):

Run: `git status --short -- src/ scripts/ frontend/src/`
Expected: empty output. If not empty, stop and ask the user.

- [ ] **Step 2: Rebuild features from the current ledger on unmodified code** (this is the determinism baseline — rebuild-vs-rebuild, not rebuild-vs-committed-file):

Run: `python scripts/02_build_features.py`
Expected: completes; prints `features_winner: N fight rows`, `features_props: M fight rows`, `pre_fight_state: K fighters`. Record N, M, K.

- [ ] **Step 3: Write and run the snapshot script:**

```python
"""baseline_hashes.py — content hash of training tables + smoke predictions."""
import sys, json, hashlib, datetime as dt
sys.path.insert(0, "src")
import pandas as pd

SCRATCH = r"C:\Users\benja\AppData\Local\Temp\claude\C--Users-benja-Desktop-UFC-Prediction-Model\9865fe1d-4237-4462-bb85-5827c95823fa\scratchpad"

def content_hash(path):
    df = pd.read_parquet(path)
    df = df.reindex(sorted(df.columns), axis=1)
    sort_cols = [c for c in ("fight_id", "fighter_id_a") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()

for name in ("features_winner", "features_props"):
    h = content_hash(f"data/processed/{name}.parquet")
    open(f"{SCRATCH}\\baseline_{name}.sha", "w").write(h)
    print(name, h)

# Smoke predictions on the next card (UFC 329) — captures live behavior pre-change.
from ufc.inference.predict_core import load_models, load_reference_data, predict_fight
import json as _json
card = _json.load(open("cards/upcoming/ufc_329_mcgregor_vs_holloway_2_2026_07_11.json"))
models = load_models(verbose=False)
fighters_df, pfs, ref_hist = load_reference_data()
out = {}
for f in card["fights"]:
    try:
        p = predict_fight(f["red"], f["blue"], rounds=f.get("rounds", 3),
                          is_title=f.get("is_title", False),
                          event_date=dt.date(2026, 7, 11), models=models,
                          fighters_df=fighters_df, pre_fight_state=pfs,
                          ref_history_df=ref_hist, run_simulation=False)
        out[f"{f['red']} vs {f['blue']}"] = {
            "p_red": round(p.prob_red, 4),
            "method": {k: round(v, 4) for k, v in p.method_probs.items()},
        }
    except Exception as e:
        out[f"{f['red']} vs {f['blue']}"] = {"error": str(e)}
json.dump(out, open(f"{SCRATCH}\\baseline_preds.json", "w"), indent=1)
print(json.dumps(out, indent=1))
```

Note: check the card JSON's actual fight-list schema first (`cards/upcoming/ufc_329_mcgregor_vs_holloway_2_2026_07_11.json` — fights hold `"red"`/`"blue"` keys per the repo's card format; adjust key names to what the file actually contains).

Run: `python "<scratchpad>\baseline_hashes.py"`
Expected: two hashes printed + per-fight probabilities incl. `Cody Garbrandt vs Adrian Yanez` around `p_red 0.636`.

---

### Task 2: Single-fighter branch in the three rating passes

**Files:**
- Modify: `src/ufc/features/ratings.py` (three insertion points: lines ~84, ~197, ~299)
- Test: `tests/test_sentinel_ratings.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `compute_elo` / `compute_glicko2` / `compute_trueskill` now emit rating columns for single-fighter fight groups (pre-rating read, NO update; pairwise fields NaN). Task 3's sentinel rows rely on this.

**Why:** all three passes currently `continue` on `len(fighters) != 2`, so a sentinel (one row per fight_id) would get NaN `elo_pre` — and NaN→`fillna(0)` Elo is catastrophic under the winner model's monotone constraints (see the warning at `matchup.py:203-209`).

- [ ] **Step 1: Write the failing test** — `tests/test_sentinel_ratings.py`:

```python
"""Single-fighter (sentinel) fight groups must receive pre-ratings, no update."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import pytest

from ufc.features.ratings import compute_elo, compute_glicko2, compute_trueskill


def _ledger():
    """Two fighters, one real fight (f1 beats f2), then one sentinel row each."""
    rows = [
        # real fight — long-form: one row per (fight, fighter)
        dict(fight_id="F1", event_date=pd.Timestamp("2025-01-01"),
             fighter_id="f1", opponent_id="f2", won=1.0, method="KO/TKO"),
        dict(fight_id="F1", event_date=pd.Timestamp("2025-01-01"),
             fighter_id="f2", opponent_id="f1", won=0.0, method="KO/TKO"),
        # sentinels — single-fighter fights, no outcome
        dict(fight_id="sentinel_f1", event_date=pd.Timestamp("2025-01-02"),
             fighter_id="f1", opponent_id=None, won=np.nan, method=None),
        dict(fight_id="sentinel_f2", event_date=pd.Timestamp("2025-01-02"),
             fighter_id="f2", opponent_id=None, won=np.nan, method=None),
    ]
    return pd.DataFrame(rows)


def test_elo_sentinel_gets_post_fight_rating():
    out = compute_elo(_ledger())
    s1 = out[out["fight_id"] == "sentinel_f1"].iloc[0]
    s2 = out[out["fight_id"] == "sentinel_f2"].iloc[0]
    r1 = out[(out["fight_id"] == "F1") & (out["fighter_id"] == "f1")].iloc[0]
    # real fight row: both start at initial rating -> elo_pre equal
    assert s1["elo_pre"] > r1["elo_pre"]          # winner's rating went UP after F1
    assert s2["elo_pre"] < r1["elo_pre"]          # loser's went DOWN
    assert pd.isna(s1["opp_elo_pre"])             # no opponent on a sentinel


def test_glicko_sentinel_gets_current_mu():
    out = compute_glicko2(_ledger())
    s1 = out[out["fight_id"] == "sentinel_f1"].iloc[0]
    s2 = out[out["fight_id"] == "sentinel_f2"].iloc[0]
    assert s1["glicko_mu_pre"] > s2["glicko_mu_pre"]   # winner > loser post-update
    assert pd.notna(s1["glicko_rd_pre"])
    assert pd.isna(s1["glicko_z"])                     # pairwise field NaN


def test_trueskill_sentinel_gets_current_mu():
    trueskill = pytest.importorskip("trueskill")
    out = compute_trueskill(_ledger())
    s1 = out[out["fight_id"] == "sentinel_f1"].iloc[0]
    s2 = out[out["fight_id"] == "sentinel_f2"].iloc[0]
    assert s1["ts_mu_pre"] > s2["ts_mu_pre"]
    assert pd.isna(s1["ts_z"])


def test_real_rows_unchanged_by_sentinels():
    """Adding sentinel rows must not change any real row's ratings."""
    with_s = compute_elo(_ledger())
    without_s = compute_elo(_ledger()[~_ledger()["fight_id"].str.startswith("sentinel")])
    for fid, f in [("F1", "f1"), ("F1", "f2")]:
        a = with_s[(with_s["fight_id"] == fid) & (with_s["fighter_id"] == f)].iloc[0]
        b = without_s[(without_s["fight_id"] == fid) & (without_s["fighter_id"] == f)].iloc[0]
        assert a["elo_pre"] == b["elo_pre"]
        assert a["opp_elo_pre"] == b["opp_elo_pre"]
```

- [ ] **Step 2: Run to verify it fails:**

Run: `python -m pytest tests/test_sentinel_ratings.py -v`
Expected: FAIL — sentinel rows have NaN `elo_pre` (`assert s1["elo_pre"] > ...` fails on NaN comparison).

- [ ] **Step 3: Implement.** In `src/ufc/features/ratings.py`, insert a single-fighter branch immediately BEFORE each `if len(fighters) != 2: continue` guard.

In `compute_elo` (guard at line ~84):

```python
        if len(fighters) == 1:
            # Sentinel row (inference-state carrier): read current rating, no update.
            f_s, _ = fighters[0]
            if f_s not in elo:
                elo[f_s] = _init_elo(f_s)
            elo_records[(fid, f_s)] = {"elo_pre": elo[f_s], "opp_elo_pre": np.nan}
            continue
        if len(fighters) != 2:
            continue
```

In `compute_glicko2` (guard at line ~197):

```python
        if len(fighters) == 1:
            # Sentinel row: read current rating, no update.
            f_s, _ = fighters[0]
            _ensure(f_s)
            glicko_records[(fid, f_s)] = {
                "glicko_mu_pre": mu[f_s], "glicko_rd_pre": phi[f_s], "glicko_z": np.nan,
            }
            continue
        if len(fighters) != 2:
            continue
```

In `compute_trueskill` (guard at line ~299):

```python
        if len(fighters) == 1:
            # Sentinel row: read current rating, no update.
            f_s, _ = fighters[0]
            if f_s not in ratings:
                ratings[f_s] = _init_ts(f_s)
            ts_records[(fid, f_s)] = {
                "ts_mu_pre": ratings[f_s].mu, "ts_sigma_pre": ratings[f_s].sigma, "ts_z": np.nan,
            }
            continue
        if len(fighters) != 2:
            continue
```

Notes: `np` is already imported in ratings.py. Deliberately NO inactivity regression in the sentinel branch — the sentinel is dated 1 day after the last fight, so the gap can never exceed the threshold; keep the branch minimal.

- [ ] **Step 4: Run tests:**

Run: `python -m pytest tests/test_sentinel_ratings.py tests/test_elo_chronology.py -v`
Expected: all PASS (including the pre-existing chronology test — proves real-fight behavior untouched).

- [ ] **Step 5: Commit:**

```bash
git add src/ufc/features/ratings.py tests/test_sentinel_ratings.py
git commit -m "feat(features): rating passes emit pre-ratings for single-fighter sentinel rows

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Sentinel rows in assemble + pre_fight_state from sentinels

**Files:**
- Modify: `src/ufc/features/assemble.py` (new function + wiring at lines ~209-230)
- Test: `tests/test_sentinel_state.py` (new)

**Interfaces:**
- Consumes: Task 2's single-fighter rating branches.
- Produces: `append_sentinel_rows(ledger: pd.DataFrame) -> pd.DataFrame` (adds `is_sentinel` bool column; one sentinel per fighter). `pre_fight_state.parquet` now contains one sentinel-derived row per fighter whose rolling features include the fighter's full history. Training tables unchanged.

- [ ] **Step 1: Write the failing test** — `tests/test_sentinel_state.py`:

```python
"""append_sentinel_rows: structure, dating, nulling, and training-table isolation."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd

from ufc.features.assemble import append_sentinel_rows, SENTINEL_KEEP_COLS


def _ledger():
    return pd.DataFrame([
        dict(fight_id="F1", event_date=pd.Timestamp("2025-01-01"), event_rank=0,
             fighter_id="f1", opponent_id="f2", won=1.0, method="KO/TKO",
             weight_class="Bantamweight", stance="ORTHO", age_years=28.0,
             reach_in=68.0, height_in=68.0, weight_lbs=135.0,
             sig_str_landed=50.0, end_round=2, scheduled_rounds=3,
             referee="Herb Dean", location="Las Vegas"),
        dict(fight_id="F1", event_date=pd.Timestamp("2025-01-01"), event_rank=0,
             fighter_id="f2", opponent_id="f1", won=0.0, method="KO/TKO",
             weight_class="Bantamweight", stance="SOUTH", age_years=30.0,
             reach_in=70.0, height_in=69.0, weight_lbs=135.0,
             sig_str_landed=30.0, end_round=2, scheduled_rounds=3,
             referee="Herb Dean", location="Las Vegas"),
        dict(fight_id="F2", event_date=pd.Timestamp("2025-06-01"), event_rank=1,
             fighter_id="f1", opponent_id="f3", won=0.0, method="U-DEC",
             weight_class="Featherweight", stance="ORTHO", age_years=28.4,
             reach_in=68.0, height_in=68.0, weight_lbs=145.0,
             sig_str_landed=40.0, end_round=3, scheduled_rounds=3,
             referee="Marc Goddard", location="London"),
        dict(fight_id="F2", event_date=pd.Timestamp("2025-06-01"), event_rank=1,
             fighter_id="f3", opponent_id="f1", won=1.0, method="U-DEC",
             weight_class="Featherweight", stance="ORTHO", age_years=25.0,
             reach_in=72.0, height_in=70.0, weight_lbs=145.0,
             sig_str_landed=60.0, end_round=3, scheduled_rounds=3,
             referee="Marc Goddard", location="London"),
    ])


def test_one_sentinel_per_fighter():
    out = append_sentinel_rows(_ledger())
    sent = out[out["is_sentinel"]]
    assert len(sent) == 3                              # f1, f2, f3
    assert sent["fighter_id"].is_unique
    assert (~out["is_sentinel"]).sum() == 4            # real rows untouched


def test_sentinel_dated_after_last_fight_and_ranked_last():
    out = append_sentinel_rows(_ledger())
    s_f1 = out[(out["is_sentinel"]) & (out["fighter_id"] == "f1")].iloc[0]
    s_f2 = out[(out["is_sentinel"]) & (out["fighter_id"] == "f2")].iloc[0]
    assert s_f1["event_date"] == pd.Timestamp("2025-06-02")   # f1's last fight + 1d
    assert s_f2["event_date"] == pd.Timestamp("2025-01-02")   # f2's last fight + 1d
    assert (out[out["is_sentinel"]]["event_rank"] > 1).all()  # sorts after all real fights


def test_sentinel_keeps_identity_nulls_fight_facts():
    out = append_sentinel_rows(_ledger())
    s = out[(out["is_sentinel"]) & (out["fighter_id"] == "f1")].iloc[0]
    assert s["weight_class"] == "Featherweight"        # carried from LAST fight
    assert s["stance"] == "ORTHO"
    assert s["fight_id"] == "sentinel_f1"
    for col in ("won", "sig_str_landed", "end_round", "scheduled_rounds"):
        assert pd.isna(s[col]), col
    assert s["opponent_id"] is None or pd.isna(s["opponent_id"])
    assert s["method"] is None or pd.isna(s["method"])
    assert s["referee"] is None or pd.isna(s["referee"])


def test_real_rows_byte_identical():
    led = _ledger()
    out = append_sentinel_rows(led)
    real = out[~out["is_sentinel"]].drop(columns=["is_sentinel"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(real, led.reset_index(drop=True))
```

- [ ] **Step 2: Run to verify it fails:**

Run: `python -m pytest tests/test_sentinel_state.py -v`
Expected: FAIL with `ImportError: cannot import name 'append_sentinel_rows'`.

- [ ] **Step 3: Implement.** In `src/ufc/features/assemble.py`, add below the `_cfg()` function (line ~28):

```python
# Columns a sentinel row keeps from the fighter's last real row. Everything
# else is nulled: a sentinel describes the FIGHTER (identity/bio/division),
# never a fight. Nulled stats are safe — causal windows exclude the current
# row, and no row sorts after a sentinel within its fighter group.
SENTINEL_KEEP_COLS = {
    "fight_id", "fighter_id", "event_date", "event_rank", "is_sentinel",
    "weight_class", "stance", "age_years", "reach_in", "height_in", "weight_lbs",
}


def append_sentinel_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    """Append one no-outcome sentinel row per fighter, dated 1 day after their
    last fight and event-ranked after every real fight.

    The per-fighter causal feature build then produces, for each sentinel, a
    feature row covering the fighter's ENTIRE completed history — this becomes
    pre_fight_state (previously: the pre-fight row of the LAST fight, which
    silently dropped that fight's result from serving; see RUNS.md).
    Sentinels carry is_sentinel=True and MUST NOT reach the training tables.
    """
    df = ledger.copy()
    df["is_sentinel"] = False
    sent = (
        df.sort_values(["fighter_id", "event_date", "fight_id"])
        .groupby("fighter_id", sort=False)
        .tail(1)
        .copy()
    )
    sent["fight_id"] = "sentinel_" + sent["fighter_id"].astype(str)
    sent["event_date"] = pd.to_datetime(sent["event_date"]) + pd.Timedelta(days=1)
    sent["event_rank"] = df["event_rank"].max() + 1
    sent["is_sentinel"] = True
    for c in sent.columns:
        if c in SENTINEL_KEEP_COLS:
            continue
        if pd.api.types.is_datetime64_any_dtype(sent[c]):
            sent[c] = pd.NaT
        elif pd.api.types.is_numeric_dtype(sent[c]):
            sent[c] = np.nan
        else:
            sent[c] = None
    return pd.concat([df, sent], ignore_index=True)
```

- [ ] **Step 4: Wire into `assemble()`.** Two edits.

Edit A — insert sentinels after the era-baseline step and before the train-mask computation (so the mask covers sentinel rows too; they land in the post-train fold automatically). Replace lines 209-216:

```python
    # ── Era + weight-class baselines (all causal rolling) ────────────────────
    print("  Computing era + weight-class baselines...")
    ledger = context.compute_era_baselines(ledger)

    # ── Sentinel rows: one per fighter, so pre_fight_state includes the last fight ──
    print("  Appending per-fighter sentinel rows (inference state)...")
    ledger = append_sentinel_rows(ledger)

    # ── Compute train mask BEFORE per-fighter feature build so causal stats use it ──
    split_cfg = _cfg()
    train_end = pd.to_datetime(split_cfg["train_end"])
    ledger_train_mask = pd.to_datetime(ledger["event_date"]) <= train_end
```

CAUTION: sentinel `event_date` can be ≤ `train_end` for long-retired fighters, putting sentinels inside the train mask used by `fill_sparse_history` / `style` / `compute_opponent_quality` median-and-quantile fits. Their feature values are real (computed from history), but to keep every train-fold statistic byte-identical, exclude them explicitly:

```python
    ledger_train_mask = (
        (pd.to_datetime(ledger["event_date"]) <= train_end)
        & ~ledger["is_sentinel"]
    )
```

Use the masked version. This is what guarantees Task 5's byte-identical check.

Edit B — replace the `pre_fight_state` block (old lines 221-230, the `groupby(...).last()`) and drop sentinels from `pf` immediately after:

```python
    # Save pre_fight_state (sentinel rows = full-history state per fighter)
    latest_state = (
        pf[pf["is_sentinel"].fillna(False)]
        .drop(columns=["is_sentinel"])
        .reset_index(drop=True)
    )
    assert latest_state["fighter_id"].is_unique, "expected exactly one sentinel per fighter"
    paths.processed("pre_fight_state").parent.mkdir(parents=True, exist_ok=True)
    parquet.write(latest_state, paths.processed("pre_fight_state"))
    print(f"  pre_fight_state: {len(latest_state)} fighters")

    # Training tables must never see sentinels.
    pf = pf[~pf["is_sentinel"].fillna(False)].drop(columns=["is_sentinel"]).reset_index(drop=True)
```

- [ ] **Step 5: Run tests:**

Run: `python -m pytest tests/test_sentinel_state.py tests/test_sentinel_ratings.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit:**

```bash
git add src/ufc/features/assemble.py tests/test_sentinel_state.py
git commit -m "feat(features): pre_fight_state built from per-fighter sentinel rows — serving state now includes each fighter's most recent fight

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Serve-time pairwise rating parity in matchup.py

**Files:**
- Modify: `src/ufc/inference/matchup.py` (after the passthrough block, line ~313; add `import math` to module imports)
- Test: `tests/test_matchup_rating_parity.py` (new)

**Interfaces:**
- Consumes: `pre_fight_state` rows (sentinel-derived after Task 3; their `opp_elo_pre`/`glicko_z`/`ts_z` are NaN — this task makes that harmless).
- Produces: `build_matchup_features` output rows where `opp_elo_pre_*`, `elo_diff_*`, `glicko_z_*`, `ts_z_*` are computed for the ACTUAL pairing (training parity). Today these are served stale (vs each fighter's last opponent) — a latent production/gate parity bug of the same family as v8.11/v8.13.

- [ ] **Step 1: Write the failing test** — `tests/test_matchup_rating_parity.py`:

```python
"""Pairwise rating features must reflect the requested pairing, not the last opponent."""
import sys
sys.path.insert(0, "src")
import math
from datetime import date

import numpy as np
import pandas as pd

from ufc.inference.matchup import build_matchup_features


def _pfs():
    """Minimal 2-fighter pre_fight_state with deliberately WRONG stale pairwise
    fields (as if each last fought someone else)."""
    base = dict(
        event_date=pd.Timestamp("2026-01-01"), event_rank=100,
        weight_class="Bantamweight", stance="ORTHO",
        age_years=30.0, reach_in=68.0, height_in=68.0, weight_lbs=135.0,
    )
    return pd.DataFrame([
        dict(fighter_id="A", fight_id="sentinel_A", elo_pre=1600.0,
             opp_elo_pre=np.nan, elo_diff=np.nan,
             glicko_mu_pre=1620.0, glicko_rd_pre=80.0, glicko_z=np.nan,
             ts_mu_pre=27.0, ts_sigma_pre=4.0, ts_z=np.nan, **base),
        dict(fighter_id="B", fight_id="sentinel_B", elo_pre=1500.0,
             opp_elo_pre=np.nan, elo_diff=np.nan,
             glicko_mu_pre=1480.0, glicko_rd_pre=120.0, glicko_z=np.nan,
             ts_mu_pre=24.0, ts_sigma_pre=6.0, ts_z=np.nan, **base),
    ])


def _fighters():
    return pd.DataFrame([
        dict(fighter_id="A", fighter_name="Fighter A", dob=pd.Timestamp("1996-01-01"),
             reach_in=68.0, height_in=68.0, weight_lbs=135.0, stance="ORTHO"),
        dict(fighter_id="B", fighter_name="Fighter B", dob=pd.Timestamp("1994-01-01"),
             reach_in=70.0, height_in=69.0, weight_lbs=135.0, stance="SOUTH"),
    ])


def test_pairwise_ratings_recomputed_for_actual_pairing():
    feat = build_matchup_features(
        "A", "B", date(2026, 7, 11), 3, False,
        pre_fight_state=_pfs(), fighters_df=_fighters(),
    )
    r = feat.iloc[0]
    assert r["opp_elo_pre_a"] == 1500.0
    assert r["opp_elo_pre_b"] == 1600.0
    assert r["elo_diff_a"] == 100.0
    assert r["elo_diff_b"] == -100.0
    gz = (1620.0 - 1480.0) / math.sqrt(80.0**2 + 120.0**2)
    assert abs(r["glicko_z_a"] - gz) < 1e-9
    assert abs(r["glicko_z_b"] + gz) < 1e-9
    tz = (27.0 - 24.0) / math.sqrt(4.0**2 + 6.0**2)
    assert abs(r["ts_z_a"] - tz) < 1e-9
    assert abs(r["ts_z_b"] + tz) < 1e-9
```

- [ ] **Step 2: Run to verify it fails:**

Run: `python -m pytest tests/test_matchup_rating_parity.py -v`
Expected: FAIL — `opp_elo_pre_a` is NaN (copied from state row, never recomputed).

- [ ] **Step 3: Implement.** In `src/ufc/inference/matchup.py`: add `import math` to the module-level imports. Then insert AFTER the passthrough block (after `row["stance_b"] = ...`, line ~313):

```python
    # ── Pairwise rating features vs the ACTUAL opponent (training parity) ────
    # State rows carry these vs each fighter's LAST opponent (stale) or NaN
    # (sentinel rows). Training computes them against the true opponent, so
    # serving must too. Mirrors ratings.py: glicko_z/ts_z are uncertainty-
    # scaled mu gaps; elo_diff is the plain rating gap.
    elo_a, elo_b = state_a.get("elo_pre"), state_b.get("elo_pre")
    if pd.notna(elo_a) and pd.notna(elo_b):
        row["opp_elo_pre_a"], row["opp_elo_pre_b"] = elo_b, elo_a
        row["elo_diff_a"], row["elo_diff_b"] = elo_a - elo_b, elo_b - elo_a
    mu_a, mu_b = state_a.get("glicko_mu_pre"), state_b.get("glicko_mu_pre")
    rd_a, rd_b = state_a.get("glicko_rd_pre"), state_b.get("glicko_rd_pre")
    if pd.notna(mu_a) and pd.notna(mu_b):
        _den = math.sqrt(
            (rd_a if pd.notna(rd_a) else 0.0) ** 2
            + (rd_b if pd.notna(rd_b) else 0.0) ** 2
        ) or 1.0
        _gz = (mu_a - mu_b) / _den
        row["glicko_z_a"], row["glicko_z_b"] = _gz, -_gz
    tmu_a, tmu_b = state_a.get("ts_mu_pre"), state_b.get("ts_mu_pre")
    tsg_a, tsg_b = state_a.get("ts_sigma_pre"), state_b.get("ts_sigma_pre")
    if pd.notna(tmu_a) and pd.notna(tmu_b):
        _den = math.sqrt(
            (tsg_a if pd.notna(tsg_a) else 0.0) ** 2
            + (tsg_b if pd.notna(tsg_b) else 0.0) ** 2
        ) or 1.0
        _tz = (tmu_a - tmu_b) / _den
        row["ts_z_a"], row["ts_z_b"] = _tz, -_tz
```

- [ ] **Step 4: Run the new test plus the existing inference tests:**

Run: `python -m pytest tests/test_matchup_rating_parity.py tests/test_attribution.py tests/test_name_match.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit:**

```bash
git add src/ufc/inference/matchup.py tests/test_matchup_rating_parity.py
git commit -m "fix(inference): recompute pairwise rating features (opp_elo/elo_diff/glicko_z/ts_z) for the actual pairing — production/gate parity

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Full rebuild + verification gauntlet

**Files:**
- No source changes. Regenerates `data/processed/*.parquet`. Creates scratchpad comparison scripts.

**Interfaces:**
- Consumes: Task 1's baseline hashes + `baseline_preds.json`.
- Produces: verified `pre_fight_state.parquet` (fresh, includes last fights); proof training tables are unchanged; gate confirmation; smoke-test delta report. This is the go/no-go gate for Tasks 6-8.

- [ ] **Step 1: Leakage audit.** Dispatch the `leakage-auditor` agent on the diff (`git diff <baseline-sha> -- src/ufc/features/ src/ufc/inference/`). Expected: no temporal/label leakage findings. The key facts for the auditor: sentinel rows carry NaN outcomes, sort last per fighter, are excluded from the train mask and training tables; causal windows are shift(1)-based.

- [ ] **Step 2: Rebuild features:**

Run: `python scripts/02_build_features.py`
Expected: completes; `pre_fight_state: K fighters` with the SAME K as Task 1 (one sentinel per fighter); `features_winner`/`features_props` row counts identical to Task 1's N and M.

- [ ] **Step 3: Byte-identity check.** Re-run the hash half of Task 1's script (content_hash of both training parquets) and compare to `baseline_features_winner.sha` / `baseline_features_props.sha`.
Expected: hashes IDENTICAL. If not: diff column sets and a few rows to find the contamination; fix before proceeding. (Likeliest culprits: train mask not excluding sentinels, or the `pf` filter applied after a merge that reindexed.)

- [ ] **Step 4: State freshness spot-check** (scratchpad script):

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
pfs = pd.read_parquet("data/processed/pre_fight_state.parquet")
assert (pfs["fight_id"] == "sentinel_" + pfs["fighter_id"]).all(), "state must be all sentinels"
# Garbrandt row — find by joining fighters table
fighters = pd.read_parquet("data/interim/fighters.parquet")
gid = fighters.loc[fighters["fighter_name"] == "Cody Garbrandt", "fighter_id"].iloc[0]
row = pfs[pfs["fighter_id"] == gid].iloc[0]
print("event_date:", row["event_date"])        # expect 2026-03-08 (last fight + 1d)
print("fights_career:", row["fights_career"])   # expect 17 (was 16)
print("elo_pre:", row["elo_pre"])               # expect != 1533.404 (March result folded in)
print("volume_trend_l3:", row["volume_trend_l3"])  # expect NEGATIVE (28 strikes vs Xiao Long)
```

Expected: all four assertions in the comments hold. `fights_career` MUST be 17 — this is the single clearest proof the lag is gone.

- [ ] **Step 5: Gate confirmation.** Because Step 3 proved training tables identical and models are untouched, gates cannot move — run Gate A as a cheap confirmation anyway:

Run: `python scripts/04_backtest.py`
Expected: identical metrics to the last recorded run (RUNS.md): acc ≥0.64, Brier ≤0.225, ECE ≤0.05, all PASS.

- [ ] **Step 6: Live smoke — prediction deltas.** Re-run the prediction half of Task 1's baseline script against the NEW `pre_fight_state`, save as `post_preds.json`, and print a side-by-side delta table for UFC 329.
Expected: probabilities move (that's the point), no errors, no NaN/0.5-collapse rows, and directionally: `P(Garbrandt)` DROPS from 0.636 (his missing fight was bad; Yanez's missing fight was a draw). Any fight whose probability moved >0.15 gets a manual eyeball: pull that fighter's last fight result and confirm the direction makes sense.

- [ ] **Step 7: Commit the regenerated state** (models untracked here; only the parquet whitelisted paths — check `.gitignore` first; if `data/processed/` is not tracked, skip this commit):

Run: `git status --short data/processed/`
If tracked, commit; if ignored, no commit needed (HF deploy in Task 8 carries the file).

---

### Task 6: Divergence flag in /api/market-lines

**Files:**
- Modify: `src/ufc/api/app.py:435-453` (rows dict in `get_market_lines`)
- Test: `tests/test_market_lines.py` (extend — it already covers this endpoint's row shape)

**Interfaces:**
- Consumes: `model_p` and `rq.yes_ask` already in scope at app.py:430-441.
- Produces: each row gains `"divergence": bool` — true when `|modelP − ask| ≥ 0.20`. Frontend (Task 7) renders it.

- [ ] **Step 1: Write the failing test.** Add to `tests/test_market_lines.py` (match the file's existing fixture style — it builds rows via the same helpers; if it tests `evaluate_market_quote` only, add a focused unit test for the row builder's flag logic instead):

```python
def test_divergence_flag_threshold():
    """|model_p - ask| >= 0.20 flags the row; below stays unflagged."""
    from ufc.api.app import _divergence_flag
    assert _divergence_flag(model_p=0.636, ask=0.25) is True    # 38.6pp gap
    assert _divergence_flag(model_p=0.55, ask=0.45) is False    # 10pp gap
    assert _divergence_flag(model_p=0.30, ask=0.52) is True     # negative side too
    assert _divergence_flag(model_p=0.65, ask=0.45) is True     # exactly 0.20 -> flag
```

- [ ] **Step 2: Run to verify it fails:**

Run: `python -m pytest tests/test_market_lines.py -v -k divergence`
Expected: FAIL — `_divergence_flag` doesn't exist.

- [ ] **Step 3: Implement.** In `src/ufc/api/app.py`, add near the other module helpers (above `get_market_lines`):

```python
# Model/market divergence guard (v8.39): a gap this large historically means
# EITHER a real edge OR information the feature store can't see (late injury,
# a last fight whose result contradicts its stat line). Flag, don't hide.
_DIVERGENCE_PP = 0.20


def _divergence_flag(model_p: float, ask: float) -> bool:
    return abs(float(model_p) - float(ask)) >= _DIVERGENCE_PP
```

And add one line to the rows dict (after `"paper": ...,` at line ~451):

```python
            "divergence": _divergence_flag(model_p, rq.yes_ask),
```

- [ ] **Step 4: Run tests:**

Run: `python -m pytest tests/test_market_lines.py tests/test_market_edge.py tests/test_market_advice_gate.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit:**

```bash
git add src/ufc/api/app.py tests/test_market_lines.py
git commit -m "feat(api): divergence flag on market-lines rows (|modelP - ask| >= 20pp)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Divergence badge in the exchange table

**Files:**
- Modify: `frontend/src/exchange.jsx` (badge helper near `_liqBadge` line ~74; render in `ExchangeRow` edge cell line ~126-128)
- Modify: `frontend/src/styles/fightpath.css` (one rule block)

**Interfaces:**
- Consumes: `r.divergence` boolean from Task 6.
- Produces: a `⚠ CHECK TAPE` chip in the edge cell when flagged, with an explanatory tooltip. No layout shift when absent.

- [ ] **Step 1: Add the badge helper** below `_liqBadge` (line ~87) in `frontend/src/exchange.jsx`:

```jsx
function _divergenceBadge(r) {
  if (!r.divergence) return null;
  const gap = Math.abs((r.modelP ?? 0) - (r.ask ?? 0)) * 100;
  return (
    <span
      className="fp-diverge"
      title={`Model and market disagree by ${gap.toFixed(0)}pp. Gaps this large can mean real edge — or information the model can't see (late injury news, a last fight whose result hides a bad performance). Check recent tape before sizing.`}
    >
      ⚠ CHECK TAPE
    </span>
  );
}
```

- [ ] **Step 2: Render it in the edge cell.** In `ExchangeRow`, replace the edge `<td>` (lines 126-128):

```jsx
      <td>
        <span className={"fp-edge-cell " + (edge >= 0 ? "pos" : "neg")}>{edge >= 0 ? "+" : ""}{(edge * 100).toFixed(1)}pp</span>
        {_divergenceBadge(r)}
      </td>
```

- [ ] **Step 3: Style.** Append to `frontend/src/styles/fightpath.css` (match the file's existing badge conventions — look at the `.fp-liq` rules and reuse their pattern/variables):

```css
/* Divergence guard chip — model/market gap ≥20pp; warning, not a skip signal */
.fp-diverge {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 6px;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border-radius: 4px;
  color: #d97706;
  border: 1px solid rgba(217, 119, 6, 0.45);
  background: rgba(217, 119, 6, 0.12);
  cursor: help;
  vertical-align: middle;
}
```

- [ ] **Step 4: Verify in the dev app.** Run `dev.bat`, open the Positions → exchange view for the UFC 329 card. Expected: Garbrandt winner row shows the `⚠ CHECK TAPE` chip (if post-sentinel `modelP` still diverges ≥20pp from the ask) with tooltip; rows under the threshold show nothing; no layout shift.

- [ ] **Step 5: Commit:**

```bash
git add frontend/src/exchange.jsx frontend/src/styles/fightpath.css
git commit -m "feat(ui): CHECK TAPE divergence chip on exchange rows

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Deploy (GATED — requires explicit user go)

**Files:** none (deploy operations only).

Do NOT execute any step here without the user saying so, per CLAUDE.md.

- [ ] **Step 1: Ask the user** for go/no-go with the Task 5 Step 6 delta table in hand (they should see how live predictions will move).
- [ ] **Step 2 (on go): Vercel** — `git push origin main` (frontend + any tracked files).
- [ ] **Step 3 (on go): HuggingFace Space** (`bgelman/fightpath-api`) — small-file commit via `HfApi().create_commit` with `CommitOperationAdd` for: `data/processed/pre_fight_state.parquet`, `src/ufc/features/ratings.py`, `src/ufc/features/assemble.py`, `src/ufc/inference/matchup.py`, `src/ufc/api/app.py`. NEVER `git lfs push --all`. See memory `reference_hf_deploy_process.md`.
- [ ] **Step 4: Post-deploy smoke** — hit `/api/cards` + `/api/market-lines` on the Space; confirm non-error, confirm one known fight's probability matches the local post-change value (process restart also clears the prediction cache).
- [ ] **Step 5: RUNS.md entry** — record as v8.39: sentinel pre_fight_state + rating-pairing parity + divergence chip; gates unchanged (features byte-identical, proof = hash match); serving probs moved (attach the delta summary).

---

## Self-Review Notes

- Spec coverage: sentinel fix (Tasks 2,3,5), training-table protection (Task 3 mask + filter, Task 5 hash proof), serve parity forced by sentinel NaNs (Task 4), divergence guard (Tasks 6,7), deploy discipline (Task 8). Original session finding — "big divergence + missing last fight = phantom edge" — is addressed by removing the lag (root cause) and flagging divergence (residual risk).
- Type consistency: `append_sentinel_rows` + `SENTINEL_KEEP_COLS` names match between Task 3 impl and tests; `_divergence_flag(model_p, ask)` matches Task 6 test; `r.divergence` matches Task 6's row key.
- Known risk, called out: `round_detail.compute_round_features` and `context.compute_context` run with sentinels present; both are believed shift-based/causal like everything else, but Task 5 Steps 2-3 (rebuild + hash identity + no-crash) are the actual proof. If the rebuild crashes on NaN sentinel fields in one of those modules, null-handling belongs in `append_sentinel_rows` (add the offending column to a dtype-aware default), not in the feature module.
