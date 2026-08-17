"""Build feature vectors for hypothetical matchups at inference time."""
from __future__ import annotations

import difflib
import math
from datetime import date

import numpy as np
import pandas as pd

from ufc.io import paths, parquet
from ufc.ingest.parse_helpers import normalize_name
from ufc.features import physical, interactions, style

_NORM_NAME_CACHE: dict[int, pd.Series] = {}

# ---------------------------------------------------------------------------
# Career-average feature columns: maps scrape_fighter_stats.py output columns
# to the pre_fight_state rolling-window columns they fill when NaN.
# All *_flavors are filled with the same career-average value.
# ---------------------------------------------------------------------------
_SLPM_COLS  = ["slpm_ctd", "slpm_l3", "slpm_l5", "slpm_2y", "slpm_decay"]
_SAPM_COLS  = ["sapm_ctd", "sapm_l3", "sapm_l5", "sapm_2y", "sapm_decay"]
_TD_RATE_COLS = ["td_per_15_ctd", "td_per_15_l3", "td_per_15_l5", "td_per_15_2y", "td_per_15_decay"]
_TD_ACC_COLS  = ["td_acc_ctd", "td_acc_l3", "td_acc_l5", "td_acc_2y", "td_acc_decay"]
_TD_DEF_COLS  = ["td_def_ctd", "td_def_l3", "td_def_l5", "td_def_2y", "td_def_decay"]
_STR_ACC_COLS = ["str_acc_ctd", "str_acc_l3", "str_acc_l5", "str_acc_2y", "str_acc_decay"]
_STR_DEF_COLS = ["str_def_ctd", "str_def_l3", "str_def_l5", "str_def_2y", "str_def_decay"]
_SUB_COLS     = ["sub_att_per_15_ctd", "sub_att_per_15_l3", "sub_att_per_15_l5",
                 "sub_att_per_15_2y", "sub_att_per_15_decay"]

_CAREER_COL_GROUPS: list[tuple[str, list[str]]] = [
    ("slpm",    _SLPM_COLS),
    ("sapm",    _SAPM_COLS),
    ("td_avg",  _TD_RATE_COLS),
    ("td_acc",  _TD_ACC_COLS),
    ("td_def",  _TD_DEF_COLS),
    ("str_acc", _STR_ACC_COLS),
    ("str_def", _STR_DEF_COLS),
    ("sub_avg", _SUB_COLS),
]

_career_stats_cache: pd.DataFrame | None = None


def _load_career_stats() -> pd.DataFrame | None:
    global _career_stats_cache
    if _career_stats_cache is not None:
        return _career_stats_cache
    csv_path = paths.raw_scraper() / "ufc_fighter_career_stats.csv"
    if not csv_path.exists():
        return None
    try:
        _career_stats_cache = pd.read_csv(csv_path, dtype={"fighter_id": str})
        return _career_stats_cache
    except Exception:
        return None


def _fill_thin_from_career(state: pd.Series, fighter_id: str) -> pd.Series:
    """Fill NaN rolling features from ufcstats career-average block.

    Only overwrites columns that are currently NaN so genuine UFC fight
    history is never replaced.  Requires 08_scrape_fighter_stats.py to have
    been run; silently skips if the CSV is absent or the fighter is not found.
    """
    career_df = _load_career_stats()
    if career_df is None:
        return state
    row = career_df[career_df["fighter_id"] == fighter_id]
    if row.empty:
        return state
    career = row.iloc[0]
    state = state.copy()
    for career_col, feature_cols in _CAREER_COL_GROUPS:
        val = career.get(career_col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        val = float(val)
        for fc in feature_cols:
            if fc in state.index and pd.isna(state[fc]):
                state[fc] = val
    return state


def find_fighter(name: str, fighters_df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return (fighter_id, canonical_name) for a given name string.

    Tries exact normalized match first, then fuzzy.
    Raises ValueError with top 5 suggestions if not found.
    """
    norm = normalize_name(name)
    # Cache normalized names by fighters_df identity (re-used across calls in a session)
    key = id(fighters_df)
    if key not in _NORM_NAME_CACHE or len(_NORM_NAME_CACHE[key]) != len(fighters_df):
        _NORM_NAME_CACHE[key] = fighters_df["fighter_name"].apply(normalize_name)
    fighters_df = fighters_df.copy()
    fighters_df["norm_name"] = _NORM_NAME_CACHE[key].values

    # Exact match
    match = fighters_df[fighters_df["norm_name"] == norm]
    if len(match) == 1:
        return match.iloc[0]["fighter_id"], match.iloc[0]["fighter_name"]
    if len(match) > 1:
        # Pick the most recently active fighter by latest event_date in pre_fight_state
        try:
            pfs = parquet.read(paths.processed("pre_fight_state"))
            last_date = (
                pfs[pfs["fighter_id"].isin(match["fighter_id"])]
                .groupby("fighter_id")["event_date"]
                .max()
            )
            best_id = last_date.idxmax()
            row = match[match["fighter_id"] == best_id].iloc[0]
            print(f"  Note: '{name}' is ambiguous ({len(match)} fighters); "
                  f"selected most recent: {row['fighter_name']} "
                  + (f"dob={row['dob']}" if "dob" in match.columns else ""))
            return row["fighter_id"], row["fighter_name"]
        except Exception:
            pass
        # Fallback: raise with candidates listed
        details = match[["fighter_id", "fighter_name"]].copy()
        if "dob" in match.columns:
            details["dob"] = match["dob"]
        raise ValueError(
            f"Ambiguous name '{name}' — {len(match)} fighters match '{norm}'.\n"
            f"Disambiguate by passing fighter_id directly. Candidates:\n"
            + "\n".join(f"  {r.fighter_id}  {r.fighter_name}"
                       + (f"  dob={r.dob}" if "dob" in details.columns else "")
                       for r in details.itertuples(index=False))
        )

    # Fuzzy match
    # name_norm is a leaf module (stdlib only, no ufc.inference imports), so a
    # top-level import is safe; kept local anyway to avoid coupling this hot path
    # to the ingest package's import graph.
    from ufc.ingest.name_norm import (
        normalize as _normalize, token_surname_match as _token_surname_match,
    )
    all_norms = fighters_df["norm_name"].tolist()
    close = difflib.get_close_matches(norm, all_norms, n=5, cutoff=0.6)
    # Guard: difflib's char-similarity metric alone can match two different real
    # fighters (e.g. "John Garza" -> "Jason Glaza"). Require the surname-aware
    # matcher to also agree before accepting a candidate. _token_surname_match
    # expects name_norm.normalize'd strings (space-preserving), which differs
    # from this file's normalize_name (punctuation-stripping) on hyphenated names
    # — so re-normalize both sides with _normalize for the guard check only.
    api_norm = _normalize(name)
    close = [c for c in close
             if _token_surname_match(api_norm, _normalize(
                 fighters_df[fighters_df["norm_name"] == c].iloc[0]["fighter_name"]))]
    if close:
        best = close[0]
        row = fighters_df[fighters_df["norm_name"] == best].iloc[0]
        suggestions = ", ".join([fighters_df[fighters_df["norm_name"] == c].iloc[0]["fighter_name"]
                                 for c in close if c != best][:4])
        print(f"  Note: '{name}' matched to '{row['fighter_name']}' (fuzzy)")
        if suggestions:
            print(f"  Other candidates: {suggestions}")
        return row["fighter_id"], row["fighter_name"]

    # Not found — show suggestions
    close_any = difflib.get_close_matches(norm, all_norms, n=5, cutoff=0.4)
    suggestions = [fighters_df[fighters_df["norm_name"] == c].iloc[0]["fighter_name"]
                   for c in close_any if len(fighters_df[fighters_df["norm_name"] == c]) > 0]
    raise ValueError(
        f"Fighter '{name}' not found in database.\n"
        f"Top suggestions: {suggestions or 'none'}"
    )


def build_matchup_features(
    red_id: str,
    blue_id: str,
    event_date: date,
    scheduled_rounds: int = 3,
    is_title: bool = False,
    pre_fight_state: pd.DataFrame | None = None,
    fighters_df: pd.DataFrame | None = None,
    location: str = "",
    referee: str = "",
    ref_history_df: pd.DataFrame | None = None,
    weight_class: str | None = None,
) -> pd.DataFrame:
    """Build a single-row wide feature DataFrame for predict.py.

    Uses pre_fight_state (latest known state per fighter) for rolling features.
    """
    if pre_fight_state is None:
        pre_fight_state = parquet.read(paths.processed("pre_fight_state"))
    if fighters_df is None:
        fighters_df = parquet.read(paths.interim("fighters"))

    pf = pre_fight_state.copy()
    pf["event_date"] = pd.to_datetime(pf["event_date"])

    def _get_state(fighter_id: str) -> pd.Series:
        row = pf[pf["fighter_id"] == fighter_id]
        if len(row) > 0:
            return row.iloc[0]

        # Unknown fighter (UFC debutant or scraping miss) — build a neutral
        # default Series with type-appropriate fallbacks so downstream string
        # ops don't crash and weight-class lookups produce sensible neutrals.
        print(f"  [warn] fighter '{fighter_id}' not in pre_fight_state — using neutral defaults")
        defaults = {}
        for c in pf.columns:
            dt = pf[c].dtype
            if pd.api.types.is_numeric_dtype(dt):
                defaults[c] = np.nan
            elif c == "stance":
                defaults[c] = "ORTHO"
            elif c == "weight_class":
                defaults[c] = "Unknown"
            elif pd.api.types.is_datetime64_any_dtype(dt):
                defaults[c] = pd.NaT
            else:
                defaults[c] = None
        defaults["fighter_id"] = fighter_id
        # Seed rating features to their true debut defaults (1500.0 / PHI0=350.0).
        # fillna(0) in winner.py would otherwise send "0 Elo" through monotone
        # constraints as "1500 pts below average" — catastrophically wrong.
        for _rc in ("elo_pre", "glicko_mu_pre", "opp_elo_pre"):
            defaults[_rc] = 1500.0
        defaults["elo_diff"] = 0.0
        defaults["glicko_rd_pre"] = 350.0  # PHI0 — high uncertainty debut
        return pd.Series(defaults, name=fighter_id)

    state_a = _get_state(red_id)
    state_b = _get_state(blue_id)

    # Refresh time-varying fields from fighters_df + event_date so they're current,
    # not stuck at the last-fight snapshot.
    def _refresh_bio(state, fid):
        bio = fighters_df[fighters_df["fighter_id"] == fid]
        if len(bio) == 0:
            return state
        b = bio.iloc[0]
        state = state.copy()
        if pd.notna(b.get("dob")):
            dob = b["dob"]
            ed = pd.Timestamp(event_date) if not isinstance(event_date, pd.Timestamp) else event_date
            state["age_years"] = (ed - pd.Timestamp(dob)).days / 365.25
        for f in ("reach_in", "height_in", "weight_lbs", "stance"):
            v = b.get(f)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                state[f] = v
        return state

    state_a = _refresh_bio(state_a, red_id)
    state_b = _refresh_bio(state_b, blue_id)

    def _refresh_layoff(state, fid):
        last = pf[pf["fighter_id"] == fid]["event_date"].max()
        if pd.notna(last):
            ed = pd.Timestamp(event_date) if not isinstance(event_date, pd.Timestamp) else event_date
            state = state.copy()
            state["layoff_days"] = max(0, (ed - last).days)
        return state

    state_a = _refresh_layoff(state_a, red_id)
    state_b = _refresh_layoff(state_b, blue_id)

    # Fill thin-data fighters' NaN rolling features from career-average block.
    # Requires 08_scrape_fighter_stats.py to have been run; silent no-op otherwise.
    state_a = _fill_thin_from_career(state_a, red_id)
    state_b = _fill_thin_from_career(state_b, blue_id)

    # Fill missing reach_in with division mean from pfs — reach is the #1 gain
    # feature; fillna(0) otherwise gives "0-inch reach" vs ~70", crushing fighters
    # whose reach isn't in ufcstats (common for new/regional-only fighters).
    def _fill_reach_division_mean(state: pd.Series) -> pd.Series:
        if pd.notna(state.get("reach_in")):
            return state
        wc_key = state.get("weight_class", "Unknown")
        # Canonical weight-class names only (strip title/tournament variants)
        wc_reach = pf[pf["weight_class"] == wc_key]["reach_in"].dropna()
        if len(wc_reach) == 0:
            wc_reach = pf["reach_in"].dropna()
        mean_reach = float(wc_reach.mean())
        state = state.copy()
        state["reach_in"] = mean_reach
        return state

    state_a = _fill_reach_division_mean(state_a)
    state_b = _fill_reach_division_mean(state_b)

    # Build wide row
    row = {"fight_id": "inference", "event_date": pd.Timestamp(event_date)}
    row["scheduled_rounds"] = scheduled_rounds
    row["is_title"] = is_title
    row["fighter_id_a"] = red_id
    row["fighter_id_b"] = blue_id

    # All features from pre_fight_state
    exclude_from_features = {
        "fight_id", "event_id", "event_date", "event_rank",
        "fighter_id", "opponent_id", "won", "method", "end_round",
        "end_time_sec", "total_fight_sec",
        "sig_str_landed", "sig_str_attempted", "sig_str_absorbed_landed",
        "sig_str_absorbed_attempted", "td_landed", "td_attempted",
        "td_absorbed_landed", "td_absorbed_attempted",
        "ctrl_sec", "ctrl_sec_absorbed", "kd_for", "kd_against",
        "sub_att_for", "sub_att_against", "rev_for", "rev_against",
        "r1_sig_str_landed",   # parity with training exclude list
        "head_landed", "body_landed", "leg_landed", "distance_landed",
        "clinch_landed", "ground_landed", "head_absorbed", "body_absorbed",
        "leg_absorbed", "distance_absorbed", "clinch_absorbed", "ground_absorbed",
        "head_attempted", "body_attempted", "leg_attempted",
        "distance_attempted", "clinch_attempted", "ground_attempted",
        "fight_min", "weight_class",  # keep as categorical
        "injury_freak",  # post-fight outcome descriptor (parity with base.py exclude)
    }

    for col in pf.columns:
        if col in exclude_from_features:
            continue
        row[f"{col}_a"] = state_a.get(col, np.nan)
        row[f"{col}_b"] = state_b.get(col, np.nan)

    # Passthrough non-feature fields
    row["age_years_a"] = state_a.get("age_years", np.nan)
    row["age_years_b"] = state_b.get("age_years", np.nan)
    row["reach_in_a"] = state_a.get("reach_in", np.nan)
    row["reach_in_b"] = state_b.get("reach_in", np.nan)
    row["height_in_a"] = state_a.get("height_in", np.nan)
    row["height_in_b"] = state_b.get("height_in", np.nan)
    row["weight_lbs_a"] = state_a.get("weight_lbs", np.nan)
    row["weight_lbs_b"] = state_b.get("weight_lbs", np.nan)
    row["stance_a"] = state_a.get("stance", "UNKNOWN")
    row["stance_b"] = state_b.get("stance", "UNKNOWN")

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

    # Altitude from features.yaml lookup
    import yaml
    with open(paths.root() / "configs" / "features.yaml") as f:
        alt_lookup = yaml.safe_load(f).get("altitude_lookup", {})
    altitude = 0.0
    if location:
        for city, alt in alt_lookup.items():
            if city.lower() in location.lower():
                altitude = float(alt)
                break
    row["altitude_meters"] = altitude

    # Era + weight-class baselines — read latest values from features_props (written by 02_build_features).
    # features_props contains these fight-level columns and is always up to date post-feature-build.
    try:
        _era_cols = ["event_date", "weight_class",
                     "era_avg_sig_str_l12mo", "wc_finish_share_l2y", "wc_5rd_dec_rate"]
        _fp = parquet.read(paths.processed("features_props"))
        _avail = [c for c in _era_cols if c in _fp.columns]
        if len(_avail) >= 3:
            baselines = _fp[_avail].drop_duplicates(["event_date", "weight_class"]).sort_values("event_date")
            latest_era = float(baselines["era_avg_sig_str_l12mo"].dropna().iloc[-1])
        else:
            baselines = None
            latest_era = 45.0
    except Exception:
        baselines = None
        latest_era = 45.0  # ~2024 era fallback
    row["era_avg_sig_str_l12mo"] = latest_era

    # Per-weight-class lookups (use latest available for the matchup's weight class)
    inferred_wc = state_a.get("weight_class") or state_b.get("weight_class") or "Unknown"

    # Catch-weight sentinel: non-canonical string (e.g. "Catch Weight") supplied by the user.
    # Canonical/None paths stay byte-identical; only this new branch fires on non-canonical.
    # weight_class_lbs cleans noisy labels ("UFC Interim Heavyweight Title" etc.) before
    # the lbs lookup — raw .get() silently failed for 172 pre-fight-state fighters (C-1).
    from ufc.features.weight_class import _WC_WEIGHT_LBS, weight_class_lbs
    is_catch = bool(weight_class) and weight_class_lbs(weight_class) is None

    if is_catch:
        # Resolve the nearest canonical class by averaging the two fighters' native poundages.
        a_native_lbs = weight_class_lbs(state_a.get("weight_class"))
        b_native_lbs = weight_class_lbs(state_b.get("weight_class"))
        if a_native_lbs is not None or b_native_lbs is not None:
            effective_lbs = (
                (a_native_lbs + b_native_lbs) / 2
                if a_native_lbs is not None and b_native_lbs is not None
                else (a_native_lbs or b_native_lbs)
            )
            # Pick gender pool: use women's names only when fighter A is native women's.
            from ufc.features.weight_class import clean_weight_class as _clean_wc
            _is_womens = _clean_wc(state_a.get("weight_class", "")).startswith("Women's")
            _candidate_wcs = {k: v for k, v in _WC_WEIGHT_LBS.items()
                              if k.startswith("Women's") == _is_womens}
            wc = min(_candidate_wcs, key=lambda k: abs(_candidate_wcs[k] - effective_lbs))
        else:
            wc = inferred_wc
    else:
        wc = weight_class if weight_class else inferred_wc

    if baselines is not None:
        try:
            wc_rows = baselines[baselines["weight_class"] == wc].dropna(subset=["wc_finish_share_l2y"])
            row["wc_finish_share_l2y"] = float(wc_rows["wc_finish_share_l2y"].iloc[-1]) if len(wc_rows) else 0.5
            wc_rows2 = baselines[baselines["weight_class"] == wc].dropna(subset=["wc_5rd_dec_rate"])
            row["wc_5rd_dec_rate"] = float(wc_rows2["wc_5rd_dec_rate"].iloc[-1]) if len(wc_rows2) else 0.55
        except Exception:
            row["wc_finish_share_l2y"] = 0.5
            row["wc_5rd_dec_rate"] = 0.55
    else:
        row["wc_finish_share_l2y"] = 0.5
        row["wc_5rd_dec_rate"] = 0.55

    # When an explicit canonical weight class is supplied, recompute the move-up/down delta
    # so it reflects the actual bout division rather than the stale pre-fight-state value.
    if weight_class and not is_catch and weight_class_lbs(weight_class) is not None:
        bout_lbs = weight_class_lbs(weight_class)
        for sfx, st_ in (("a", state_a), ("b", state_b)):
            native_lbs = weight_class_lbs(st_.get("weight_class"))
            if native_lbs is not None:
                row[f"weight_class_change_lbs_{sfx}"] = bout_lbs - native_lbs
    elif is_catch:
        # For catch-weight bouts, anchor each fighter's delta to the effective poundage.
        a_native_lbs = weight_class_lbs(state_a.get("weight_class"))
        b_native_lbs = weight_class_lbs(state_b.get("weight_class"))
        _eff = (
            (a_native_lbs + b_native_lbs) / 2
            if a_native_lbs is not None and b_native_lbs is not None
            else (a_native_lbs or b_native_lbs)
        )
        if _eff is not None:
            if a_native_lbs is not None:
                row["weight_class_change_lbs_a"] = _eff - a_native_lbs
            if b_native_lbs is not None:
                row["weight_class_change_lbs_b"] = _eff - b_native_lbs

    # Referee stoppage tendency — look up from history, lazy-loading if not supplied.
    # Default = population mean (~0.5) so unknown referee is neutral, not floor-biased.
    ref_thresh = 0.5
    if ref_history_df is None:
        from ufc.inference.ref_history import get_ref_history
        ref_history_df = get_ref_history()
    if ref_history_df is not None and len(ref_history_df) and "referee_stoppage_threshold" in ref_history_df.columns:
        ref_thresh = float(ref_history_df["referee_stoppage_threshold"].mean())
        if referee:
            norm = str(referee).strip().lower()
            if "_ref_norm" in ref_history_df.columns:
                ref_rows = ref_history_df[ref_history_df["_ref_norm"] == norm]
            else:
                ref_rows = ref_history_df[ref_history_df["referee"] == referee]
            if len(ref_rows):
                ref_thresh = float(ref_rows["referee_stoppage_threshold"].mean())
    row["referee_stoppage_threshold"] = ref_thresh

    df = pd.DataFrame([row])
    df = physical.compute_physical(df)
    df = interactions.compute_interactions(df)

    # Apply PCA style components from saved artifact (or zero if not available)
    pca_files = sorted(paths.outputs_models().glob("pca_style_*.joblib"))
    if pca_files:
        df = style.apply_style_pca(df, pca_files[-1])
    else:
        for i in range(1, 6):
            df[f"style_pc{i}"] = 0.0

    return df
