"""Singleton model/reference loader for the FastAPI service.

Call `startup()` once at process launch (FastAPI lifespan).
Everything else imports from this module's cached state.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import date
from typing import Any

import pandas as pd

_lock = threading.Lock()
_state: dict[str, Any] = {}

# Process-lifetime prediction cache — see predict()'s docstring for why this
# is safe (model artifacts are fixed for the life of a process; a redeploy or
# --reload restart always gets a brand-new empty cache along with them).
_PREDICT_CACHE_MAX = 200
_predict_cache: "OrderedDict[tuple, Any]" = OrderedDict()


def _predict_cache_key(red, blue, rounds, is_title, event_date, weight_class, referee, location, n_simulate) -> tuple:
    return (red, blue, rounds, is_title, event_date, weight_class, referee or "", location or "", n_simulate)


def startup() -> None:
    """Load core models and reference data at startup.

    Prop models (sig_strikes, takedowns, r1_sig_strikes, duration) are deferred
    to first use to stay within the 512 MB Render free-tier limit.
    """
    from ufc.inference.predict_core import _find_latest_model, load_reference_data
    from ufc.models.winner import WinnerModel
    from ufc.models.method import MethodClassifier

    print("[service] Loading core models…")
    models: dict[str, Any] = {}

    winner_path = _find_latest_model("winner_ensemble_*.joblib")
    if winner_path is None:
        raise RuntimeError(
            "No winner_ensemble_*.joblib found under outputs/models/ (or prod/). "
            "Startup aborted rather than silently serving prob=0.5 — check LFS "
            "smudge / checkout before serving."
        )
    print(f"  Loading winner model: {winner_path.name}")
    models["winner"] = WinnerModel.load(winner_path)

    method_path = _find_latest_model("method_clf_*.joblib")
    if method_path is None:
        raise RuntimeError(
            "No method_clf_*.joblib found under outputs/models/ (or prod/). "
            "Startup aborted rather than silently serving base-rate methods — "
            "check LFS smudge / checkout before serving."
        )
    print(f"  Loading method model: {method_path.name}")
    models["method"] = MethodClassifier.load(method_path)

    print("[service] Loading reference data…")
    fighters_df, pre_fight_state, ref_history_df = load_reference_data()

    # Build fighter UFC-record lookup from ledger (W, L, D per fighter_id)
    records: dict[str, tuple[int, int, int]] = {}
    try:
        from ufc.io import paths as _paths
        ledger_path = _paths.processed("ledger")
        ledger_df = pd.read_parquet(ledger_path)
        for fid, grp in ledger_df.groupby("fighter_id"):
            w = int((grp["won"] == 1).sum())
            l = int((grp["won"] == 0).sum())
            d = int(grp["won"].isna().sum())
            records[str(fid)] = (w, l, d)
        print(f"[service] Loaded UFC records for {len(records)} fighters.")
    except Exception as exc:
        print(f"[service] Warning: could not build fighter records: {exc}")

    # Overlay full MMA records from career stats CSV (wins_total/losses_total/draws_total)
    try:
        from ufc.io import paths as _paths2
        csv_path = _paths2.raw_scraper() / "ufc_fighter_career_stats.csv"
        if csv_path.exists():
            cs = pd.read_csv(csv_path, dtype={"fighter_id": str})
            for _, row in cs.iterrows():
                fid = str(row["fighter_id"])
                w = row.get("wins_total")
                l = row.get("losses_total")
                d = row.get("draws_total")
                if pd.notna(w) and pd.notna(l):
                    records[fid] = (int(w), int(l), int(d) if pd.notna(d) else 0)
            print(f"[service] Overlaid full MMA records from career stats CSV.")
    except Exception as exc:
        print(f"[service] Warning: could not overlay full MMA records: {exc}")

    with _lock:
        _state["models"] = models
        _state["fighters_df"] = fighters_df
        _state["pre_fight_state"] = pre_fight_state
        _state["ref_history_df"] = ref_history_df
        _state["records"] = records
        _state["history_cache"] = None
        _state["prop_models_loaded"] = False
    print("[service] Ready.")


def _ensure_prop_models() -> None:
    """Lazily load large prop models on first prop prediction request."""
    with _lock:
        if _state.get("prop_models_loaded"):
            return
        # Load inside the lock — blocks concurrent requests briefly but prevents
        # double-loading on the free tier (which would definitely OOM).
        from ufc.inference.predict_core import _find_latest_model
        from ufc.models.props_count import HurdleCountModel
        from ufc.models.props_duration import DurationModel

        print("[service] Loading prop models (first prop request)…")
        models = _state["models"]

        for key, pattern, loader in [
            ("sig_strikes",    "props_sig_strikes_*.joblib",    HurdleCountModel.load),
            ("takedowns",      "props_takedowns_*.joblib",       HurdleCountModel.load),
            ("r1_sig_strikes", "props_r1_sig_strikes_*.joblib",  HurdleCountModel.load),
            ("knockdowns",     "props_knockdowns_*.joblib",      HurdleCountModel.load),
            ("sub_attempts",   "props_sub_attempts_*.joblib",    HurdleCountModel.load),
            ("r1_takedowns",   "props_r1_takedowns_*.joblib",    HurdleCountModel.load),
            ("body_sig_strikes","props_body_sig_strikes_*.joblib",HurdleCountModel.load),
            ("leg_sig_strikes", "props_leg_sig_strikes_*.joblib", HurdleCountModel.load),
            ("ctrl_time",      "props_ctrl_time_*.joblib",       HurdleCountModel.load),
            ("duration",       "props_duration_*.joblib",        DurationModel.load),
        ]:
            path = _find_latest_model(pattern)
            if path:
                print(f"  Loading {key}: {path.name}")
                models[key] = loader(path)

        _state["prop_models_loaded"] = True
        print("[service] Prop models loaded.")


def _get(key: str) -> Any:
    v = _state.get(key)
    if v is None:
        raise RuntimeError(f"service.startup() not called (missing '{key}')")
    return v


def predict(
    red: str,
    blue: str,
    rounds: int,
    is_title: bool,
    event_date: date,
    weight_class: str | None = None,
    referee: str = "",
    location: str = "",
    n_simulate: int = 50000,
):
    """Run the full prediction pipeline with cached models/reference data.

    Results are cached for the process lifetime (bounded LRU, see
    _PREDICT_CACHE_MAX), keyed on every argument that can change the answer.
    This is safe because the model artifacts themselves are fixed for the
    life of a process — they only change on a fresh deploy or --reload
    restart, both of which always start with a brand-new empty cache too.
    The only thing that's genuinely different call-to-call is Monte Carlo
    noise inside sim_samples, which no caller depends on being fresh.

    Callers must treat the returned FightPrediction as read-only: it may be
    the same object handed to a previous caller, and will be handed to future
    ones. (Verified: no downstream code — app.py, serialize.py, prop_cdf.py,
    market_edge.py — mutates it; the only attribute writes happen below,
    before the object is cached.)

    A simultaneous first request for the exact same never-yet-cached fight
    from two threads may compute it twice (the lock isn't held across the
    actual compute) — harmless, just a few duplicated seconds of CPU on a
    rare race, never a correctness issue.
    """
    key = _predict_cache_key(red, blue, rounds, is_title, event_date, weight_class, referee, location, n_simulate)
    with _lock:
        cached = _predict_cache.get(key)
        if cached is not None:
            _predict_cache.move_to_end(key)
            return cached

    _ensure_prop_models()
    from ufc.inference.predict_core import predict_fight
    pred = predict_fight(
        red_name=red,
        blue_name=blue,
        rounds=rounds,
        is_title=is_title,
        event_date=event_date,
        models=_get("models"),
        fighters_df=_get("fighters_df"),
        pre_fight_state=_get("pre_fight_state"),
        n_simulate=n_simulate,
        location=location,
        referee=referee,
        ref_history_df=_get("ref_history_df"),
        run_simulation=True,
        verbose=False,
        weight_class=weight_class,
    )
    recs = _state.get("records", {})
    pred.record_red  = recs.get(str(pred.red_id))
    pred.record_blue = recs.get(str(pred.blue_id))

    with _lock:
        _predict_cache[key] = pred
        _predict_cache.move_to_end(key)
        while len(_predict_cache) > _PREDICT_CACHE_MAX:
            _predict_cache.popitem(last=False)

    return pred


def readiness() -> dict[str, bool]:
    """Per-component readiness booleans for /api/ready.

    winner/method/reference_data are required — startup() now raises on
    missing artifacts, so these should always be True once the process is
    actually serving requests. props_loaded is lazy (see _ensure_prop_models,
    deferred to the first prop request to stay within the free-tier memory
    limit), so it legitimately starts False.
    """
    models = _state.get("models") or {}
    return {
        "winner": "winner" in models,
        "method": "method" in models,
        "props_loaded": bool(_state.get("prop_models_loaded")),
        "reference_data": _state.get("fighters_df") is not None,
    }


def get_fighters_df() -> pd.DataFrame:
    return _get("fighters_df")


def get_ref_history_df() -> pd.DataFrame:
    return _get("ref_history_df")


def get_models() -> dict:
    return _get("models")


def get_history_feed(n_events: int = 10) -> dict:
    """Return per-event predicted-vs-actual rows plus full-DB totals.

    Computed once and cached for the process lifetime.
    """
    with _lock:
        cached = _state.get("history_cache")
    if cached is not None:
        return cached

    feed = _build_history_feed(n_events)
    with _lock:
        _state["history_cache"] = feed
    return feed


def _load_eval_winner():
    """EVAL-tier winner (locked train ≤2023) for HONEST held-out History numbers.

    The served/prod winner trains on ALL data → it has no honest test set
    (evaluating it on 2025-26 = in-sample ≈ 100%, the "99.9% hit rate" bug).
    The eval winner genuinely never saw 2024-26, so its accuracy there is real.
    Load from the eval dir explicitly (NOT the prod-preferring serving loader).
    Falls back to the served winner only if no eval winner is on disk.
    """
    from ufc.io import paths
    from ufc.models.winner import WinnerModel
    files = sorted(paths.outputs_models().glob("winner_ensemble_*.joblib"),
                   key=lambda p: p.stat().st_mtime)
    if files:
        return WinnerModel.load(files[-1])
    return get_models().get("winner")


def _norm_name(s) -> str:
    """Loose key for matching a fighter across the log and the scrape.

    They disagree on punctuation and suffixes ("Khalil Rountree Jr." vs
    "Khalil Rountree Jr"), so drop everything but letters and digits.
    """
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _build_history_feed(n_events: int) -> dict:
    from ufc.io import paths, parquet
    from ufc.training.splits import get_splits
    from ufc.training.symmetrize import symmetrize

    winner_model = _load_eval_winner()
    if winner_model is None:
        return {"events": [], "totals": {}}

    try:
        features_winner = parquet.read(paths.processed("features_winner"))
    except Exception:
        return {"events": [], "totals": {}}

    features_winner["event_date"] = pd.to_datetime(features_winner["event_date"])
    splits = get_splits(features_winner)
    all_hist = features_winner.dropna(subset=["won_a"]).copy()
    if len(all_hist) == 0:
        return {"events": [], "totals": {}}

    # Full-database counts (whole ledger — honest, just a size)
    total_db_events = int(all_hist["event_date"].nunique())
    total_db_fights  = len(all_hist)

    # ── Name + event/card-order maps (used for keys and the feed) ───────────
    fighters_df = get_fighters_df()
    id_to_name: dict[str, str] = {}
    name_col = next((c for c in ("fighter_name", "name") if c in fighters_df.columns), None)
    if "fighter_id" in fighters_df.columns and name_col:
        id_to_name = dict(zip(fighters_df["fighter_id"].astype(str),
                              fighters_df[name_col].astype(str)))

    def _name(fid: str | None, fallback: str) -> str:
        if fid and fid in id_to_name:
            return id_to_name[fid]
        return fallback or str(fid or "?")

    def _row_names(row) -> tuple[str, str]:
        red = (row.get("fighter_a_name")
               or _name(row.get("fighter_id_a") or row.get("fighter_a_id"), row.get("fighter_a", "")))
        blue = (row.get("fighter_b_name")
                or _name(row.get("fighter_id_b") or row.get("fighter_b_id"), row.get("fighter_b", "")))
        return str(red), str(blue)

    event_id_to_name: dict[str, str] = {}
    # Authoritative name per DATE. The live prediction log freezes event_name at
    # prediction time, so a card renamed after a main-event change (Ankalaev vs.
    # Rountree Jr -> vs. Guskov, 2026-07-25) keeps the stale booking name on every
    # row logged before the switch. events_interim is re-scraped, so it is current.
    event_date_to_name: dict[str, str] = {}
    try:
        events_interim = parquet.read(paths.interim("events"))
        if "event_id" in events_interim.columns and "event_name" in events_interim.columns:
            event_id_to_name = dict(zip(events_interim["event_id"].astype(str),
                                        events_interim["event_name"].astype(str)))
        if "event_date" in events_interim.columns and "event_name" in events_interim.columns:
            event_date_to_name = {
                str(d)[:10]: str(n)
                for d, n in zip(events_interim["event_date"], events_interim["event_name"])
            }
    except Exception:
        pass

    fight_id_to_pos: dict[str, int] = {}
    # Same ordering, keyed by the fighter pair instead of fight_id — the live log
    # stores names, not fight ids. Order-agnostic (red/blue can swap between the
    # log and the scrape), and interim row order is card order (main event first).
    pair_to_pos: dict[frozenset, int] = {}
    try:
        fights_interim = parquet.read(paths.interim("fights"))
        if "fight_id" in fights_interim.columns and "event_name" in fights_interim.columns:
            for _, ev_grp in fights_interim.groupby("event_name"):
                for pos, fid in enumerate(ev_grp["fight_id"]):
                    fight_id_to_pos[str(fid)] = pos
        if {"fighter_a_name", "fighter_b_name", "event_name"} <= set(fights_interim.columns):
            for _, ev_grp in fights_interim.groupby("event_name"):
                for pos, (a, b) in enumerate(zip(ev_grp["fighter_a_name"],
                                                 ev_grp["fighter_b_name"])):
                    pair_to_pos[frozenset((_norm_name(a), _norm_name(b)))] = pos
    except Exception:
        pass

    def _events_from(df: "pd.DataFrame") -> list[dict]:
        """Group a scored frame (with _pred_prob_a, _red, _blue) into event cards."""
        out: list[dict] = []
        df = df.copy()
        df["event_date"] = pd.to_datetime(df["event_date"])
        for ev_date, grp in df.sort_values("event_date").groupby("event_date"):
            fights_out, n_correct = [], 0
            for _, row in grp.iterrows():
                p_a = float(row["_pred_prob_a"])
                actual_win_a = bool(row["won_a"])
                pred_win_a = p_a >= 0.5
                red, blue = row["_red"], row["_blue"]
                correct = pred_win_a == actual_win_a
                n_correct += int(correct)
                fights_out.append({
                    "red": red, "blue": blue, "pRed": round(p_a, 3),
                    "predWinner": red if pred_win_a else blue,
                    "actualWinner": red if actual_win_a else blue,
                    "correct": correct,
                    "_cardPos": fight_id_to_pos.get(str(row.get("fight_id", "")), 9999),
                })
            fights_out.sort(key=lambda f: f["_cardPos"])
            for f in fights_out:
                f.pop("_cardPos", None)
            event_id = str(grp["event_id"].iloc[0]) if "event_id" in grp.columns else ""
            ev_name = event_id_to_name.get(event_id) or _infer_event_name(grp)
            total = len(fights_out)
            out.append({
                "id": ev_date.strftime("%Y-%m-%d"), "event": ev_name,
                "date": ev_date.strftime("%Y-%m-%d"),
                "correct": n_correct, "total": total,
                "hitRate": round(n_correct / total, 3) if total else 0.0,
                "fights": fights_out,
            })
        return out

    # ── Live (prod) forward record + the keys it has "taken over" ───────────
    # Single accumulating tally (user's choice, knowingly mixing two models):
    #   • everything up to "now" = eval model held-out backtest (the baseline)
    #   • every card from now on = the served prod model, logged pre-fight + graded
    # A fight is counted ONCE: if it's in the live log it counts as prod (live),
    # otherwise it counts as eval — so a card never double-counts when it both
    # enters the eval test window AND gets graded by prod.
    from ufc.inference.prediction_log import build_live_record, load_log, matchup_key
    live = build_live_record(n_events=n_events)
    try:
        live_keys = set(load_log()["key"].tolist())
    except Exception:
        live_keys = set()

    # ── Eval model on held-out test split, excluding live-logged fights ─────
    eval_correct = eval_fights = 0
    eval_feed: list[dict] = []
    test_df = features_winner[splits["test"]].dropna(subset=["won_a"]).copy().reset_index(drop=True)
    if len(test_df) > 0:
        test_sym = symmetrize(test_df)
        n_test = len(test_df)
        p_test = winner_model.predict_proba(test_sym)
        test_df["_pred_prob_a"] = (p_test[:n_test] + (1.0 - p_test[n_test:])) / 2.0
        names = [_row_names(r) for _, r in test_df.iterrows()]
        test_df["_red"] = [n[0] for n in names]
        test_df["_blue"] = [n[1] for n in names]
        test_df["_key"] = [
            matchup_key(d.strftime("%Y-%m-%d"), rd, bl)
            for d, (rd, bl) in zip(pd.to_datetime(test_df["event_date"]), names)
        ]
        eval_df = test_df[~test_df["_key"].isin(live_keys)].copy()
        ec = (eval_df["_pred_prob_a"] >= 0.5) == eval_df["won_a"].astype(bool)
        eval_correct = int(ec.sum())
        eval_fights = int(len(eval_df))
        eval_feed = _events_from(eval_df)

    # ── Combined single tally ───────────────────────────────────────────────
    comb_correct = eval_correct + int(live["correct"])
    comb_fights  = eval_fights + int(live["fights"])
    comb_wrong   = comb_fights - comb_correct
    comb_rate    = round(comb_correct / comb_fights, 3) if comb_fights else 0.0

    # ── Merged event feed (live overrides same-date eval), most recent first ─
    by_date: dict[str, dict] = {}
    for ev in eval_feed:
        by_date[ev["date"]] = ev
    for ev in live.get("events", []):
        ev["live"] = True  # prod forward record (logged pre-fight); absent = eval backtest
        # The live feed is built from the prediction log, which has neither card order
        # nor a current event name. Repair both here, where the scraped maps exist.
        ev["event"] = event_date_to_name.get(ev["date"], ev["event"])
        if pair_to_pos:
            ev["fights"] = sorted(
                ev["fights"],
                key=lambda f: pair_to_pos.get(
                    frozenset((_norm_name(f["red"]), _norm_name(f["blue"]))), 9999),
            )
        by_date[ev["date"]] = ev
    all_events_asc = sorted(by_date.values(), key=lambda e: e["date"])
    feed = list(reversed(all_events_asc))[:n_events]

    # Full-history per-event series (correct/total only, no fight rows) — powers the
    # History page's cumulative "title reign" chart without needing the fight-level feed.
    series = [
        {
            "date": ev["date"], "event": ev["event"],
            "correct": ev["correct"], "total": ev["total"],
            "live": bool(ev.get("live")),
        }
        for ev in all_events_asc
    ]

    return {
        "events": feed,
        "series": series,
        "totals": {
            "dbEvents":  total_db_events,
            "dbFights":  total_db_fights,
            "correct":   comb_correct,
            "wrong":     comb_wrong,
            "fights":    comb_fights,
            "hitRate":   comb_rate,
            "evalFights":  eval_fights,
            "liveFights":  int(live["fights"]),
            "livePending": int(live.get("pending", 0)),
            "liveSince":   live.get("since"),
        },
    }


def get_ledger_summary() -> dict:
    """Advisory Model-vs-Market aggregation from the row-level prop ledger.

    Read-only aggregation of the same forward prop log 08b/08c already grade —
    no model behavior implications. Cached for process lifetime like
    get_history_feed. Returns available=False when there are no resolved rows yet
    (e.g. a fresh deploy with no local prediction history) so the frontend can
    show an explainer instead of an empty page.
    """
    with _lock:
        cached = _state.get("ledger_summary_cache")
    if cached is not None:
        return cached
    summary = _build_ledger_summary()
    summary["exchange"] = _build_exchange_summary()
    with _lock:
        _state["ledger_summary_cache"] = summary
    return summary


def _clv_delta_fav(df: "pd.DataFrame") -> "pd.Series":
    """Signed line-movement delta, positive = closed in our favor. Mirrors the
    match logic in scripts/08c_report_prop_ledger.py (side determines direction)."""
    close = pd.to_numeric(df["close_line_value"], errors="coerce")
    line = pd.to_numeric(df["line_value"], errors="coerce")
    return (close - line).where(df["side"] == "over", line - close)


def _build_ledger_summary() -> dict:
    """DFS (Power Play/Flat Multi) prop ledger only — see _build_exchange_summary for
    the Kalshi lane. Filtered by platform so the two lanes never mix in one metric."""
    from ufc.inference.prop_prediction_log import load_log

    try:
        df = load_log()
    except Exception:
        return {"available": False}

    res = df[(df["status"] == "resolved") & (df["platform"] != "kalshi")].copy()
    if len(res) == 0:
        return {"available": False}

    res["hit"] = res["hit"].astype(bool)
    res["edge_pct"] = pd.to_numeric(res["edge_pct"], errors="coerce")
    res["model_prob"] = pd.to_numeric(res["model_prob"], errors="coerce")
    n_graded = len(res)

    # Edge picks are the meaningful subset (mirrors scripts/08c_report_prop_ledger.py):
    # raw all-props hit rate is ~50% by construction — both O/U sides of efficient
    # lines get logged, so it's not a skill measure. "overall" and "byMarket" below
    # report edge picks only; edgeBuckets keeps the <0% bucket for contrast.
    edges = res[res["edge_pct"] > 0].copy()
    n_edges = len(edges)

    close_num = pd.to_numeric(res["close_line_value"], errors="coerce")
    clv_rows = res[close_num.notna()].copy()
    beat_close = beat_close_n = None
    if len(clv_rows) > 0:
        delta_fav = _clv_delta_fav(clv_rows)
        moved = clv_rows[delta_fav != 0]
        if len(moved) > 0:
            beat_close = float((delta_fav.loc[moved.index] > 0).mean())
            beat_close_n = int(len(moved))

    # Edge buckets: model_prob - breakeven, in percentage points. Spans the full
    # resolved set (including <0%) so the diagnostic-only picks read as contrast,
    # not as part of the "does the model beat the market" verdict.
    edge_pts = res["edge_pct"] * 100
    bucket_defs = [(-100, 0, "<0%"), (0, 3, "0-3%"), (3, 6, "3-6%"), (6, 10, "6-10%"), (10, 100, "10%+")]
    edge_buckets = []
    for lo, hi, label in bucket_defs:
        sub = res[(edge_pts > lo) & (edge_pts <= hi)]
        if len(sub) == 0:
            continue
        edge_buckets.append({
            "label": label, "n": int(len(sub)),
            "hitRate": round(float(sub["hit"].mean()), 4),
            "expectedRate": round(float(sub["model_prob"].mean()), 4),
        })

    by_market = []
    for mkt, sub in edges.groupby("market"):
        row = {
            "market": mkt, "n": int(len(sub)),
            "hitRate": round(float(sub["hit"].mean()), 4),
            "avgEdge": round(float(sub["edge_pct"].mean()), 4),
        }
        mkt_close = close_num.loc[sub.index]
        mkt_clv = sub[mkt_close.notna()]
        if len(mkt_clv) > 0:
            d = _clv_delta_fav(mkt_clv)
            moved = mkt_clv[d != 0]
            row["beatClose"] = round(float((d.loc[moved.index] > 0).mean()), 4) if len(moved) > 0 else None
        else:
            row["beatClose"] = None
        by_market.append(row)
    by_market.sort(key=lambda r: -r["n"])

    clv_series = []
    if len(clv_rows) > 0:
        clv_rows["_ev"] = clv_rows["event_date"].astype(str).str[:10]
        for ev, sub in clv_rows.groupby("_ev"):
            d = _clv_delta_fav(sub)
            moved = sub[d != 0]
            if len(moved) == 0:
                continue
            clv_series.append({
                "date": ev, "n": int(len(moved)),
                "beatClose": round(float((d.loc[moved.index] > 0).mean()), 4),
            })
        clv_series.sort(key=lambda r: r["date"])

    return {
        "available": True,
        "overall": {
            "picks": n_edges, "graded": n_graded,
            "hitRate": round(float(edges["hit"].mean()), 4) if n_edges else None,
            "avgEdge": round(float(edges["edge_pct"].mean()), 4) if n_edges else None,
            "beatClose": round(beat_close, 4) if beat_close is not None else None,
            "beatCloseN": beat_close_n,
        },
        "edgeBuckets": edge_buckets,
        "byMarket": by_market,
        "clvSeries": clv_series,
    }


def _exchange_bucket(sub: "pd.DataFrame") -> dict:
    if len(sub) == 0:
        return {"n": 0, "hitRate": None, "avgEdge": None, "roiNetFees": None}
    hit = sub["hit"].astype(bool)
    be = pd.to_numeric(sub["breakeven"], errors="coerce").clip(lower=1e-6)
    roi = float((hit.astype(float) / be).mean() - 1.0)
    return {
        "n": int(len(sub)),
        "hitRate": round(float(hit.mean()), 4),
        "avgEdge": round(float(pd.to_numeric(sub["edge_pct"], errors="coerce").mean()), 4),
        "roiNetFees": round(roi, 4),
    }


def _build_exchange_summary() -> dict:
    """Kalshi winner/method ledger — CLV in cents, taker headline + maker
    counterfactual (see 07b_log_prop_lines._log_kalshi_lines). Rows use side="over"
    so _clv_delta_fav's existing sign convention (close > open = favorable) is
    correct unchanged for a Kalshi "yes" price."""
    from ufc.inference.prop_prediction_log import load_log

    try:
        df = load_log()
    except Exception:
        return {"available": False}

    ex = df[(df["status"] == "resolved") & (df["platform"] == "kalshi")].copy()
    if len(ex) == 0:
        return {"available": False}

    ex["hit"] = ex["hit"].astype(bool)
    ex["edge_pct"] = pd.to_numeric(ex["edge_pct"], errors="coerce")
    ex["breakeven"] = pd.to_numeric(ex["breakeven"], errors="coerce")
    ex["model_prob"] = pd.to_numeric(ex["model_prob"], errors="coerce")
    ex["line_value"] = pd.to_numeric(ex["line_value"], errors="coerce")

    taker = ex[ex["odds_type"] == "taker"].copy()
    maker = ex[ex["odds_type"] == "maker"].copy()

    # Model-vs-market scatter points (model prob vs the ask it was priced against).
    points = [
        {"modelP": round(float(row["model_prob"]), 4), "marketP": round(float(row["line_value"]), 4), "hit": bool(row["hit"])}
        for _, row in taker.iterrows()
        if pd.notna(row["model_prob"]) and pd.notna(row["line_value"])
    ]

    close_num = pd.to_numeric(taker["close_line_value"], errors="coerce")
    clv_rows = taker[close_num.notna()].copy()
    avg_clv_cents = beat_close = beat_close_n = None
    if len(clv_rows) > 0:
        delta_cents = _clv_delta_fav(clv_rows) * 100.0
        avg_clv_cents = round(float(delta_cents.mean()), 2)
        moved = clv_rows[delta_cents != 0]
        if len(moved) > 0:
            beat_close = round(float((delta_cents.loc[moved.index] > 0).mean()), 4)
            beat_close_n = int(len(moved))

    by_market = [
        {"market": mkt, **_exchange_bucket(sub)}
        for mkt, sub in taker.groupby("market")
    ]
    by_market.sort(key=lambda r: -r["n"])

    clv_series = []
    if len(clv_rows) > 0:
        clv_rows = clv_rows.copy()
        clv_rows["_ev"] = clv_rows["event_date"].astype(str).str[:10]
        for ev, sub in clv_rows.groupby("_ev"):
            d = _clv_delta_fav(sub) * 100.0
            clv_series.append({"date": ev, "n": int(len(sub)), "avgClvCents": round(float(d.mean()), 2)})
        clv_series.sort(key=lambda r: r["date"])

    return {
        "available": True,
        "overall": _exchange_bucket(taker),
        "makerCounterfactual": _exchange_bucket(maker),
        "points": points,
        "avgClvCents": avg_clv_cents,
        "beatClose": beat_close,
        "beatCloseN": beat_close_n,
        "byMarket": by_market,
        "clvSeries": clv_series,
    }


def _infer_event_name(grp: "pd.DataFrame") -> str:
    for col in ("event_name", "event", "event_label"):
        if col in grp.columns:
            val = grp[col].iloc[0]
            if val and str(val) != "nan":
                return str(val)
    date_str = str(grp["event_date"].iloc[0])[:10]
    return f"UFC Event {date_str}"
