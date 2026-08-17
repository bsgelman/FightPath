"""Grade pending prop predictions against resolved results.

Run AFTER results are scraped + features rebuilt. Fills realized_stat, hit, resolved_at
for any pending row whose fight has now happened.

    python scripts/08b_grade_props.py            # grade pending rows
    python scripts/08b_grade_props.py --regrade  # ALSO re-resolve already-graded rows

Match key: (event_date, sorted normalized name pair). The fighter a prop measures is
resolved by NAME (not corner order) because features_props' a/b ordering does NOT track
the card's red/blue corners — assuming it does silently grades the wrong fighter's stat.

Markets graded
--------------
* count props   (per-fighter stat O/U a numeric line) — sig_strikes, takedowns, ...
* duration      (fight-level total seconds O/U a line; corner == "fight")
* finish family (fighter-directional binary @ 0.5): finish / ko_finish / sub_finish /
                 r{1..5}_finish / r{1..5}_ko. realized = 1.0 iff the NAMED fighter wins by
                 that finish type (r{k}_ko requires the win to be KO/TKO specifically, not
                 any finish, in round k), graded with the model's own method convention
                 (METHOD_MAP) so the ledger is consistent with what the model predicted.
* Kalshi (winner, method_ko/sub/dec, distance, mof_ko/sub/dec, end_before_r{N},
  win_in_r{N}, vicround_other) — direct win/method/round boolean, no line_value
  threshold (Kalshi rows are priced, not a stat O/U). Settlement for all eleven
  kinds is table-driven — see ufc.valuation.kalshi_grading.settle(), the single
  source of truth for what counts as a hit. Uses the RAW features_props method
  label (not method_class) so DQ correctly hits the KO/TKO-family kinds instead
  of being folded into "decision" the way the model's 3-class space treats it.

No-contests (method == "NC") void every Kalshi kind EXCEPT vicround_other, which
Kalshi's own rules settle YES on a no-contest (bucketed with Decision/Draw). Their
count/duration stats still grade normally regardless.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pandas as pd

from ufc.io import paths, parquet
from ufc.inference.prediction_log import norm_name, matchup_key
from ufc.inference.prop_prediction_log import load_log, save_log, now_iso
from ufc.models.method import METHOD_CLASSES, METHOD_MAP
from ufc.valuation.kalshi_grading import DEC_RAW_METHODS, FightOutcome, kind_spec, settle as kalshi_settle

# ── Market → features_props column(s) ───────────────────────────────────────────
# Per-fighter count markets: (col_a, col_b). The named fighter picks the side.
_COUNT_MARKETS: dict[str, tuple[str, str]] = {
    "sig_strikes":      ("sig_str_landed_a",    "sig_str_landed_b"),
    "r1_sig_strikes":   ("r1_sig_str_landed_a", "r1_sig_str_landed_b"),
    "takedowns":        ("td_landed_a",         "td_landed_b"),
    "knockdowns":       ("kd_for_a",            "kd_for_b"),
    "sub_attempts":     ("sub_att_for_a",       "sub_att_for_b"),
    "r1_takedowns":     ("r1_td_landed_a",      "r1_td_landed_b"),
    "body_sig_strikes": ("body_landed_a",       "body_landed_b"),
    "leg_sig_strikes":  ("leg_landed_a",        "leg_landed_b"),
    "ctrl_time":        ("ctrl_sec_a",          "ctrl_sec_b"),
}
# Fight-level markets: single column, corner-independent.
_FIGHT_MARKETS: dict[str, str] = {
    "duration": "total_fight_sec",
}
# Fighter-directional binary finish markets (line is 0.5; realized is 1.0/0.0).
_FINISH_MARKETS = frozenset({
    "finish", "ko_finish", "sub_finish",
    "r1_finish", "r2_finish", "r3_finish", "r4_finish", "r5_finish",
    "r1_ko", "r2_ko", "r3_ko", "r4_ko", "r5_ko",
})


def _finish_round(market: str) -> int | None:
    """Return k for 'r{k}_finish' / 'r{k}_ko' markets, else None (matches prop_cdf.py)."""
    if market in ("ko_finish", "sub_finish", "finish"):
        return None
    if market.startswith("r") and (market.endswith("_finish") or market.endswith("_ko")):
        try:
            return int(market[1:market.index("_")])
        except (ValueError, IndexError):
            pass
    return None


def _is_round_ko(market: str) -> bool:
    """True for 'r{k}_ko' markets — KO/TKO-only, unlike 'r{k}_finish' (any finish)."""
    return market.startswith("r") and market.endswith("_ko")


def _method_class(method: str) -> str:
    """Map a raw result method to the model's 3-class space (KO/TKO|SUB|DEC)."""
    return METHOD_CLASSES[METHOD_MAP.get(str(method), 2)]


def _safe_int(v) -> int | None:
    """Best-effort int conversion for a features_props numeric field. None on
    missing/NaN/unparsable — callers treat that as 'not knowable yet'."""
    try:
        if v is None or pd.isna(v):
            return None
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _finish_realized(market: str, fighter_won: bool,
                     method_class: str, end_round) -> float:
    """1.0 iff the NAMED fighter wins by this finish type, mirroring prop_cdf.py."""
    if not fighter_won:
        return 0.0
    if market == "ko_finish":
        return 1.0 if method_class == "KO/TKO" else 0.0
    if market == "sub_finish":
        return 1.0 if method_class == "SUB" else 0.0
    if market == "finish":
        return 1.0 if method_class != "DEC" else 0.0
    k = _finish_round(market)
    if k is not None:
        try:
            er = int(round(float(end_round)))
        except (TypeError, ValueError):
            return 0.0
        is_fin = (method_class == "KO/TKO") if _is_round_ko(market) else (method_class != "DEC")
        return 1.0 if (is_fin and er == k) else 0.0
    return 0.0


def _build_result_map() -> dict:
    """matchup_key -> resolved fight outcome (names, winner, method, round, stats)."""
    fp = parquet.read(paths.processed("features_props"))
    fp = fp.dropna(subset=["fight_id"]).copy()
    fp["event_date"] = pd.to_datetime(fp["event_date"])

    id_to_name: dict = {}
    try:
        fdf = parquet.read(paths.interim("fighters"))
        ncol = next((c for c in ("fighter_name", "name") if c in fdf.columns), None)
        if "fighter_id" in fdf.columns and ncol:
            id_to_name = dict(zip(fdf["fighter_id"].astype(str), fdf[ncol].astype(str)))
    except Exception:
        pass

    def _nm(row, side):
        v = row.get(f"fighter_{side}_name") or row.get(f"fighter_name_{side}")
        if isinstance(v, str) and v:
            return v
        fid = row.get(f"fighter_id_{side}") or row.get(f"fighter_{side}_id")
        return id_to_name.get(str(fid), "")

    out: dict = {}
    for _, r in fp.iterrows():
        a, b = _nm(r, "a"), _nm(r, "b")
        if not a or not b:
            continue
        mk = matchup_key(r["event_date"].strftime("%Y-%m-%d"), a, b)
        if mk in out:
            continue  # one row per fight; first wins

        won_a = r.get("won_a")
        method = r.get("method")
        stats = {}
        for market, (col_a, col_b) in _COUNT_MARKETS.items():
            if col_a in r and col_b in r:
                stats[market] = {"a": r.get(col_a), "b": r.get(col_b)}

        out[mk] = {
            "name_a": a, "name_b": b,
            "won_a": (None if won_a is None or pd.isna(won_a) else bool(won_a)),
            "method": method,
            "method_class": _method_class(method),
            "end_round": r.get("end_round"),
            "scheduled_rounds": r.get("scheduled_rounds"),
            "total_fight_sec": r.get("total_fight_sec"),
            "stats": stats,
        }
    return out


def _which_side(fres: dict, fighter: str, corner: str) -> str | None:
    """Return 'a' or 'b' for the named fighter, by NAME match. None if unresolved."""
    nf = norm_name(fighter)
    if nf:
        if nf == norm_name(fres["name_a"]):
            return "a"
        if nf == norm_name(fres["name_b"]):
            return "b"
    return None


def _resolve_row(row, fres: dict):
    """Return (realized_value, hit_bool, status) for one prop row, or (None, None, None)
    if the result is not yet available (leave pending)."""
    market   = str(row["market"])
    side     = str(row["side"])
    line_val = float(row["line_value"])

    # ── Fight-level (duration): corner-independent ──────────────────────────────
    if market in _FIGHT_MARKETS:
        realized = fres.get(_FIGHT_MARKETS[market])
        if realized is None or pd.isna(realized):
            return None, None, None
        realized = float(realized)
        hit = realized > line_val if side == "over" else realized <= line_val
        return realized, hit, "resolved"

    # ── Kalshi (all 11 kinds): table-driven settlement, see kalshi_grading.py ──
    kspec = kind_spec(market)
    if kspec is not None:
        raw_method = str(fres.get("method"))
        fighter_won = None
        if kspec.needs_fighter:
            ab = _which_side(fres, str(row.get("fighter", "")), str(row.get("corner", "")))
            won_a = fres.get("won_a")
            if ab is not None and won_a is not None:
                fighter_won = won_a if ab == "a" else (not won_a)
        is_draw = fres.get("won_a") is None and raw_method in DEC_RAW_METHODS
        outcome = FightOutcome(
            fighter_won=fighter_won,
            raw_method=raw_method,
            end_round=_safe_int(fres.get("end_round")),
            scheduled_rounds=_safe_int(fres.get("scheduled_rounds")),
            total_fight_sec=fres.get("total_fight_sec"),
            is_draw=is_draw,
        )
        return kalshi_settle(market, outcome)

    # ── Finish family: directional binary on the NAMED fighter ──────────────────
    if market in _FINISH_MARKETS:
        if str(fres.get("method")) == "NC":
            return None, None, "void"           # no-contest voids the bet
        ab = _which_side(fres, str(row.get("fighter", "")), str(row.get("corner", "")))
        won_a = fres.get("won_a")
        if ab is None or won_a is None:
            return None, None, None             # winner/fighter unresolved -> pending
        fighter_won = won_a if ab == "a" else (not won_a)
        realized = _finish_realized(market, fighter_won,
                                    fres["method_class"], fres.get("end_round"))
        hit = realized > line_val if side == "over" else realized <= line_val
        return realized, hit, "resolved"

    # ── Per-fighter count markets ───────────────────────────────────────────────
    if market in _COUNT_MARKETS:
        ab = _which_side(fres, str(row.get("fighter", "")), str(row.get("corner", "")))
        mstats = fres.get("stats", {}).get(market)
        if ab is None or mstats is None:
            return None, None, None
        realized = mstats.get(ab)
        if realized is None or pd.isna(realized):
            return None, None, None
        realized = float(realized)
        hit = realized > line_val if side == "over" else realized <= line_val
        return realized, hit, "resolved"

    return None, None, None  # unknown market


def main():
    regrade = "--regrade" in sys.argv
    log = load_log()

    if regrade:
        target = log["status"].isin(["pending", "resolved", "void"])
    else:
        target = log["status"] == "pending"

    if not target.any():
        print("[grade-props] No rows to grade.")
        return

    results = _build_result_map()
    graded = voided = 0

    for idx in log.index[target]:
        row = log.loc[idx]
        ed  = str(row["event_date"])[:10]
        mk  = matchup_key(ed, str(row["red"]), str(row["blue"]))
        fres = results.get(mk)
        if fres is None:
            continue

        realized, hit, status = _resolve_row(row, fres)
        if status is None:
            continue

        if status == "void":
            log.at[idx, "realized_stat"] = None
            log.at[idx, "hit"]           = None
            log.at[idx, "status"]        = "void"
            log.at[idx, "resolved_at"]   = now_iso()
            voided += 1
            continue

        log.at[idx, "realized_stat"] = realized
        log.at[idx, "hit"]           = bool(hit)
        log.at[idx, "status"]        = "resolved"
        log.at[idx, "resolved_at"]   = now_iso()
        graded += 1

    if graded or voided:
        save_log(log)

    resolved_total = int((log["status"] == "resolved").sum())
    hits_total     = int(log.loc[log["status"] == "resolved", "hit"].astype(bool).sum())
    still_pending  = int((log["status"] == "pending").sum())
    void_total     = int((log["status"] == "void").sum())
    rate = hits_total / resolved_total if resolved_total else 0.0
    verb = "re-graded" if regrade else "graded"
    print(f"[grade-props] {verb} {graded} this run "
          f"({voided} voided) · prop ledger now "
          f"{hits_total}/{resolved_total} ({rate*100:.1f}% hit) · "
          f"{still_pending} pending · {void_total} void")


if __name__ == "__main__":
    main()
