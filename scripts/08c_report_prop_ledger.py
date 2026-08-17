"""Forward prop-ledger performance report - the honest betting track record.

Reads data/predictions/prop_log.parquet (graded by 08b_grade_props.py) and reports
what actually measures model skill on props: **edge picks vs their own break-even**,
NOT the raw all-props hit rate (which is ~50% mechanically - operators set O/U lines
at the median, and logging both sides of a .5 line forces one hit + one miss).

    python scripts/08c_report_prop_ledger.py                 # whole resolved ledger
    python scripts/08c_report_prop_ledger.py --event 2026-06-27
    python scripts/08c_report_prop_ledger.py --edge-thresh 0.05

Metrics (edge picks = model_prob > break-even + thresh):
  hit%        realized hit rate of the picks
  be%         mean break-even implied by the payout (the bar to beat)
  edge_real   hit% - be%      (>0 = the bucket beat its price)
  roi         mean(hit/be) - 1  per-leg EV proxy priced AT break-even
              (NOT a placeable-parlay P&L; treats each leg as a standalone bet at 1/be)
  claimed     mean model_prob  (what the model said) - vs hit% shows calibration

CLV: closing lines are captured by 07c_capture_closing_lines.py (run Saturday near
card-lock). Once captured, resolved rows show beat-close% and per-market avg line delta
in the CLV section below the calibration table.

Writes outputs/reports/prop_ledger_<date>.md + .parquet for trend tracking.
"""
import math
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd

from ufc.io import paths
from ufc.inference.prop_prediction_log import load_log

# Min graded edge picks before a per-market calibration verdict is trusted.
# Below this the market is reported as "thin" (no flag) so we never cry wolf on noise.
_CALIB_MIN_N = 25
_CALIB_Z = 1.96   # ~95% two-sided


def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _bucket_stats(df: pd.DataFrame) -> dict:
    """Edge-aware stats for a set of resolved prop rows."""
    n = len(df)
    if n == 0:
        return {"n": 0, "hit": np.nan, "be": np.nan, "edge_real": np.nan,
                "roi": np.nan, "claimed": np.nan}
    hit = df["hit"].astype(bool)
    be  = df["breakeven"].clip(lower=1e-6)
    roi = float((hit.astype(float) / be).mean() - 1.0)
    return {
        "n": int(n),
        "hit": float(hit.mean()),
        "be": float(df["breakeven"].mean()),
        "edge_real": float(hit.mean() - df["breakeven"].mean()),
        "roi": roi,
        "claimed": float(df["model_prob"].mean()),
    }


def _calib(df: pd.DataFrame, min_n: int = _CALIB_MIN_N) -> dict:
    """Poisson-binomial calibration test: do realized hits match the model's claims?

    Each pick i is a claim p_i = model_prob with outcome hit_i in {0,1}. If the model
    is perfectly calibrated, observed hits ~ PoissonBinomial(p_i), with
    mean = sum(p_i), var = sum(p_i(1-p_i)). z<0 => fewer hits than claimed = OVERCONFIDENT.
    Flags only once n >= min_n so the verdict grows trustworthy with sample, not noise.
    """
    n = len(df)
    if n == 0:
        return {"n": 0, "claimed": np.nan, "realized": np.nan, "z": np.nan,
                "p": np.nan, "flag": "none"}
    p = df["model_prob"].to_numpy(dtype=float)
    obs = float(df["hit"].astype(bool).sum())
    exp = float(p.sum())
    var = float((p * (1.0 - p)).sum())
    z = (obs - exp) / math.sqrt(var) if var > 1e-9 else np.nan
    pval = math.erfc(abs(z) / math.sqrt(2)) if z == z else np.nan   # two-sided normal approx
    if n < min_n:
        flag = "thin"
    elif z == z and z <= -_CALIB_Z:
        flag = "OVERCONFIDENT"
    elif z == z and z >= _CALIB_Z:
        flag = "underconfident"
    else:
        flag = "calibrated"
    return {"n": n, "claimed": exp / n, "realized": obs / n, "z": z, "p": pval, "flag": flag}


def _fmt_calib(c: dict) -> str:
    if c["n"] == 0:
        return f"{c['n']:>4}        -         -        -    -"
    zs = f"{c['z']:>+5.2f}" if c["z"] == c["z"] else "   - "
    ps = f"{c['p']*100:>5.1f}%" if c["p"] == c["p"] else "   - "
    return (f"{c['n']:>4}  {c['claimed']*100:>7.1f}%  {c['realized']*100:>7.1f}%  "
            f"{zs}  {ps}  {c['flag']}")


_CALIB_HDR = f"{'n':>4}  {'claimed':>8}  {'realized':>8}  {'z':>5}  {'p':>5}  flag"

_CLV_HDR = f"{'market':<20}  {'n_close':>7}  {'moved':>5}  {'beat%':>6}  {'avg_delta':>9}"


def _clv_market_stats(df: pd.DataFrame) -> dict:
    """Per-market CLV fields for parquet trend tracking."""
    if "close_line_value" not in df.columns:
        return {"clv_n_with_close": 0, "clv_n_moved": 0,
                "clv_beat_close_pct": np.nan, "clv_avg_delta_fav": np.nan}
    close_num = pd.to_numeric(df["close_line_value"], errors="coerce")
    sub = df[close_num.notna()].copy()
    if len(sub) == 0:
        return {"clv_n_with_close": 0, "clv_n_moved": 0,
                "clv_beat_close_pct": np.nan, "clv_avg_delta_fav": np.nan}
    sub["close_line_value"] = pd.to_numeric(sub["close_line_value"], errors="coerce")
    sub["line_value"] = pd.to_numeric(sub["line_value"], errors="coerce")
    # signed delta: positive = favorable (line moved in our direction)
    sub["delta_fav"] = (sub["close_line_value"] - sub["line_value"]).where(
        sub["side"] == "over",
        sub["line_value"] - sub["close_line_value"],
    )
    sub["moved"] = sub["close_line_value"] != sub["line_value"]
    moved = sub[sub["moved"]]
    n_moved = len(moved)
    n_fav = int((moved["delta_fav"] > 0).sum()) if n_moved > 0 else 0
    beat_pct = n_fav / n_moved if n_moved > 0 else np.nan
    avg_d = float(moved["delta_fav"].mean()) if n_moved > 0 else np.nan
    return {
        "clv_n_with_close": len(sub),
        "clv_n_moved": n_moved,
        "clv_beat_close_pct": beat_pct,
        "clv_avg_delta_fav": avg_d,
    }


def _fmt(s: dict) -> str:
    if s["n"] == 0:
        return f"{s['n']:>4}      -        -        -        -        -"
    return (f"{s['n']:>4}  {s['hit']*100:>6.1f}%  {s['be']*100:>6.1f}%  "
            f"{s['edge_real']*100:>+7.1f}pp  {s['roi']*100:>+7.1f}%  {s['claimed']*100:>6.1f}%")


_HDR = f"{'n':>4}  {'hit':>7}  {'be':>7}  {'edge_real':>9}  {'roi':>8}  {'claimed':>7}"


def _report_exchange(log: pd.DataFrame, event: str | None) -> None:
    """Kalshi winner/method section — taker headline, maker as a counterfactual
    row (does fees, not spread, explain a missed edge?)."""
    ex = log[(log["status"] == "resolved") & (log["platform"] == "kalshi")].copy()
    if event:
        ex = ex[ex["event_date"].astype(str).str[:10] == event]
    if len(ex) == 0:
        return

    ex = _num(ex, ["model_prob", "breakeven", "edge_pct", "line_value", "close_line_value"])
    ex["hit"] = ex["hit"].astype(bool)
    taker = ex[ex["odds_type"] == "taker"]
    maker = ex[ex["odds_type"] == "maker"]

    print("")
    print("=== Exchange (Kalshi) Ledger ===")
    print(f"resolved: {len(taker)} taker rows, {len(maker)} maker counterfactual rows")
    print(f"  {'lane':<10}{_HDR}")
    print(f"  {'taker':<10}{_fmt(_bucket_stats(taker))}")
    print(f"  {'maker*':<10}{_fmt(_bucket_stats(maker))}   (*counterfactual, not tradable)")
    # Fill-aware maker lane: a resting bid+1c only fills when the price trades down
    # through it — grading just those rows prices in adverse selection. Needs yes_bid
    # (logged from 2026-07-12) + a captured close; older rows are excluded, not assumed.
    mk = maker.copy()
    mk["yes_bid"] = pd.to_numeric(mk.get("yes_bid"), errors="coerce")
    mk["close"] = pd.to_numeric(mk["close_line_value"], errors="coerce")
    known = mk[mk["yes_bid"].notna() & mk["close"].notna()]
    if len(known):
        filled = known[known["close"] <= known["yes_bid"] + 0.01]
        print(f"  {'maker-fill':<10}{_fmt(_bucket_stats(filled))}   "
              f"(close<=bid+1c proxy; {len(filled)}/{len(known)} filled)")
    print("  per market:")
    for mkt, df in sorted(taker.groupby("market"), key=lambda kv: -len(kv[1])):
        print(f"    {mkt:<20}{_fmt(_bucket_stats(df))}")

    close_num = pd.to_numeric(taker["close_line_value"], errors="coerce")
    clv = taker[close_num.notna()].copy()
    if len(clv) == 0:
        print("CLV: no closing lines captured yet (run 07c near card-lock).")
        return
    clv["delta_cents"] = (
        pd.to_numeric(clv["close_line_value"], errors="coerce")
        - pd.to_numeric(clv["line_value"], errors="coerce")
    ) * 100.0
    print(f"CLV: avg {clv['delta_cents'].mean():+.2f}c over {len(clv)} taker picks with a close "
          f"(positive = price moved in our favor)")
    print("  per market (PAPER->LIVE flip evidence, see configs/market_advice.yaml):")
    for mkt, df in sorted(clv.groupby("market"), key=lambda kv: -len(kv[1])):
        print(f"    {mkt:<20}{len(df):>4}  avg {df['delta_cents'].mean():+7.2f}c")


def main():
    args = sys.argv[1:]
    event = None
    thresh = 0.0
    if "--event" in args:
        event = args[args.index("--event") + 1]
    if "--edge-thresh" in args:
        thresh = float(args[args.index("--edge-thresh") + 1])

    log = load_log()
    res = log[(log["status"] == "resolved") & (log["platform"] != "kalshi")].copy()
    if event:
        res = res[res["event_date"].astype(str).str[:10] == event]
    if len(res) == 0:
        print(f"[ledger] No resolved DFS prop rows{' for ' + event if event else ''}.")
        _report_exchange(log, event)
        return

    res = _num(res, ["model_prob", "breakeven", "edge_pct", "line_value", "close_line_value"])
    res["hit"] = res["hit"].astype(bool)
    edges = res[res["edge_pct"] > thresh].copy()

    scope = event or f"{res['event_date'].astype(str).str[:10].min()} -> {res['event_date'].astype(str).str[:10].max()}"
    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit(f"=== Forward Prop Ledger Report - {scope} ===")
    emit(f"resolved props: {len(res)}  |  edge picks (edge>{thresh*100:.0f}%): {len(edges)}  "
         f"|  void/pending excluded")
    emit("")
    emit("RAW all-props hit rate (~50% by construction - both sides logged + efficient lines):")
    emit(f"  {res['hit'].mean()*100:.1f}%  ({int(res['hit'].sum())}/{len(res)})  <- not a skill measure; see edge picks below")
    emit("")
    emit(f"EDGE PICKS - the meaningful subset (model > payout break-even):  {_HDR}")
    emit(f"  {'ALL EDGES':<22}{_fmt(_bucket_stats(edges))}")
    emit("  " + "-" * 78)

    # group families
    fam = {
        "count/duration": edges[~edges["market"].str.contains("finish", na=False)],
        "finish-family":  edges[edges["market"].str.contains("finish", na=False)],
    }
    for name, df in fam.items():
        emit(f"  {name:<22}{_fmt(_bucket_stats(df))}")
    emit("")
    emit(f"  per market:                          {_HDR}")
    rows_pq = []
    for mkt, df in sorted(edges.groupby("market"), key=lambda kv: -len(kv[1])):
        s = _bucket_stats(df); s["market"] = mkt
        c = _calib(df)
        clv = _clv_market_stats(df)
        s.update({f"calib_{k}": v for k, v in c.items() if k != "n"})
        s.update(clv)
        rows_pq.append(s)
        emit(f"    {mkt:<20}{_fmt(s)}")

    # running per-event (so each card shows up)
    if not event:
        emit("")
        emit(f"  per event (edge picks):              {_HDR}")
        for ev, df in edges.groupby(edges["event_date"].astype(str).str[:10]):
            emit(f"    {ev:<20}{_fmt(_bucket_stats(df))}")

    # ---- CALIBRATION (do realized hits match what the model claimed?) ------------
    emit("")
    emit(f"CALIBRATION - edge picks, Poisson-binomial (flags only at n>={_CALIB_MIN_N}):  {_CALIB_HDR}")
    emit(f"  {'ALL EDGES':<22}{_fmt_calib(_calib(edges))}")
    emit("  " + "-" * 60)
    for mkt, df in sorted(edges.groupby("market"), key=lambda kv: -len(kv[1])):
        emit(f"    {mkt:<20}{_fmt_calib(_calib(df))}")
    flagged = [mkt for mkt, df in edges.groupby("market")
               if _calib(df)["flag"] == "OVERCONFIDENT"]
    if flagged:
        emit("")
        emit(f"  >> OVERCONFIDENT markets (trim these or widen the edge threshold): {', '.join(sorted(flagged))}")

    # ---- CLV (closing-line value) -----------------------------------------------
    emit("")
    clv_sub = edges.copy()
    if "close_line_value" not in clv_sub.columns or clv_sub["close_line_value"].isna().all():
        emit("CLV: no closing lines captured yet (run 07c on a Saturday closing pull).")
    else:
        clv_sub = clv_sub[clv_sub["close_line_value"].notna()].copy()
        clv_sub["delta_fav"] = (
            (clv_sub["close_line_value"] - clv_sub["line_value"]).where(
                clv_sub["side"] == "over",
                clv_sub["line_value"] - clv_sub["close_line_value"],
            )
        )
        clv_sub["moved"] = clv_sub["close_line_value"] != clv_sub["line_value"]
        n_with_close = len(clv_sub)
        moved_all = clv_sub[clv_sub["moved"]]
        n_moved = len(moved_all)
        n_no_move = n_with_close - n_moved
        n_fav = int((moved_all["delta_fav"] > 0).sum()) if n_moved > 0 else 0
        beat_pct = n_fav / n_moved if n_moved > 0 else float("nan")

        emit(f"CLV (Flat Multi-only; Power Play disabled): {n_with_close} edge picks have a closing line")
        emit(f"  moved: {n_moved}  no-move: {n_no_move}")
        if n_moved > 0:
            emit(f"  beat-close rate: {n_fav}/{n_moved} = {beat_pct*100:.1f}%"
                 f"  (% of moved lines that shifted in our favor)")
        else:
            emit("  beat-close rate: n/a (no lines moved)")
        emit(f"  per market:  {_CLV_HDR}")
        for mkt, mdf in sorted(clv_sub.groupby("market"), key=lambda kv: -len(kv[1])):
            mn_close = len(mdf)
            mm = mdf[mdf["moved"]]
            mn_moved = len(mm)
            mn_fav = int((mm["delta_fav"] > 0).sum()) if mn_moved > 0 else 0
            mbp = f"{mn_fav/mn_moved*100:.1f}%" if mn_moved > 0 else "  n/a"
            avg_d = f"{mm['delta_fav'].mean():+.2f}" if mn_moved > 0 else "   n/a"
            emit(f"    {mkt:<20}  {mn_close:>7}  {mn_moved:>5}  {mbp:>6}  {avg_d:>9}")

    # ---- VERDICT ----------------------------------------------------------------
    e_all = _bucket_stats(edges)
    c_all = _calib(edges)
    if e_all["n"]:
        verdict = ("beat break-even PASS" if e_all["edge_real"] > 0 else "under break-even FAIL")
        emit("")
        emit(f"VERDICT (edge picks): hit {e_all['hit']*100:.1f}% vs break-even {e_all['be']*100:.1f}% "
             f"-> {verdict};  ROI {e_all['roi']*100:+.1f}% (per-leg proxy)")
        cflag = (c_all["flag"] if c_all["flag"] not in ("thin", "none")
                 else f"thin sample (n={c_all['n']}<{_CALIB_MIN_N}) - no verdict yet")
        emit(f"CALIBRATION (overall): claimed {c_all['claimed']*100:.1f}%, realized {c_all['realized']*100:.1f}% "
             f"(z={c_all['z']:+.2f}) -> {cflag}")
    emit("")
    emit(f"_NOTE: small samples per card are high-variance - judge ROI/edge_real/calibration over many events._")
    emit(f"_CLV requires closing lines captured by 07c (Saturday) before the card; see CLV section above._")

    # ---- DRIFT SUMMARY (compact per-market table for the Monday refresh log) ---
    # Surfaces calibration drift + CLV without being asked - read by refresh_history.ps1's
    # Log wrapper, which captures this script's stdout.
    if rows_pq:
        emit("")
        emit("=== DRIFT SUMMARY (per market) ===")
        emit(f"{'market':<20} {'n':>4} {'hit':>7} {'be':>7} {'calib_z':>8} "
             f"{'calib_p':>8} {'flag':<13} {'clv_n':>5} {'beat%':>7}")
        warn_mkts = []
        for s in rows_pq:
            z, p = s.get("calib_z", float("nan")), s.get("calib_p", float("nan"))
            flag = s.get("calib_flag", "-")
            clv_n = s.get("clv_n_with_close") or 0
            beat = s.get("clv_beat_close_pct", float("nan"))
            zs = f"{z:>+8.2f}" if z == z else f"{'-':>8}"
            ps = f"{p*100:>7.1f}%" if p == p else f"{'-':>8}"
            bs = f"{beat*100:>6.1f}%" if beat == beat else f"{'-':>7}"
            emit(f"{s['market']:<20} {s['n']:>4} {s['hit']*100:>6.1f}% {s['be']*100:>6.1f}% "
                 f"{zs} {ps} {flag:<13} {clv_n:>5} {bs}")
            if (p == p and p < 0.10) or (clv_n >= 5 and beat == beat and beat < 0.5):
                warn_mkts.append(s["market"])
        if warn_mkts:
            emit(f"  >> WARNING: calib_p<0.10 or beat-close<50% (n>=5) in: {', '.join(sorted(warn_mkts))}")

    # write report artifacts
    rep = paths.outputs_reports(); rep.mkdir(parents=True, exist_ok=True)
    tag = event or "all"
    md = rep / f"prop_ledger_{date.today().isoformat()}_{tag}.md"
    md.write_text("# Forward Prop Ledger Report\n\n```\n" + "\n".join(lines) + "\n```\n",
                  encoding="utf-8")
    if rows_pq:
        pd.DataFrame(rows_pq).to_parquet(rep / f"prop_ledger_{date.today().isoformat()}_{tag}.parquet",
                                         index=False)
    print(f"\n[ledger] report written to {md.name}")

    _report_exchange(log, event)


if __name__ == "__main__":
    main()
