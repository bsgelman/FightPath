"""FastAPI application — all endpoints for the FightPath React UI.

Start with:
    uvicorn ufc.api.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import numpy as np
from contextlib import asynccontextmanager
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ufc.valuation.lines import Line
from ufc.valuation.edge import evaluate_line
from ufc.valuation.prop_menu import STANDARD_LINES, CANONICAL_TO_FRONTEND

from ufc.api import service, serialize, schemas
from ufc.api.ratelimit import check_rate_limit

logger = logging.getLogger(__name__)


# ── Prop trust tiers ─────────────────────────────────────────────────────────

def _load_prop_trust() -> dict[str, str]:
    """Return {canonical_key: tier} from configs/prop_trust.yaml."""
    import yaml
    from ufc.io import paths
    cfg_path = paths.root() / "configs" / "prop_trust.yaml"
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return dict(cfg.get("tiers", {}))
    except Exception:
        return {}


_PROP_TRUST: dict[str, str] = _load_prop_trust()
# Frontend market-key → tier (used by /api/prop-trust endpoint)
_PROP_TRUST_FRONTEND: dict[str, str] = {
    CANONICAL_TO_FRONTEND.get(k, k): v for k, v in _PROP_TRUST.items()
}


# ── Exchange (Kalshi) advice status ─────────────────────────────────────────

def _advice_paper_for_kind(cfg: dict, kind: str) -> bool:
    """Pure per-kind lookup: status_by_kind[kind] -> status_by_kind[family]
    (kind with trailing digits stripped, e.g. "end_before_r2" -> "end_before_r")
    -> global status -> "paper". Any unrecognized value fails closed to paper."""
    by_kind = cfg.get("status_by_kind") or {}
    if kind in by_kind:
        value = by_kind[kind]
    else:
        family = re.sub(r"\d+$", "", kind)
        value = by_kind.get(family, cfg.get("status", "paper"))
    return str(value).lower() != "live"


def _load_market_advice_cfg() -> dict:
    """Load configs/market_advice.yaml. Empty dict on any error (every lookup
    against an empty cfg falls back to "paper" — never silently ships live
    advice on a config problem)."""
    import yaml
    from ufc.io import paths
    cfg_path = paths.root() / "configs" / "market_advice.yaml"
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _market_advice_paper(kind: str) -> bool:
    """True while `kind` is advisory-PAPER (configs/market_advice.yaml)."""
    return _advice_paper_for_kind(_load_market_advice_cfg(), kind)


# ── Lifespan: load models once ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    service.startup()
    yield


app = FastAPI(title="FightPath API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    rejected = check_rate_limit(request)
    if rejected is not None:
        return rejected
    return await call_next(request)


_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins.split(",") if _cors_origins != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


@app.middleware("http")
async def _add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cards_dir() -> Path:
    from ufc.io import paths
    return paths.root() / "cards" / "upcoming"


def _list_card_files() -> list[Path]:
    d = _cards_dir()
    if not d.exists():
        return []
    return sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_card_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _card_list_item(path: Path, data: dict) -> dict:
    return {
        "id":        path.stem,
        "label":     data.get("event_name", path.stem.replace("_", " ").title()),
        "eventDate": data.get("event_date", ""),
        "nFights":   len(data.get("matchups", [])),
    }


def _parse_event_date(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except (ValueError, TypeError):
        return date.today()


def _card_still_listed(data: dict, now: datetime | None = None) -> bool:
    """Whether a scraped card should still appear in the dropdown.

    A card stays selectable past its start so a fight in progress / just finished
    doesn't vanish mid-analysis:
      • if an explicit start time is known → keep until 24h after that start;
      • otherwise (ufcstats gives date only) → keep until 2 days after the date.
    Unparseable dates are kept (fail-open).
    """
    now = now or datetime.now()
    ev_date_str = data.get("event_date", "")
    try:
        ev_date = date.fromisoformat(ev_date_str)
    except (ValueError, TypeError):
        return True
    t_raw = str(data.get("event_time") or data.get("start_time") or "").strip()
    if t_raw:
        try:
            ev_dt = datetime.combine(ev_date, dtime.fromisoformat(t_raw))
            return now < ev_dt + timedelta(hours=24)
        except (ValueError, TypeError):
            pass
    return now.date() <= ev_date + timedelta(days=2)


# Maps UI payout_key → backend payout_type string
_PAYOUT_MAP: dict[str, str] = {
    "pp_power_2": "powerplay_power_2pick",
    "pp_power_3": "powerplay_power_3pick",
    "pp_power_4": "powerplay_power_4pick",
    "pp_power_5": "powerplay_power_5pick",
    "ud_2": "flatmulti_power_2pick",
    "ud_3": "flatmulti_power_3pick",
    "ud_5": "flatmulti_power_5pick",
}

# Maps UI payout_key → number of legs (for parlay EV calculation)
_PAYOUT_LEGS: dict[str, int] = {
    "pp_power_2": 2, "pp_power_3": 3, "pp_power_4": 4, "pp_power_5": 5,
    "ud_2": 2, "ud_3": 3, "ud_5": 5,
}

_PayoutKey = Literal[
    "pp_power_2", "pp_power_3", "pp_power_4", "pp_power_5",
    "ud_2", "ud_3", "ud_5",
]

_CARD_ID_RE = re.compile(r"^[\w\-]+$")

def _validate_card_id(card_id: str) -> None:
    if not _CARD_ID_RE.fullmatch(card_id):
        raise HTTPException(status_code=400, detail="Invalid card ID")


_last_refresh: float = 0.0
_card_cache: dict = {}  # (card_id, mtime) -> response dict


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/ready")
def ready():
    """Real readiness probe (unlike /api/health, which is pure liveness and
    doesn't check anything). 503 until winner/method/reference_data are up —
    startup() now raises on missing required artifacts, so a green /api/ready
    means predictions are actually backed by real models, not silent
    prob=0.5 / base-rate fallbacks. props_loaded is expected False until the
    first prop request (lazy-loaded to fit the free-tier memory limit)."""
    components = service.readiness()
    all_required_ok = components["winner"] and components["method"] and components["reference_data"]
    return JSONResponse(content=components, status_code=200 if all_required_ok else 503)


# ── Meta ──────────────────────────────────────────────────────────────────────

@app.get("/api/meta")
def meta():
    from ufc.io import paths
    models = service.get_models()
    version = "unknown"
    last_sync = "unknown"
    model_dir = paths.outputs_models()
    winner_files = sorted(model_dir.glob("winner_ensemble_*.joblib"),
                          key=lambda p: p.stat().st_mtime)
    if winner_files:
        p = winner_files[-1]
        version = p.stem.split("_")[-1]
        last_sync = p.stat().st_mtime
        from datetime import datetime
        last_sync = datetime.fromtimestamp(float(last_sync)).strftime("%Y-%m-%d")

    # Best-effort backtest metrics from most recent report
    hit_rate = 0.0
    roi = 0.0
    reports_dir = paths.outputs_reports()
    bt_files = sorted(reports_dir.glob("backtest_*.md"),
                      key=lambda p: p.stat().st_mtime)
    if bt_files:
        txt = bt_files[-1].read_text(encoding="utf-8")
        import re
        m = re.search(r"accuracy[:\s]+([0-9.]+)", txt, re.I)
        if m:
            hit_rate = float(m.group(1))
        m = re.search(r"roi[:\s]+([0-9.-]+)", txt, re.I)
        if m:
            roi = float(m.group(1))

    return {
        "version": version,
        "lastSync": last_sync,
        "cardsAnalyzed": len(_list_card_files()),
        "hitRate": hit_rate,
        "roi": roi,
        "units": roi,
        "calib": [],
    }


# ── Card list ─────────────────────────────────────────────────────────────────

@app.get("/api/cards")
def list_cards():
    items = []
    for p in _list_card_files():
        data = _load_card_json(p)
        item = _card_list_item(p, data)
        # Keep cards through fight night + grace window (see _card_still_listed).
        if not _card_still_listed(data):
            continue
        item["_sortDate"] = item.get("eventDate", "")
        items.append(item)
    # Chronological order: next upcoming first
    items.sort(key=lambda x: x["_sortDate"])
    for item in items:
        item.pop("_sortDate", None)
    return items


@app.post("/api/cards/refresh")
def refresh_cards():
    global _last_refresh
    if time.time() - _last_refresh < 60:
        raise HTTPException(status_code=429, detail="Refresh rate limit: wait 60 s between refreshes")
    _last_refresh = time.time()
    try:
        from ufc.ingest.scrape_upcoming import scrape_upcoming_cards
        scrape_upcoming_cards()
    except Exception:
        logger.exception("card refresh failed")
        raise HTTPException(status_code=500, detail="Card refresh failed")
    _card_cache.clear()
    return list_cards()


# ── Card predict ──────────────────────────────────────────────────────────────

@app.get("/api/cards/{card_id}")
def get_card(card_id: str):
    _validate_card_id(card_id)
    path = _cards_dir() / f"{card_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Card '{card_id}' not found")

    cache_key = (card_id, path.stat().st_mtime)
    if cache_key in _card_cache:
        return _card_cache[cache_key]

    data = _load_card_json(path)

    event_date_str = data.get("event_date", str(date.today()))
    ev_date = _parse_event_date(event_date_str)

    fights_out: list[dict] = []
    errors: list[str] = []
    unavailable: list[dict] = []
    for idx, m in enumerate(data.get("matchups", [])):
        # ufcstats gives no broadcast segmentation; modern cards run a 5-fight main
        # card, so default to that. Odd cards (6-fight mains, early prelims) can set
        # card_segment per matchup via cards/overrides.json.
        slot = "title" if m.get("is_title") else (
            m.get("card_segment") or ("main" if idx < 5 else "prelim")
        )
        try:
            pred = service.predict(
                red=m["red"],
                blue=m["blue"],
                rounds=int(m.get("scheduled_rounds", 3)),
                is_title=bool(m.get("is_title", False)),
                event_date=ev_date,
                weight_class=m.get("weight_class") or None,
                referee=m.get("referee", ""),
                location=data.get("location", ""),
            )
            fight_dict = serialize.serialize_fight(pred, slot=slot, idx=idx)
            # Card-specified class wins (cleaned); otherwise keep serialize's inferred value.
            if m.get("weight_class"):
                fight_dict["weightClass"] = serialize._clean_weight_class(m.get("weight_class"))
            fights_out.append(fight_dict)
        except Exception as exc:
            logger.warning("predict failed for %s vs %s: %s", m["red"], m["blue"], exc)
            errors.append(f"{m['red']} vs {m['blue']}: prediction unavailable")
            # Structured stub, kept OUT of fights_out: downstream consumers (Best
            # Bets, Portfolio, Prop Lab, Kalshi pricer, filters.js, serialize) all
            # assume every fights[] entry has win probabilities.
            unavailable.append({
                "id": f"unavail_{idx}",
                "idx": idx,
                "slot": slot,
                "red": m["red"],
                "blue": m["blue"],
                "rounds": int(m.get("scheduled_rounds", 3)),
                "isTitle": bool(m.get("is_title", False)),
                "reason": str(exc),
            })
            continue

    event = {
        "code": card_id.upper()[:8],
        "name": data.get("event_name", card_id.replace("_", " ").title()),
        "venue": data.get("location", ""),
        "date": event_date_str,
    }
    response = {"event": event, "fights": fights_out}
    if errors:
        response["errors"] = errors
    if unavailable:
        response["unavailableFights"] = unavailable

    _card_cache[cache_key] = response
    return response


# ── Exchange (Kalshi) market lines ──────────────────────────────────────────

_market_lines_cache: dict[str, tuple[float, dict]] = {}   # card_id -> (fetched_at, response)
_MARKET_LINES_TTL_SEC = 60.0
_MARKET_LINES_FRESH_FLOOR_SEC = 15.0   # min age before ?fresh=1 re-pulls (public endpoint)

# Model/market divergence guard (v8.39): a gap this large historically means
# EITHER a real edge OR information the feature store can't see (late injury,
# a last fight whose result contradicts its stat line). Flag, don't hide.
_DIVERGENCE_PP = 0.20


def _divergence_flag(model_p: float, ask: float) -> bool:
    return abs(float(model_p) - float(ask)) >= _DIVERGENCE_PP


@app.get("/api/market-lines/{card_id}")
def get_market_lines(card_id: str, fresh: bool = False):
    """Kalshi quotes for a card across all six market families (winner, method,
    distance, mof, rounds, vicround), priced against the same predictions served
    by /api/cards/{card_id}. Each row's `paper` flag is gated per market kind
    (configs/market_advice.yaml status_by_kind) — a kind only flips to live once
    its own forward ledger clears the pre-registered CLV bar; every kind starts
    paper. DFS round/duration advising (05c) is unaffected and out of scope here.
    `?fresh=1` bypasses the 60s TTL, but never bypasses the 15s floor."""
    _validate_card_id(card_id)
    path = _cards_dir() / f"{card_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Card '{card_id}' not found")

    cached = _market_lines_cache.get(card_id)
    if cached is not None:
        age = time.time() - cached[0]
        if age < _MARKET_LINES_FRESH_FLOOR_SEC or (not fresh and age < _MARKET_LINES_TTL_SEC):
            return cached[1]

    from ufc.ingest.market_lines import fetch_all_markets, resolve_markets_to_card
    from ufc.valuation.market_edge import evaluate_market_quote, model_prob_for_quote

    data = _load_card_json(path)
    ev_date = _parse_event_date(data.get("event_date", str(date.today())))

    card_matchups = [
        (
            m["red"], m["blue"], int(m.get("scheduled_rounds", 3)),
            bool(m.get("is_title", False)), ev_date,
            m.get("weight_class") or None, m.get("referee", ""), data.get("location", ""),
        )
        for m in data.get("matchups", [])
    ]
    if not card_matchups:
        response = {"rows": [], "errors": ["Card has no matchups"], "fetchedAt": time.time()}
        _market_lines_cache[card_id] = (time.time(), response)
        return response

    quotes, errors = fetch_all_markets(with_depth=True)
    if not quotes:
        response = {"rows": [], "errors": errors, "fetchedAt": time.time()}
        _market_lines_cache[card_id] = (time.time(), response)
        return response

    resolved, _unresolved = resolve_markets_to_card(quotes, card_matchups)
    advice_cfg = _load_market_advice_cfg()

    preds: dict[int, object] = {}
    rows: list[dict] = []
    for rq in resolved:
        if rq.yes_ask is None:
            continue
        fi = rq.fight_idx
        if fi not in preds:
            red, blue, rds, is_title, edate, wc, ref, loc = card_matchups[fi]
            try:
                preds[fi] = service.predict(
                    red=red, blue=blue, rounds=rds, is_title=is_title,
                    event_date=edate, weight_class=wc, referee=ref, location=loc,
                )
            except Exception as exc:
                logger.warning("market-lines predict failed for %s vs %s: %s", red, blue, exc)
                preds[fi] = None
        pred = preds[fi]
        if pred is None:
            continue

        model_p = model_prob_for_quote(rq, pred)
        if model_p is None:
            continue

        edge = evaluate_market_quote(rq, model_p)
        rows.append({
            "fightIdx": fi,
            "fighter": rq.fighter_name,
            "corner": rq.corner,
            "marketKind": rq.market_kind,
            "modelP": round(model_p, 4),
            "ask": rq.yes_ask,
            "bid": rq.yes_bid,
            "feeAdjBE": round(edge.breakeven, 4),
            "edgePct": round(edge.edge_pct, 4),
            "kelly": round(edge.kelly, 4),
            "stakeCapUsd": edge.stake_cap_usd,
            "depthUsd": rq.depth_usd_3c,
            "liqTier": edge.liq_tier,
            "volume": rq.volume,
            "venue": "kalshi",
            "paper": _advice_paper_for_kind(advice_cfg, rq.market_kind),
            "divergence": _divergence_flag(model_p, rq.yes_ask),
            "ticker": rq.market_ticker,
        })

    response = {"rows": rows, "errors": errors, "fetchedAt": time.time()}
    _market_lines_cache[card_id] = (time.time(), response)
    return response


# ── Manual prediction ─────────────────────────────────────────────────────────

@app.post("/api/predict")
def predict_manual(req: schemas.PredictRequest):
    try:
        ev_date = date.fromisoformat(req.event_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid event_date; expected YYYY-MM-DD")
    try:
        pred = service.predict(
            red=req.red,
            blue=req.blue,
            rounds=req.rounds,
            is_title=req.is_title,
            event_date=ev_date,
            weight_class=req.weight_class,
            referee=req.referee,
            location=req.location,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    fight_dict = serialize.serialize_fight(pred, slot="manual", idx=0)
    # Prefer the explicit override; otherwise keep the inferred weight class from serialize.
    if req.weight_class:
        fight_dict["weightClass"] = req.weight_class
    return fight_dict


# ── Reference data ────────────────────────────────────────────────────────────

@app.get("/api/fighters")
def get_fighters():
    fighters_df = service.get_fighters_df()
    col = "name" if "name" in fighters_df.columns else fighters_df.columns[0]
    names = sorted(fighters_df[col].dropna().astype(str).tolist())
    return names


@app.get("/api/referees")
def get_referees():
    from ufc.inference.ref_history import known_referee_names
    return known_referee_names()


@app.get("/api/weight-classes")
def get_weight_classes():
    return [
        "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
        "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
        "Heavyweight", "Women's Strawweight", "Women's Flyweight",
        "Women's Bantamweight", "Women's Featherweight",
    ]


# ── Positions ─────────────────────────────────────────────────────────────────





# ── Prop trust tiers endpoint ────────────────────────────────────────────────

@app.get("/api/prop-trust")
def get_prop_trust():
    """Return prop trust tiers keyed by frontend market key.

    TRUST  = real resolution + edge-pick CI clears breakeven.
    WATCH  = calibrated, weak/uncertain edge; monitor via ledger.
    CUT    = no resolution / structurally unpredictable; flagged low-confidence.
    """
    return _PROP_TRUST_FRONTEND


# ── Live lines ────────────────────────────────────────────────────────────────

# Power Play/Flat Multi are unofficial endpoints behind Cloudflare that have already
# rate-limited/blocked this account once (see prop_lines.py kill-switch history).
# This route previously had NO cache, so every page load / bot hit / repeat request
# fired a real outbound call to both platforms — mirror the proven _market_lines_cache
# pattern (below) with a single global slot (route takes no params, unlike per-card
# market-lines) so any burst of requests collapses to at most one real fetch per TTL.
_live_lines_cache: tuple[float, dict] | None = None
_LIVE_LINES_TTL_SEC = 90.0



# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.post("/api/portfolio")
def grade_portfolio(req: schemas.PortfolioRequest):
    if not req.legs:
        return {
            "nLegs": 0, "individualProbs": [], "naiveJointProb": 1.0,
            "mcJointProb": 1.0, "correlationAdj": 0.0,
            "ev": 0.0, "breakeven": 0.0, "kelly": 0.0, "verdict": "empty",
        }

    n_legs = _PAYOUT_LEGS.get(req.payout_key, 2)
    mult = req.mult

    probs = [leg.model_p for leg in req.legs]
    naive_joint = float(np.prod(probs))

    # Simple MC correlation model (no sim_samples on client legs)
    rng = np.random.default_rng(42)
    n = 50000
    hit_matrix = np.ones(n, dtype=bool)
    for p in probs:
        hits = rng.random(n) < p
        hit_matrix &= hits
    mc_joint = float(hit_matrix.mean())
    corr_adj = mc_joint - naive_joint

    breakeven = mult ** (-1.0 / n_legs)
    ev = mc_joint * mult - 1.0
    kelly = max(0.0, (mc_joint - breakeven) / (1.0 - breakeven)) if breakeven < 1.0 else 0.0

    if ev > 0.10:
        verdict = "strong"
    elif ev > 0:
        verdict = "marginal"
    else:
        verdict = "negative"

    return {
        "nLegs":           len(req.legs),
        "individualProbs": [round(p, 4) for p in probs],
        "naiveJointProb":  round(naive_joint, 4),
        "mcJointProb":     round(mc_joint, 4),
        "correlationAdj":  round(corr_adj, 4),
        "ev":              round(ev, 4),
        "breakeven":       round(breakeven, 4),
        "kelly":           round(kelly, 4),
        "verdict":         verdict,
    }


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history")
def get_history(n_events: int = Query(10, ge=1, le=200, alias="n_events")):
    feed = service.get_history_feed(n_events=n_events)
    return feed


@app.get("/api/ledger_summary")
def get_ledger_summary():
    return service.get_ledger_summary()


# ── Export ────────────────────────────────────────────────────────────────────

@app.get("/api/export/card/{card_id}.csv")
def export_card(card_id: str):
    card = get_card(card_id)
    rows = ["fight,slot,rounds,redName,blueName,pRed,pBlue,ko,sub,dec,inside,medianMin"]
    for f in card["fights"]:
        rows.append(",".join(str(v) for v in [
            f"{f['a']['name']} vs {f['b']['name']}",
            f["slot"], f["rounds"],
            f["a"]["name"], f["b"]["name"],
            f["a"]["pWin"], f["b"]["pWin"],
            f["method"]["ko"], f["method"]["sub"], f["method"]["dec"],
            f["inside"], f.get("medianMin", ""),
        ]))
    csv_text = "\n".join(rows)
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={card_id}.csv"},
    )


