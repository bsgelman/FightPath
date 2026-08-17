"""Fetch and parse live prediction-market quotes from Kalshi (public REST, no auth).

Kalshi's public market data endpoints require no authentication (~30 req/s rate
limit). Schema notes verified live 2026-07-02/03 against
https://api.elections.kalshi.com/trade-api/v2:
  - Winner markets (series KXUFCFIGHT): one binary market PER FIGHTER. Fighter
    identity comes from `yes_sub_title` ("Benoit Saint-Denis") — clean, no regex
    needed. Title-regex is kept only as a fallback for schema drift.
  - Method markets (series KXUFCMOV): one binary market per (fighter, method).
    `custom_strike` is structured: {"Method": "Submission"|"KO/TKO/DQ"|"Decision",
    "Participant": "<fighter name>"} — again no regex needed. Method markets open
    ~fight week; absence is normal, not an error.
  - Distance (KXUFCDISTANCE): one binary market PER FIGHT ("goes the distance"),
    ticker suffix "-DIST". No custom_strike, no fighter identity — fight-level.
  - Method-of-finish (KXUFCMOF): one binary market PER FIGHT per method — "either
    competitor wins by X". `custom_strike` = {"Method": "Submission"|"KO/TKO/DQ"|
    "Decision"|"Draw/No Contest"} (no "Participant" key — fight-level). The
    "-DRAW" suffix / "Draw/No Contest" method is intentionally skipped: the model
    has no draw probability, matching the existing unmapped-outcome-skip
    convention used for method markets.
  - Rounds (KXUFCROUNDS): one binary market PER FIGHT per round boundary — "ends
    before round r", ticker suffix is a bare digit (observed 2, 3; any digit must
    be accepted since 5-round main events add more boundaries). Fight-level.
  - Round of victory (KXUFCVICROUND): one binary market per (fighter, round) —
    "X to win in Round r" — PLUS one fight-level "-OTHER" market ("Decision /
    Draw / No Contest"). `custom_strike` = {"Participant": "<fighter name>"|
    "Other", "Round": "<digit>"|"Decision / Draw / No Contest"}. Per Kalshi's
    rules text (verified live): OTHER settles YES on a no-contest too (a
    house-convention override — every other Kalshi kind here voids on NC), and a
    technical decision counts as OTHER, not a round win; DQ counts as a round win
    (rules_primary explicitly lists "KO, TKO, Submission, or Disqualification").
  - Prices are `*_dollars` string fields (e.g. "0.5600"); volume/open_interest are
    `*_fp` string fields.
  - Never key fighter identity off the ticker suffix (e.g. "-SAI", "-NAS3") — it's
    a truncated/ambiguous fragment, not reliable for disambiguation. Fighter
    identity comes only from `custom_strike.Participant` or a title/sub_title
    regex fallback.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional

import requests

from ufc.ingest.name_norm import normalize as _normalize, token_surname_match as _token_surname_match

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_WINNER = "KXUFCFIGHT"
SERIES_METHOD = "KXUFCMOV"
SERIES_DISTANCE = "KXUFCDISTANCE"
SERIES_MOF = "KXUFCMOF"
SERIES_ROUNDS = "KXUFCROUNDS"
SERIES_VICROUND = "KXUFCVICROUND"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

_METHOD_MAP = {
    "submission": "method_sub",
    "decision": "method_dec",
    # "KO/TKO/DQ" and any KO-bearing label -> method_ko (checked via substring below)
}

# Fight-level method-of-finish (KXUFCMOF) — same Method vocabulary as KXUFCMOV,
# mapped to fight-level kinds instead of per-fighter ones. "Draw/No Contest" has
# no entry (skipped, matching the -DRAW-suffix skip below).
_MOF_METHOD_MAP = {
    "submission": "mof_sub",
    "decision": "mof_dec",
}
# Ticker-suffix fallback for KXUFCMOF when custom_strike is absent (schema drift).
_MOF_SUFFIX_MAP = {"KOTKODQ": "mof_ko", "SUB": "mof_sub", "DEC": "mof_dec"}  # -DRAW omitted

_EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_TITLE_FIGHTER_RE = re.compile(r"^Will (.+?) win the", re.IGNORECASE)
_ROUNDS_SUFFIX_RE = re.compile(r"^(\d+)$")
_VICROUND_TITLE_RE = re.compile(r"^(.+?)\s+to win in Round\s+(\d+)$", re.IGNORECASE)

_FIGHT_LEVEL_EXACT_KINDS = frozenset({"distance", "mof_ko", "mof_sub", "mof_dec", "vicround_other"})


def is_fight_level_kind(market_kind: str) -> bool:
    """True for Kalshi market kinds with no single-fighter identity (priced/graded
    at the fight level, resolved to a card fight via event-ticker code rather than
    fighter-name matching)."""
    return market_kind in _FIGHT_LEVEL_EXACT_KINDS or market_kind.startswith("end_before_r")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketQuote:
    platform: str            # 'kalshi'
    event_ticker: str
    market_ticker: str
    market_kind: str         # 'winner' | 'method_ko' | 'method_sub' | 'method_dec' |
                             # 'distance' | 'mof_ko' | 'mof_sub' | 'mof_dec' |
                             # 'end_before_r{N}' | 'win_in_r{N}' | 'vicround_other'
    fighter_name: str
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    last_price: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]
    depth_usd_3c: Optional[float] = None   # populated only when orderbook was fetched
    fetched_at: str = ""


@dataclass
class ResolvedMarketQuote(MarketQuote):
    card_red: str = ""
    card_blue: str = ""
    fight_idx: int = -1
    corner: str = ""


# ---------------------------------------------------------------------------
# Small parse helpers
# ---------------------------------------------------------------------------

def _dollars(raw) -> Optional[float]:
    """Parse a Kalshi '*_dollars' string field. None/empty/unparsable -> None."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _parse_event_date(event_ticker: str) -> Optional[date]:
    """Extract the event date from a Kalshi event ticker, e.g.
    'KXUFCFIGHT-26JUL11SAIPIM' -> date(2026, 7, 11). None if the pattern is absent."""
    m = _EVENT_DATE_RE.search(event_ticker or "")
    if not m:
        return None
    yy, mon, dd = m.groups()
    try:
        return datetime.strptime(f"{yy}{mon.title()}{dd}", "%y%b%d").date()
    except ValueError:
        return None


def _map_method(method_label: str) -> Optional[str]:
    """Map a Kalshi custom_strike.Method value to a canonical market_kind suffix."""
    norm = (method_label or "").strip().lower()
    if "ko" in norm or "tko" in norm:
        return "method_ko"
    return _METHOD_MAP.get(norm)


def _extract_fighter_name_from_title(title: str) -> Optional[str]:
    m = _TITLE_FIGHTER_RE.match(title or "")
    return m.group(1).strip() if m else None


def _ticker_suffix(ticker: str) -> str:
    """The last '-'-delimited segment of a Kalshi market ticker, e.g.
    'KXUFCROUNDS-26JUN27ABDNAS-3' -> '3'. Used only to detect market TYPE
    (which series-specific bucket a market falls into) — never to derive
    fighter identity (see module docstring)."""
    return (ticker or "").rsplit("-", 1)[-1]


def _event_code(event_ticker: str) -> str:
    """The fight-identity portion of an event ticker, e.g.
    'KXUFCDISTANCE-26JUN27ABDNAS' -> '26JUN27ABDNAS'. Shared across every series
    for the same fight, so it's used to join fight-level quotes (no fighter
    identity) to a fight already resolved by name from another series."""
    parts = (event_ticker or "").split("-", 1)
    return parts[1] if len(parts) == 2 else event_ticker


def _map_mof_method(method_label: str) -> Optional[str]:
    """Map a KXUFCMOF custom_strike.Method value to a fight-level market_kind."""
    norm = (method_label or "").strip().lower()
    if "ko" in norm or "tko" in norm:
        return "mof_ko"
    return _MOF_METHOD_MAP.get(norm)


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Accept": "application/json"})
    return s


# ---------------------------------------------------------------------------
# Pure parsers (no network — testable with fixtures)
# ---------------------------------------------------------------------------

def _parse_winner_markets(payload: dict) -> list[MarketQuote]:
    """Pure parse of a KXUFCFIGHT /markets response into per-fighter MarketQuotes."""
    quotes: list[MarketQuote] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for m in payload.get("markets", []):
        fighter_name = m.get("yes_sub_title") or _extract_fighter_name_from_title(m.get("title", ""))
        if not fighter_name:
            continue
        quotes.append(MarketQuote(
            platform="kalshi",
            event_ticker=m.get("event_ticker", ""),
            market_ticker=m.get("ticker", ""),
            market_kind="winner",
            fighter_name=fighter_name,
            yes_bid=_dollars(m.get("yes_bid_dollars")),
            yes_ask=_dollars(m.get("yes_ask_dollars")),
            last_price=_dollars(m.get("last_price_dollars")),
            volume=_dollars(m.get("volume_fp")),
            open_interest=_dollars(m.get("open_interest_fp")),
            fetched_at=fetched_at,
        ))
    return quotes


def _parse_method_markets(payload: dict) -> list[MarketQuote]:
    """Pure parse of a KXUFCMOV /markets response into per-(fighter,method) MarketQuotes.

    Markets whose custom_strike is missing or whose Method isn't KO/Submission/Decision
    are skipped (schema drift or an unmapped outcome) — the winner lane must not be
    blocked by a method-market shape change."""
    quotes: list[MarketQuote] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for m in payload.get("markets", []):
        strike = m.get("custom_strike") or {}
        method_label = strike.get("Method")
        fighter_name = strike.get("Participant")
        if not method_label or not fighter_name:
            continue
        kind = _map_method(method_label)
        if kind is None:
            continue
        quotes.append(MarketQuote(
            platform="kalshi",
            event_ticker=m.get("event_ticker", ""),
            market_ticker=m.get("ticker", ""),
            market_kind=kind,
            fighter_name=fighter_name,
            yes_bid=_dollars(m.get("yes_bid_dollars")),
            yes_ask=_dollars(m.get("yes_ask_dollars")),
            last_price=_dollars(m.get("last_price_dollars")),
            volume=_dollars(m.get("volume_fp")),
            open_interest=_dollars(m.get("open_interest_fp")),
            fetched_at=fetched_at,
        ))
    return quotes


def _parse_distance_markets(payload: dict) -> list[MarketQuote]:
    """Pure parse of a KXUFCDISTANCE /markets response — one fight-level market
    per fight ("goes the distance"), identified by the '-DIST' ticker suffix."""
    quotes: list[MarketQuote] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for m in payload.get("markets", []):
        ticker = m.get("ticker", "")
        if _ticker_suffix(ticker) != "DIST":
            continue
        quotes.append(MarketQuote(
            platform="kalshi",
            event_ticker=m.get("event_ticker", ""),
            market_ticker=ticker,
            market_kind="distance",
            fighter_name="",
            yes_bid=_dollars(m.get("yes_bid_dollars")),
            yes_ask=_dollars(m.get("yes_ask_dollars")),
            last_price=_dollars(m.get("last_price_dollars")),
            volume=_dollars(m.get("volume_fp")),
            open_interest=_dollars(m.get("open_interest_fp")),
            fetched_at=fetched_at,
        ))
    return quotes


def _parse_mof_markets(payload: dict) -> list[MarketQuote]:
    """Pure parse of a KXUFCMOF /markets response — one fight-level market per
    method ("either competitor wins by X"). Prefers custom_strike.Method (no
    Participant key — this series has no per-fighter identity); falls back to
    the ticker suffix if custom_strike is missing (schema drift). The Draw/No
    Contest outcome (suffix '-DRAW') is intentionally unmapped and skipped."""
    quotes: list[MarketQuote] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for m in payload.get("markets", []):
        ticker = m.get("ticker", "")
        strike = m.get("custom_strike") or {}
        method_label = strike.get("Method")
        if method_label:
            kind = _map_mof_method(method_label)
        else:
            kind = _MOF_SUFFIX_MAP.get(_ticker_suffix(ticker))
        if kind is None:
            continue
        quotes.append(MarketQuote(
            platform="kalshi",
            event_ticker=m.get("event_ticker", ""),
            market_ticker=ticker,
            market_kind=kind,
            fighter_name="",
            yes_bid=_dollars(m.get("yes_bid_dollars")),
            yes_ask=_dollars(m.get("yes_ask_dollars")),
            last_price=_dollars(m.get("last_price_dollars")),
            volume=_dollars(m.get("volume_fp")),
            open_interest=_dollars(m.get("open_interest_fp")),
            fetched_at=fetched_at,
        ))
    return quotes


def _parse_rounds_markets(payload: dict) -> list[MarketQuote]:
    """Pure parse of a KXUFCROUNDS /markets response — one fight-level market per
    round boundary ("ends before round r"). The ticker suffix is a bare digit;
    any digit is accepted (5-round main events add more boundaries than the
    2/3 observed on prelims)."""
    quotes: list[MarketQuote] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for m in payload.get("markets", []):
        ticker = m.get("ticker", "")
        mm = _ROUNDS_SUFFIX_RE.match(_ticker_suffix(ticker))
        if not mm:
            continue
        r = int(mm.group(1))
        quotes.append(MarketQuote(
            platform="kalshi",
            event_ticker=m.get("event_ticker", ""),
            market_ticker=ticker,
            market_kind=f"end_before_r{r}",
            fighter_name="",
            yes_bid=_dollars(m.get("yes_bid_dollars")),
            yes_ask=_dollars(m.get("yes_ask_dollars")),
            last_price=_dollars(m.get("last_price_dollars")),
            volume=_dollars(m.get("volume_fp")),
            open_interest=_dollars(m.get("open_interest_fp")),
            fetched_at=fetched_at,
        ))
    return quotes


def _parse_vicround_markets(payload: dict) -> list[MarketQuote]:
    """Pure parse of a KXUFCVICROUND /markets response — one market per
    (fighter, round) ("X to win in Round r") plus one fight-level '-OTHER'
    market ("Decision / Draw / No Contest"). Prefers custom_strike (Participant/
    Round); falls back to the '-OTHER' suffix or a yes_sub_title regex ("<name>
    to win in Round <r>") if custom_strike is missing. Never derives a fighter
    name from the ticker's ABBR fragment — an unparseable market is skipped."""
    quotes: list[MarketQuote] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for m in payload.get("markets", []):
        ticker = m.get("ticker", "")
        strike = m.get("custom_strike") or {}
        participant = strike.get("Participant")

        kind: Optional[str] = None
        fighter_name = ""
        if participant is not None:
            if participant == "Other":
                kind = "vicround_other"
            else:
                try:
                    r = int(str(strike.get("Round", "")).strip())
                except (TypeError, ValueError):
                    continue
                kind, fighter_name = f"win_in_r{r}", participant
        elif _ticker_suffix(ticker) == "OTHER":
            kind = "vicround_other"
        else:
            title_match = _VICROUND_TITLE_RE.match(m.get("yes_sub_title", "") or "")
            if not title_match:
                continue
            fighter_name, r = title_match.group(1).strip(), int(title_match.group(2))
            kind = f"win_in_r{r}"

        quotes.append(MarketQuote(
            platform="kalshi",
            event_ticker=m.get("event_ticker", ""),
            market_ticker=ticker,
            market_kind=kind,
            fighter_name=fighter_name,
            yes_bid=_dollars(m.get("yes_bid_dollars")),
            yes_ask=_dollars(m.get("yes_ask_dollars")),
            last_price=_dollars(m.get("last_price_dollars")),
            volume=_dollars(m.get("volume_fp")),
            open_interest=_dollars(m.get("open_interest_fp")),
            fetched_at=fetched_at,
        ))
    return quotes


def _orderbook_depth_usd(orderbook: dict, best_ask: float, cents: int = 3) -> float:
    """Approximate resting liquidity available to a yes-side taker near best_ask.

    Kalshi's orderbook returns bid-only ladders for yes and no (no asks). Crossing
    the yes ask consumes no-side bids priced near (1 - best_ask) — a no bid at price
    q is economically a yes ask at (1-q). Sum no-bid depth within `cents` of that
    implied yes-ask price, expressed in yes-equivalent dollars (qty * (1-price)).
    A liquidity-tier proxy only — not an execution/fill-price guarantee.
    """
    book = orderbook.get("orderbook_fp") or {}
    no_levels = book.get("no_dollars") or []
    threshold = cents / 100.0
    total = 0.0
    for price_str, qty_str in no_levels:
        try:
            price = float(price_str)
            qty = float(qty_str)
        except (ValueError, TypeError):
            continue
        yes_ask_equiv = 1.0 - price
        if abs(yes_ask_equiv - best_ask) <= threshold + 1e-9:
            total += qty * yes_ask_equiv
    return total


# ---------------------------------------------------------------------------
# Network fetchers
# ---------------------------------------------------------------------------

def _fetch_markets_page(session: requests.Session, series_ticker: str, status: str,
                         cursor: str = "", timeout: int = 10) -> dict:
    params = {"series_ticker": series_ticker, "status": status, "limit": 100}
    if cursor:
        params["cursor"] = cursor
    resp = session.get(f"{KALSHI_BASE}/markets", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _fetch_all_pages(session: requests.Session, series_ticker: str, status: str,
                      timeout: int = 10, max_pages: int = 10) -> dict:
    """Follow cursor pagination, merging 'markets' lists into one payload."""
    merged: dict = {"markets": []}
    cursor = ""
    for _ in range(max_pages):
        page = _fetch_markets_page(session, series_ticker, status, cursor, timeout)
        merged["markets"].extend(page.get("markets", []))
        cursor = page.get("cursor", "")
        if not cursor:
            break
    return merged


def fetch_kalshi_winner(session: requests.Session | None = None, timeout: int = 10) -> list[MarketQuote]:
    """Fetch open KXUFCFIGHT (winner) markets. Raises on network/parse failure."""
    s = session or _make_session()
    payload = _fetch_all_pages(s, SERIES_WINNER, "open", timeout)
    return _parse_winner_markets(payload)


def fetch_kalshi_method(session: requests.Session | None = None, timeout: int = 10) -> list[MarketQuote]:
    """Fetch open KXUFCMOV (method) markets. Raises on network/parse failure.

    Absence of open method markets is normal (they open ~fight week) — callers
    should treat an empty list as non-fatal, distinct from a raised exception."""
    s = session or _make_session()
    payload = _fetch_all_pages(s, SERIES_METHOD, "open", timeout)
    return _parse_method_markets(payload)


@dataclass(frozen=True)
class SeriesSpec:
    """One Kalshi series in the advising lane: which endpoint to hit, how to
    parse it, and how failures/depth-fetching should be handled."""
    series_ticker: str
    parser: Callable[[dict], list[MarketQuote]]
    required: bool    # True only for winner: a failure here IS a lane-level error
    with_depth: bool  # orderbook depth is only fetched for winner/method (keeps
                      # public-API call volume down for the four newer series)
    label: str        # short name used in human-readable error strings


# MUST keep KXUFCFIGHT first and required=True: it's the only lane a failure in
# any other series is guaranteed not to affect (see fetch_all_markets below).
SERIES_TABLE: tuple[SeriesSpec, ...] = (
    SeriesSpec(SERIES_WINNER, _parse_winner_markets, True, True, "winner"),
    SeriesSpec(SERIES_METHOD, _parse_method_markets, False, True, "method"),
    SeriesSpec(SERIES_DISTANCE, _parse_distance_markets, False, False, "distance"),
    SeriesSpec(SERIES_MOF, _parse_mof_markets, False, False, "mof"),
    SeriesSpec(SERIES_ROUNDS, _parse_rounds_markets, False, False, "rounds"),
    SeriesSpec(SERIES_VICROUND, _parse_vicround_markets, False, False, "vicround"),
)


def _fetch_series(session: requests.Session, spec: SeriesSpec, timeout: int) -> list[MarketQuote]:
    payload = _fetch_all_pages(session, spec.series_ticker, "open", timeout)
    return spec.parser(payload)


def fetch_orderbook_depth(market_ticker: str, best_ask: float,
                           session: requests.Session | None = None, timeout: int = 10) -> Optional[float]:
    """Fetch orderbook depth (yes-equivalent $ within 3c of best_ask) for one market.
    Returns None on any failure — depth is advisory, never blocks a quote."""
    s = session or _make_session()
    try:
        resp = s.get(f"{KALSHI_BASE}/markets/{market_ticker}/orderbook", timeout=timeout)
        resp.raise_for_status()
        return _orderbook_depth_usd(resp.json(), best_ask)
    except Exception:
        return None


def fetch_all_markets(
    platforms: tuple[str, ...] = ("kalshi",),
    timeout: int = 10,
    with_depth: bool = True,
) -> tuple[list[MarketQuote], list[str]]:
    """Fetch all six Kalshi series (winner, method, distance, mof, rounds,
    vicround). Never raises — per-source errors are returned as human-readable
    strings (pattern mirrors prop_lines.fetch_all). A failure in ANY series
    other than winner can never block quotes already collected from the others
    (each series is fetched in its own try/except). Writes last_pull.json + a
    timestamped history snapshot for offline debugging and CLV tracking."""
    from ufc.io import paths

    if "kalshi" not in platforms:
        return [], [f"Unknown platform(s) {platforms!r} — only 'kalshi' is supported"]

    session = _make_session()
    quotes: list[MarketQuote] = []
    errors: list[str] = []
    depth_quotes: list[MarketQuote] = []

    for spec in SERIES_TABLE:
        try:
            series_quotes = _fetch_series(session, spec, timeout)
        except Exception as exc:
            errors.append(f"Kalshi {spec.label}: {exc}")
            continue
        if spec.required and not series_quotes:
            errors.append(f"Kalshi {spec.label}: 0 open markets (no UFC card currently listed)")
        quotes.extend(series_quotes)
        if spec.with_depth:
            depth_quotes.extend(series_quotes)

    if with_depth:
        for q in depth_quotes:
            if q.yes_ask is not None:
                q.depth_usd_3c = fetch_orderbook_depth(q.market_ticker, q.yes_ask, session=session, timeout=timeout)

    if quotes:
        try:
            out_dir = paths.external_market_lines()
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            payload = [
                {
                    "platform": q.platform, "event_ticker": q.event_ticker,
                    "market_ticker": q.market_ticker, "market_kind": q.market_kind,
                    "fighter_name": q.fighter_name, "yes_bid": q.yes_bid, "yes_ask": q.yes_ask,
                    "last_price": q.last_price, "volume": q.volume, "open_interest": q.open_interest,
                    "depth_usd_3c": q.depth_usd_3c, "fetched_at": q.fetched_at,
                }
                for q in quotes
            ]
            with open(out_dir / "last_pull.json", "w") as f:
                json.dump(payload, f, indent=2)
            hist_dir = out_dir / "history"
            hist_dir.mkdir(exist_ok=True)
            with open(hist_dir / f"{stamp}_kalshi.json", "w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    return quotes, errors


# ---------------------------------------------------------------------------
# Card resolution
# ---------------------------------------------------------------------------

def resolve_markets_to_card(
    quotes: list[MarketQuote],
    card_matchups: list,
    fighters_df=None,
) -> tuple[list[ResolvedMarketQuote], list[str]]:
    """Match Kalshi fighter names to fighters on the loaded card (card-restricted).

    card_matchups: list of 8-tuples (red, blue, rounds, is_title, event_date,
                   weight_class, referee, location) — same shape as prop_lines.

    Reuses prop_lines' token-aware surname matcher (the fuzzy/override machinery
    lives there; name matching is the historical bug source, e.g. "Aswell Jr." —
    do not reimplement it here). Adds an event-date guard: a quote's event_ticker
    date must be within 1 day of the matched card's event_date, so the same
    fighter name on two different cards can't cross-resolve.

    Fight-level quotes (distance, mof_*, end_before_r*, vicround_other — no
    fighter identity, see is_fight_level_kind) can't go through name matching at
    all, so they're resolved in a second pass: each series shares its event
    ticker's fight-identity code (e.g. "26JUN27ABDNAS") with every other series
    for the same fight, so a fight-level quote joins onto whichever fight a
    NAMED quote (pass 1, above) already resolved for that code. The same
    event-date guard applies to the fight-level quote's own event_ticker.

    Returns (resolved_quotes, unresolved_names).
    """
    name_lookup: dict[str, tuple[int, str, str, str, object]] = {}
    for idx, m in enumerate(card_matchups):
        red, blue, event_date = m[0], m[1], m[4]
        name_lookup[_normalize(red)] = (idx, "red", red, blue, event_date)
        name_lookup[_normalize(blue)] = (idx, "blue", red, blue, event_date)
    all_norms = list(name_lookup.keys())

    named_quotes = [q for q in quotes if not is_fight_level_kind(q.market_kind)]
    fight_level_quotes = [q for q in quotes if is_fight_level_kind(q.market_kind)]

    resolved: list[ResolvedMarketQuote] = []
    unresolved: list[str] = []
    seen_unresolved: set[str] = set()

    # ── Pass 1: named quotes resolve by fighter name ────────────────────────
    event_code_map: dict[str, tuple[int, str, str]] = {}
    for q in named_quotes:
        norm = _normalize(q.fighter_name)

        if norm in name_lookup:
            hit = name_lookup[norm]
        else:
            tok = [cn for cn in all_norms if _token_surname_match(norm, cn)]
            hit = name_lookup[tok[0]] if len(tok) == 1 else None

        if hit is None:
            if q.fighter_name not in seen_unresolved:
                seen_unresolved.add(q.fighter_name)
                unresolved.append(q.fighter_name)
            continue

        fight_idx, corner, card_red, card_blue, event_date = hit

        quote_date = _parse_event_date(q.event_ticker)
        if quote_date is not None and event_date is not None:
            delta_days = abs((quote_date - event_date).days)
            if delta_days > 1:
                if q.fighter_name not in seen_unresolved:
                    seen_unresolved.add(q.fighter_name)
                    unresolved.append(q.fighter_name)
                continue

        resolved.append(ResolvedMarketQuote(
            platform=q.platform, event_ticker=q.event_ticker, market_ticker=q.market_ticker,
            market_kind=q.market_kind, fighter_name=q.fighter_name, yes_bid=q.yes_bid,
            yes_ask=q.yes_ask, last_price=q.last_price, volume=q.volume,
            open_interest=q.open_interest, depth_usd_3c=q.depth_usd_3c, fetched_at=q.fetched_at,
            card_red=card_red, card_blue=card_blue, fight_idx=fight_idx, corner=corner,
        ))
        event_code_map.setdefault(_event_code(q.event_ticker), (fight_idx, card_red, card_blue))

    # ── Pass 2: fight-level quotes join via the shared event-ticker code ───
    for q in fight_level_quotes:
        hit = event_code_map.get(_event_code(q.event_ticker))
        if hit is None:
            unresolved.append(f"[fight-level] {q.market_ticker}")
            continue

        fight_idx, card_red, card_blue = hit
        event_date = card_matchups[fight_idx][4]
        quote_date = _parse_event_date(q.event_ticker)
        if quote_date is not None and event_date is not None:
            if abs((quote_date - event_date).days) > 1:
                unresolved.append(f"[fight-level] {q.market_ticker}")
                continue

        resolved.append(ResolvedMarketQuote(
            platform=q.platform, event_ticker=q.event_ticker, market_ticker=q.market_ticker,
            market_kind=q.market_kind, fighter_name="", yes_bid=q.yes_bid,
            yes_ask=q.yes_ask, last_price=q.last_price, volume=q.volume,
            open_interest=q.open_interest, depth_usd_3c=q.depth_usd_3c, fetched_at=q.fetched_at,
            card_red=card_red, card_blue=card_blue, fight_idx=fight_idx, corner="fight",
        ))

    return resolved, unresolved
