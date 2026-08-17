"""Scrape upcoming UFC event cards from ufcstats.com.

ufcstats.com gates every page behind a JS SHA-256 proof-of-work challenge.
This module solves it in pure Python (hashlib + requests.Session) and emits
standard card JSONs compatible with parse_card() (src/ufc/inference/card.py).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

BASE = "http://ufcstats.com"
_UPCOMING_URL = f"{BASE}/statistics/events/upcoming?page=all"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

_DIVISIONS = {
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight",
}

# Only these fields are permitted in overrides.json entries
_OVERRIDE_FIELDS = frozenset({
    "scheduled_rounds", "is_title", "weight_class", "referee", "location",
    "card_segment",
})


# ---------------------------------------------------------------------------
# PoW-solving HTTP fetch
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA})
    return s


def _solve_get(s: requests.Session, url: str, timeout: int = 30, _retries: int = 2) -> str:
    r = s.get(url, timeout=timeout)
    if "Checking your browser" not in r.text:
        return r.text
    if _retries <= 0:
        raise RuntimeError(
            f"ufcstats.com PoW challenge not cleared after retries for {url}. "
            "The challenge format may have changed."
        )
    m_nonce = re.search(r'nonce="([0-9a-f]+)"', r.text)
    m_diff  = re.search(r"target=new Array\((\d+)\+1\)", r.text)
    if not m_nonce or not m_diff:
        raise RuntimeError(
            "PoW challenge detected but could not parse nonce/difficulty. "
            "ufcstats.com markup may have changed."
        )
    nonce = m_nonce.group(1)
    diff  = int(m_diff.group(1))
    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith("0" * diff):
        n += 1
    s.post(
        f"{BASE}/__c",
        data={"nonce": nonce, "n": n},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    return _solve_get(s, url, timeout, _retries - 1)


# ---------------------------------------------------------------------------
# Parse upcoming events list
# ---------------------------------------------------------------------------

def fetch_upcoming_events(s: requests.Session) -> list[dict[str, Any]]:
    html = _solve_get(s, _UPCOMING_URL)
    soup = BeautifulSoup(html, "html.parser")

    name_tags = soup.find_all("a", class_="b-link b-link_style_black")
    date_tags = soup.find_all("span", class_="b-statistics__date")
    loc_tags  = soup.find_all(
        "td",
        class_="b-statistics__table-col b-statistics__table-col_style_big-top-padding",
    )

    events: list[dict[str, Any]] = []
    for i, tag in enumerate(name_tags):
        raw_date = date_tags[i].text.strip() if i < len(date_tags) else ""
        raw_loc  = loc_tags[i].text.strip()  if i < len(loc_tags)  else ""
        try:
            event_date = datetime.strptime(raw_date, "%B %d, %Y").date()
        except ValueError:
            continue
        events.append({
            "event_name": tag.text.strip(),
            "date": event_date,
            "location": raw_loc,
            "url": tag["href"],
        })
    return events


# ---------------------------------------------------------------------------
# Parse bouts for one event
# ---------------------------------------------------------------------------

def _normalize_weight_class(raw: str) -> str | None:
    cleaned = re.sub(r"\b(UFC|Interim|Title|Bout|Championship)\b", "", raw, flags=re.I).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned in _DIVISIONS:
        return cleaned
    if re.search(r"catch", cleaned, re.I):
        return "Catch Weight"
    return None


def _fetch_fight_title_label(s: requests.Session, fight_url: str) -> str:
    """Fetch the fight detail page and return the fight-title label (e.g. 'UFC Interim Heavyweight Title Bout')."""
    try:
        html = _solve_get(s, fight_url)
        soup = BeautifulSoup(html, "html.parser")
        el = soup.find("i", class_="b-fight-details__fight-title")
        return el.get_text(" ", strip=True) if el else ""
    except Exception:
        return ""


def fetch_event_bouts(s: requests.Session, url: str) -> list[dict[str, Any]]:
    html = _solve_get(s, url)
    soup = BeautifulSoup(html, "html.parser")

    rows = soup.find_all("tr", class_=re.compile(r"b-fight-details__table-row"))
    bouts: list[dict[str, Any]] = []
    for tr in rows:
        corners = tr.find_all("a", class_="b-link b-link_style_black")
        if len(corners) < 2:
            continue
        red  = corners[0].text.strip()
        blue = corners[1].text.strip()
        if not red or not blue:
            continue

        # Fight detail URL lives in data-link attribute on the <tr>
        fight_url = tr.get("data-link", "")

        # Fetch fight detail page for authoritative title label
        fight_title_label = _fetch_fight_title_label(s, fight_url) if fight_url else ""

        # Fall back to weight-class td if fight detail fetch failed
        tds = tr.find_all("td")
        wc_raw = fight_title_label
        if not wc_raw:
            for td in tds:
                txt = td.get_text(" ", strip=True)
                if re.search(r"weight|catch|bout", txt, re.I):
                    wc_raw = txt
                    break

        is_title   = "Title" in wc_raw
        weight_class = _normalize_weight_class(wc_raw)
        # first bout is the main event; title bouts → 5 rounds
        is_main    = len(bouts) == 0
        sched_rounds = 5 if (is_title or is_main) else 3

        bouts.append({
            "red":              red,
            "blue":             blue,
            "scheduled_rounds": sched_rounds,
            "is_title":         is_title,
            "weight_class":     weight_class,
            "referee":          "",
            "props":            [],
        })
    return bouts


# ---------------------------------------------------------------------------
# Build card dict (standard parse_card schema + provenance metadata)
# ---------------------------------------------------------------------------

def build_card_dict(event: dict[str, Any], bouts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_name":      event["event_name"],
        "event_date":      event["date"].isoformat(),
        "location":        event["location"],
        "default_payout":  "powerplay_power_2pick",
        "source":          "ufcstats_scrape",
        "event_url":       event["url"],
        "scraped_at":      datetime.now(timezone.utc).isoformat(),
        "matchups":        bouts,
    }


# ---------------------------------------------------------------------------
# Slugify helper
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# ---------------------------------------------------------------------------
# Prune stale card files
# ---------------------------------------------------------------------------

def _prune_stale(out_dir: Path, written_names: set[str],
                 written_urls: set[str] | None = None) -> None:
    """Delete old card JSONs not in the latest scrape, but keep same-night
    cards through their grace window (must mirror _card_still_listed in
    src/ufc/api/app.py — ufcstats drops an event from "upcoming" once it goes
    live, so an unconditional prune deletes the fight-night card mid-event).

    Exception to the grace window: a card superseded by a rename. Filenames are
    slugged from the event name, so when ufcstats renames an event (a headliner
    withdraws: "Ankalaev vs. Rountree Jr" → "Ankalaev vs. Guskov") the scrape
    writes a NEW file and the old one survives on date alone. Both then show in
    the dropdown AND both get iterated by 07_fetch_props / 07b_log_prop_lines /
    07c_capture_closing_lines, logging lines for bouts that will never happen.
    Matching event_url identifies the supersession, and we only delete once the
    replacement is on disk — so the fight-night card is never lost.
    """
    written_urls = written_urls or set()
    for old in out_dir.glob("*.json"):
        if old.name in written_names:
            continue
        try:
            data = json.loads(old.read_text(encoding="utf-8"))
            ev_date = date.fromisoformat(data["event_date"])
        except Exception:
            continue  # unreadable/unparseable → fail-open, keep the file
        if data.get("event_url") in written_urls:
            old.unlink()
            continue
        if date.today() > ev_date + timedelta(days=2):
            old.unlink()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def scrape_upcoming_cards(
    limit: int = 5,
    out_dir: Path | None = None,
    prune: bool = True,
) -> list[Path]:
    """Scrape the next `limit` upcoming UFC events and write card JSONs.

    Returns list of written paths. Safe to call repeatedly (idempotent names).
    """
    from ufc.io import paths as _paths  # deferred to avoid import cycle at module load

    if out_dir is None:
        out_dir = _paths.upcoming_cards()
    out_dir.mkdir(parents=True, exist_ok=True)

    s = _make_session()
    all_events = fetch_upcoming_events(s)

    today = date.today()
    upcoming = sorted(
        (e for e in all_events if e["date"] >= today),
        key=lambda e: e["date"],
    )[:limit]

    if not upcoming:
        return []

    # Load overrides file (cards/overrides.json) — applied after scrape
    overrides_path = out_dir.parent / "overrides.json"
    all_overrides: dict[str, list[dict]] = {}
    if overrides_path.exists():
        try:
            raw_ov = json.loads(overrides_path.read_text(encoding="utf-8"))
            all_overrides = {k: v for k, v in raw_ov.items() if not k.startswith("_")}
        except Exception:
            pass

    written: list[Path] = []
    written_urls: set[str] = set()   # only events that actually produced a file
    for event in upcoming:
        bouts = fetch_event_bouts(s, event["url"])
        if not bouts:
            continue
        card = build_card_dict(event, bouts)
        slug = _slugify(event["event_name"])
        fname = f"{slug}_{event['date']:%Y_%m_%d}.json"

        # Apply overrides keyed by card slug
        card_overrides = all_overrides.get(fname.replace(".json", ""), [])
        for override in card_overrides:
            r, b = override.get("red"), override.get("blue")
            for bout in card["matchups"]:
                if bout["red"] == r and bout["blue"] == b:
                    for field, val in override.items():
                        if field in _OVERRIDE_FIELDS:
                            bout[field] = val

        fpath = out_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        written.append(fpath)
        written_urls.add(event["url"])

    if prune and written:
        written_names = {p.name for p in written}
        _prune_stale(out_dir, written_names, written_urls)

    return written
