"""Tests for scrape_upcoming._prune_stale()."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.ingest.scrape_upcoming import _prune_stale

_URL = "http://ufcstats.com/event-details/681d07e328798ec0"


def _card(tmp_path, name, ev_date, url=_URL):
    p = tmp_path / name
    p.write_text(json.dumps({"event_date": ev_date.isoformat(), "event_url": url}),
                 encoding="utf-8")
    return p


def test_prune_stale_drops_renamed_card_but_keeps_unrelated_same_night(tmp_path):
    # ufcstats renamed the event mid-week (headliner withdrew), so the scrape
    # wrote a new slug and the old file survived on event_date alone — both then
    # show in the dropdown and both get iterated by the prop-line loggers.
    today = date.today()
    superseded = _card(tmp_path, "fn_ankalaev_vs_rountree_jr.json", today)
    fresh = _card(tmp_path, "fn_ankalaev_vs_guskov.json", today)
    other = _card(tmp_path, "fn_other_event.json", today, url="http://x/other")

    _prune_stale(tmp_path, {fresh.name}, {_URL})

    assert not superseded.exists()   # replacement is on disk, safe to drop
    assert fresh.exists()
    assert other.exists()            # different event entirely — untouched


def test_prune_stale_keeps_card_when_rescrape_produced_no_replacement(tmp_path):
    # fetch_event_bouts() returned nothing for this event, so no file was
    # written and its url is NOT in written_urls. Deleting here would wipe the
    # live card mid-event — the exact failure mode the grace window exists for.
    today = date.today()
    existing = _card(tmp_path, "fn_tonight.json", today)

    _prune_stale(tmp_path, set(), set())

    assert existing.exists()


def test_prune_stale_still_deletes_past_cards(tmp_path):
    old = _card(tmp_path, "fn_last_month.json", date.today() - timedelta(days=30))

    _prune_stale(tmp_path, set(), set())

    assert not old.exists()
