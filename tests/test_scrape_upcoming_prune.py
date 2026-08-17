"""_prune_stale must keep a same-night card through its grace window.

Regression for the UFC 329 (2026-07-11) incident: ufcstats.com drops an event
from its "upcoming" list once it goes live, so an unconditional prune on
refresh deleted the fight-night card mid-event. Grace window mirrors
_card_still_listed in src/ufc/api/app.py (event_date + 2 days).
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.ingest.scrape_upcoming import _prune_stale


def _write_card(out_dir: Path, name: str, event_date: str | None) -> Path:
    data = {"event_name": name, "matchups": []}
    if event_date is not None:
        data["event_date"] = event_date
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_prune_keeps_todays_card_and_removes_stale_one(tmp_path):
    today_path = _write_card(tmp_path, "today_card", date.today().isoformat())
    stale_path = _write_card(
        tmp_path, "stale_card", (date.today() - timedelta(days=10)).isoformat()
    )

    _prune_stale(tmp_path, written_names=set())

    assert today_path.exists(), "same-night card was deleted before its grace window elapsed"
    assert not stale_path.exists(), "stale card should have been pruned"


def test_prune_fails_open_on_missing_event_date(tmp_path):
    bad_path = _write_card(tmp_path, "no_date_card", None)

    _prune_stale(tmp_path, written_names=set())

    assert bad_path.exists(), "unparseable event_date must be kept, not deleted"
