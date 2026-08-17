"""Parse upcoming fight card JSON into matchup specs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class PropSpec:
    market: str
    fighter: str   # 'red', 'blue', or 'fight'
    side: str      # 'over', 'under'
    line: float
    payout: str = "powerplay_power_3pick"


@dataclass
class MatchupSpec:
    red: str
    blue: str
    scheduled_rounds: int = 3
    is_title: bool = False
    weight_class: str | None = None
    referee: str = ""
    props: list[PropSpec] = field(default_factory=list)


@dataclass
class CardSpec:
    event_name: str
    event_date: date
    matchups: list[MatchupSpec]
    default_payout: str = "powerplay_power_3pick"
    location: str = ""


def parse_card(path: Path | str) -> CardSpec:
    """Parse a card JSON file into a CardSpec."""
    with open(path) as f:
        data = json.load(f)

    event_date = date.fromisoformat(data["event_date"])
    matchups = []

    for m in data.get("matchups", []):
        props = []
        for p in m.get("props", []):
            line_val = p.get("line", 0.0)
            if isinstance(line_val, str):
                from ufc.valuation.lines import _parse_line_value
                line_val = _parse_line_value(line_val)
            props.append(PropSpec(
                market=p["market"],
                fighter=p.get("fighter", "fight"),
                side=p["side"],
                line=float(line_val),
                payout=p.get("payout", data.get("default_payout", "powerplay_power_3pick")),
            ))
        matchups.append(MatchupSpec(
            red=m["red"],
            blue=m["blue"],
            scheduled_rounds=int(m.get("scheduled_rounds", 3)),
            is_title=bool(m.get("is_title", False)),
            weight_class=m.get("weight_class") or None,
            referee=m.get("referee", ""),
            props=props,
        ))

    return CardSpec(
        event_name=data.get("event_name", "Unknown Event"),
        event_date=event_date,
        matchups=matchups,
        default_payout=data.get("default_payout", "powerplay_power_3pick"),
        location=data.get("location", ""),
    )
