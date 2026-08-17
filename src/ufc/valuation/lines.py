"""Line source abstractions for DFS prop ingestion."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Line:
    market: str        # 'sig_strikes', 'takedowns', 'duration_sec', 'method', 'winner'
    side: str          # 'over', 'under', or category string
    line_value: float  # e.g. 52.5
    payout_type: str   # e.g. 'powerplay_power_2pick'
    payout_multiplier: float  # e.g. 3.0 (per-pick)
    fighter_id: Optional[str] = None  # None for fight-level props
    fighter_name: Optional[str] = None
    source: str = "manual"


class LineSource(ABC):
    @abstractmethod
    def fetch(self) -> list[Line]:
        ...


class CLILineSource(LineSource):
    """Parse --prop flags from argparse strings.

    Format: 'market:fighter:side:line' or 'market:side:line'
    Examples:
      sig_strikes:red:over:52.5
      takedowns:red:under:0.5
      duration:over:13.5min
      winner:red
    """

    def __init__(self, prop_strings: list[str], payout_type: str = "powerplay_power_3pick",
                 red_id: str | None = None, blue_id: str | None = None,
                 red_name: str = "Red", blue_name: str = "Blue"):
        self.prop_strings = prop_strings
        self.payout_type = payout_type
        self.red_id = red_id
        self.blue_id = blue_id
        self.red_name = red_name
        self.blue_name = blue_name

    def fetch(self) -> list[Line]:
        from ufc.valuation.payouts import get_payout_multiplier
        mult = get_payout_multiplier(self.payout_type)
        lines = []
        for s in self.prop_strings:
            parts = s.split(":")
            if len(parts) < 2:
                continue
            market = parts[0].lower()
            # Parse duration with 'min' suffix
            if len(parts) == 3:
                # market:side:line (fight-level)
                side = parts[1].lower()
                line_val = _parse_line_value(parts[2])
                lines.append(Line(market, side, line_val, self.payout_type, mult, source="cli"))
            elif len(parts) == 4:
                # market:fighter:side:line
                fighter_str = parts[1].lower()
                side = parts[2].lower()
                line_val = _parse_line_value(parts[3])
                fid = self.red_id if fighter_str == "red" else self.blue_id
                fname = self.red_name if fighter_str == "red" else self.blue_name
                lines.append(Line(market, side, line_val, self.payout_type, mult,
                                  fighter_id=fid, fighter_name=fname, source="cli"))
            elif len(parts) == 2:
                # market:fighter (winner)
                fighter_str = parts[1].lower()
                fid = self.red_id if fighter_str == "red" else self.blue_id
                fname = self.red_name if fighter_str == "red" else self.blue_name
                lines.append(Line(market, "win", 0.0, self.payout_type, mult,
                                  fighter_id=fid, fighter_name=fname, source="cli"))
        return lines


class JSONLineSource(LineSource):
    """Load lines from a JSON file."""

    def __init__(self, path: Path, payout_type: str = "powerplay_power_3pick"):
        self.path = Path(path)
        self.payout_type = payout_type

    def fetch(self) -> list[Line]:
        from ufc.valuation.payouts import get_payout_multiplier
        with open(self.path) as f:
            data = json.load(f)
        mult = get_payout_multiplier(self.payout_type)
        lines = []
        for item in data.get("lines", []):
            lines.append(Line(
                market=item["market"],
                side=item["side"],
                line_value=float(item.get("line", 0)),
                payout_type=item.get("payout", self.payout_type),
                payout_multiplier=item.get("multiplier", mult),
                fighter_id=item.get("fighter_id"),
                fighter_name=item.get("fighter_name"),
                source="json",
            ))
        return lines


class MockLineSource(LineSource):
    """Generate lines around model medians for testing."""

    def __init__(self, fighter_id: str, median_sig_strikes: float = 50.0,
                 median_td: float = 1.5, median_duration_rounds: float = 2.5,
                 payout_type: str = "powerplay_power_2pick"):
        self.fighter_id = fighter_id
        self.median_sig_strikes = median_sig_strikes
        self.median_td = median_td
        self.median_duration_rounds = median_duration_rounds
        self.payout_type = payout_type

    def fetch(self) -> list[Line]:
        from ufc.valuation.payouts import get_payout_multiplier
        mult = get_payout_multiplier(self.payout_type)
        return [
            Line("sig_strikes", "over", self.median_sig_strikes + 5.5,
                 self.payout_type, mult, fighter_id=self.fighter_id, source="mock"),
            Line("sig_strikes", "under", self.median_sig_strikes + 5.5,
                 self.payout_type, mult, fighter_id=self.fighter_id, source="mock"),
            Line("takedowns", "over", self.median_td + 0.5,
                 self.payout_type, mult, fighter_id=self.fighter_id, source="mock"),
            Line("duration_sec", "over", self.median_duration_rounds * 300,
                 self.payout_type, mult, source="mock"),
        ]


class HTTPLineSource(LineSource):
    """Stub for future API integrations."""

    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url
        self.api_key = api_key

    def fetch(self) -> list[Line]:
        raise NotImplementedError(
            "HTTPLineSource is a stub. Implement when API credentials are available."
        )


def _parse_line_value(s: str) -> float:
    """Parse '52.5', '13.5min', '2.5rounds' etc."""
    s = s.strip().lower()
    if s.endswith("min"):
        return float(s[:-3]) * 60  # convert to seconds
    if s.endswith("rounds") or s.endswith("round"):
        return float(s.split("round")[0]) * 300  # convert to seconds
    try:
        return float(s)
    except ValueError:
        return 0.0
