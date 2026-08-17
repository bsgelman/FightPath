"""Tests for market_lines (Kalshi ingest) parse/map/resolve — no network calls."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from ufc.ingest.market_lines import (
    MarketQuote,
    ResolvedMarketQuote,
    SeriesSpec,
    SERIES_TABLE,
    _parse_event_date,
    _dollars,
    _map_method,
    _parse_winner_markets,
    _parse_method_markets,
    _parse_distance_markets,
    _parse_mof_markets,
    _parse_rounds_markets,
    _parse_vicround_markets,
    _orderbook_depth_usd,
    is_fight_level_kind,
    resolve_markets_to_card,
    fetch_all_markets,
)


# ---------------------------------------------------------------------------
# _parse_event_date
# ---------------------------------------------------------------------------

class TestParseEventDate:
    def test_july_date(self):
        assert _parse_event_date("KXUFCFIGHT-26JUL11SAIPIM") == date(2026, 7, 11)

    def test_june_date(self):
        assert _parse_event_date("KXUFCMOV-26JUN27ABDNAS") == date(2026, 6, 27)

    def test_malformed_returns_none(self):
        assert _parse_event_date("KXUFCFIGHT-GARBAGE") is None

    def test_empty_returns_none(self):
        assert _parse_event_date("") is None


# ---------------------------------------------------------------------------
# _dollars
# ---------------------------------------------------------------------------

class TestDollars:
    def test_parses_string(self):
        assert _dollars("0.5600") == 0.56

    def test_none_returns_none(self):
        assert _dollars(None) is None

    def test_empty_string_returns_none(self):
        assert _dollars("") is None

    def test_garbage_returns_none(self):
        assert _dollars("not-a-number") is None


# ---------------------------------------------------------------------------
# _map_method
# ---------------------------------------------------------------------------

class TestMapMethod:
    def test_submission(self):
        assert _map_method("Submission") == "method_sub"

    def test_ko_tko_dq(self):
        assert _map_method("KO/TKO/DQ") == "method_ko"

    def test_decision(self):
        assert _map_method("Decision") == "method_dec"

    def test_unknown_returns_none(self):
        assert _map_method("Draw") is None


# ---------------------------------------------------------------------------
# _parse_winner_markets (pure parse of a KXUFCFIGHT /markets payload)
# ---------------------------------------------------------------------------

class TestParseWinnerMarkets:
    def _payload(self):
        return {
            "cursor": "",
            "markets": [
                {
                    "ticker": "KXUFCFIGHT-26JUL11SAIPIM-SAI",
                    "event_ticker": "KXUFCFIGHT-26JUL11SAIPIM",
                    "yes_sub_title": "Benoit Saint-Denis",
                    "title": "Will Benoit Saint-Denis win the Saint-Denis vs Pimblett professional MMA fight scheduled for Jul 11, 2026?",
                    "yes_bid_dollars": "0.5500",
                    "yes_ask_dollars": "0.5600",
                    "last_price_dollars": "0.5600",
                    "volume_fp": "21041.94",
                    "open_interest_fp": "20792.92",
                    "status": "active",
                },
                {
                    "ticker": "KXUFCFIGHT-26JUL11SAIPIM-PIM",
                    "event_ticker": "KXUFCFIGHT-26JUL11SAIPIM",
                    "yes_sub_title": "Paddy Pimblett",
                    "title": "Will Paddy Pimblett win the Saint-Denis vs Pimblett professional MMA fight scheduled for Jul 11, 2026?",
                    "yes_bid_dollars": "0.4400",
                    "yes_ask_dollars": "0.4500",
                    "last_price_dollars": "0.4500",
                    "volume_fp": "88587.32",
                    "open_interest_fp": "82771.86",
                    "status": "active",
                },
            ],
        }

    def test_two_quotes_parsed(self):
        quotes = _parse_winner_markets(self._payload())
        assert len(quotes) == 2

    def test_fighter_name_from_yes_sub_title(self):
        quotes = _parse_winner_markets(self._payload())
        names = {q.fighter_name for q in quotes}
        assert names == {"Benoit Saint-Denis", "Paddy Pimblett"}

    def test_market_kind_winner(self):
        quotes = _parse_winner_markets(self._payload())
        assert all(q.market_kind == "winner" for q in quotes)

    def test_prices_parsed(self):
        quotes = _parse_winner_markets(self._payload())
        sai = [q for q in quotes if q.fighter_name == "Benoit Saint-Denis"][0]
        assert sai.yes_bid == 0.55
        assert sai.yes_ask == 0.56
        assert sai.last_price == 0.56
        assert sai.volume == 21041.94
        assert sai.open_interest == 20792.92

    def test_platform_and_tickers(self):
        quotes = _parse_winner_markets(self._payload())
        sai = [q for q in quotes if q.fighter_name == "Benoit Saint-Denis"][0]
        assert sai.platform == "kalshi"
        assert sai.market_ticker == "KXUFCFIGHT-26JUL11SAIPIM-SAI"
        assert sai.event_ticker == "KXUFCFIGHT-26JUL11SAIPIM"

    def test_missing_yes_sub_title_falls_back_to_title_regex(self):
        payload = {
            "markets": [{
                "ticker": "KXUFCFIGHT-26JUL11XXXYYY-XXX",
                "event_ticker": "KXUFCFIGHT-26JUL11XXXYYY",
                "title": "Will Jane Doe win the Doe vs Roe professional MMA fight scheduled for Jul 11, 2026?",
                "yes_bid_dollars": "0.5000",
                "yes_ask_dollars": "0.5100",
                "last_price_dollars": "0.5000",
                "volume_fp": "0",
                "open_interest_fp": "0",
            }]
        }
        quotes = _parse_winner_markets(payload)
        assert len(quotes) == 1
        assert quotes[0].fighter_name == "Jane Doe"

    def test_empty_markets_returns_empty(self):
        assert _parse_winner_markets({"markets": []}) == []


# ---------------------------------------------------------------------------
# _parse_method_markets (pure parse of a KXUFCMOV /markets payload)
# ---------------------------------------------------------------------------

class TestParseMethodMarkets:
    def _payload(self):
        return {
            "markets": [
                {
                    "ticker": "KXUFCMOV-26JUN27ABDNAS-NASSUB",
                    "event_ticker": "KXUFCMOV-26JUN27ABDNAS",
                    "custom_strike": {"Method": "Submission", "Participant": "Jefferson Nascimento"},
                    "yes_sub_title": "Jefferson Nascimento by Submission",
                    "yes_bid_dollars": "0.0000",
                    "yes_ask_dollars": "0.0200",
                    "last_price_dollars": "0.0100",
                    "volume_fp": "2233.50",
                    "open_interest_fp": "2207.50",
                },
                {
                    "ticker": "KXUFCMOV-26JUN27ABDNAS-NASKOTKODQ",
                    "event_ticker": "KXUFCMOV-26JUN27ABDNAS",
                    "custom_strike": {"Method": "KO/TKO/DQ", "Participant": "Jefferson Nascimento"},
                    "yes_sub_title": "Jefferson Nascimento by KO/TKO/DQ",
                    "yes_bid_dollars": "0.1000",
                    "yes_ask_dollars": "0.1200",
                    "last_price_dollars": "0.1100",
                    "volume_fp": "500.0",
                    "open_interest_fp": "400.0",
                },
                {
                    "ticker": "KXUFCMOV-26JUN27ABDNAS-NASDEC",
                    "event_ticker": "KXUFCMOV-26JUN27ABDNAS",
                    "custom_strike": {"Method": "Decision", "Participant": "Jefferson Nascimento"},
                    "yes_sub_title": "Jefferson Nascimento by Decision",
                    "yes_bid_dollars": "0.6000",
                    "yes_ask_dollars": "0.6200",
                    "last_price_dollars": "0.6100",
                    "volume_fp": "900.0",
                    "open_interest_fp": "800.0",
                },
            ]
        }

    def test_three_quotes_parsed(self):
        quotes = _parse_method_markets(self._payload())
        assert len(quotes) == 3

    def test_market_kinds_mapped(self):
        quotes = _parse_method_markets(self._payload())
        kinds = {q.market_kind for q in quotes}
        assert kinds == {"method_sub", "method_ko", "method_dec"}

    def test_fighter_name_from_custom_strike_participant(self):
        quotes = _parse_method_markets(self._payload())
        assert all(q.fighter_name == "Jefferson Nascimento" for q in quotes)

    def test_unknown_method_skipped(self):
        payload = {"markets": [{
            "ticker": "X-DRAW", "event_ticker": "X",
            "custom_strike": {"Method": "Draw", "Participant": "Someone"},
            "yes_sub_title": "Someone by Draw",
            "yes_bid_dollars": "0.01", "yes_ask_dollars": "0.02",
            "last_price_dollars": "0.01", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        assert _parse_method_markets(payload) == []

    def test_missing_custom_strike_skipped(self):
        payload = {"markets": [{
            "ticker": "X-Y", "event_ticker": "X",
            "yes_sub_title": "Someone by Submission",
            "yes_bid_dollars": "0.01", "yes_ask_dollars": "0.02",
            "last_price_dollars": "0.01", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        assert _parse_method_markets(payload) == []


# ---------------------------------------------------------------------------
# _orderbook_depth_usd
# ---------------------------------------------------------------------------

class TestOrderbookDepthUsd:
    def test_sums_levels_within_threshold(self):
        # no-bid at 0.44 => implied yes-ask 0.56 (== best_ask, within 0c)
        # no-bid at 0.43 => implied yes-ask 0.57 (1c away, within 3c)
        # no-bid at 0.40 => implied yes-ask 0.60 (4c away, excluded)
        orderbook = {"orderbook_fp": {"no_dollars": [
            ["0.4400", "100.00"],
            ["0.4300", "50.00"],
            ["0.4000", "9999.00"],
        ]}}
        depth = _orderbook_depth_usd(orderbook, best_ask=0.56, cents=3)
        # 100 * 0.56 + 50 * 0.57
        assert depth == pytest.approx(100.0 * 0.56 + 50.0 * 0.57)

    def test_empty_book_returns_zero(self):
        assert _orderbook_depth_usd({"orderbook_fp": {}}, best_ask=0.5) == 0.0

    def test_missing_orderbook_key_returns_zero(self):
        assert _orderbook_depth_usd({}, best_ask=0.5) == 0.0


# ---------------------------------------------------------------------------
# resolve_markets_to_card
# ---------------------------------------------------------------------------

class TestResolveMarketsToCard:
    def _card_matchups(self):
        return [
            ("Benoit Saint-Denis", "Paddy Pimblett", 3, False, date(2026, 7, 11),
             "Lightweight", "", "London, UK"),
        ]

    def _quote(self, fighter_name, event_ticker="KXUFCFIGHT-26JUL11SAIPIM", market_kind="winner"):
        return MarketQuote(
            platform="kalshi", event_ticker=event_ticker,
            market_ticker=event_ticker + "-X", market_kind=market_kind,
            fighter_name=fighter_name, yes_bid=0.5, yes_ask=0.51,
            last_price=0.5, volume=100.0, open_interest=100.0,
            depth_usd_3c=None, fetched_at="2026-07-02T00:00:00Z",
        )

    def test_exact_match_red_corner(self):
        resolved, unresolved = resolve_markets_to_card(
            [self._quote("Benoit Saint-Denis")], self._card_matchups())
        assert len(resolved) == 1
        assert resolved[0].corner == "red"
        assert resolved[0].fight_idx == 0
        assert unresolved == []

    def test_blue_corner(self):
        resolved, _ = resolve_markets_to_card(
            [self._quote("Paddy Pimblett")], self._card_matchups())
        assert resolved[0].corner == "blue"

    def test_unresolved_name(self):
        resolved, unresolved = resolve_markets_to_card(
            [self._quote("Nobody Here")], self._card_matchups())
        assert resolved == []
        assert "Nobody Here" in unresolved

    def test_event_date_mismatch_excluded(self):
        # Fighter name matches, but event ticker date is >1 day off the card date —
        # must not cross-match (guards against a fighter name appearing on two cards).
        resolved, unresolved = resolve_markets_to_card(
            [self._quote("Benoit Saint-Denis", event_ticker="KXUFCFIGHT-26SEP05SAIPIM")],
            self._card_matchups())
        assert resolved == []
        assert "Benoit Saint-Denis" in unresolved

    def test_event_date_within_one_day_included(self):
        resolved, _ = resolve_markets_to_card(
            [self._quote("Benoit Saint-Denis", event_ticker="KXUFCFIGHT-26JUL12SAIPIM")],
            self._card_matchups())
        assert len(resolved) == 1

    def test_empty_card(self):
        resolved, unresolved = resolve_markets_to_card(
            [self._quote("Benoit Saint-Denis")], [])
        assert resolved == []
        assert unresolved == ["Benoit Saint-Denis"]

    def test_empty_quotes(self):
        resolved, unresolved = resolve_markets_to_card([], self._card_matchups())
        assert resolved == []
        assert unresolved == []

    def test_hyphenated_surname_vs_unhyphenated_card_name(self):
        # Real-world mismatch: Kalshi's title gives "Saint-Denis" (hyphen) but the
        # card JSON has "Saint Denis" (space) — must still resolve to one fighter.
        card_matchups = [
            ("Benoit Saint Denis", "Paddy Pimblett", 3, False, date(2026, 7, 11),
             "Lightweight", "", "London, UK"),
        ]
        resolved, unresolved = resolve_markets_to_card(
            [self._quote("Benoit Saint-Denis")], card_matchups)
        assert len(resolved) == 1
        assert resolved[0].corner == "red"
        assert unresolved == []


# ---------------------------------------------------------------------------
# _parse_distance_markets (KXUFCDISTANCE — fight-level, no fighter identity)
# ---------------------------------------------------------------------------

class TestParseDistanceMarkets:
    def _payload(self):
        return {"markets": [{
            "ticker": "KXUFCDISTANCE-26JUN27ABDNAS-DIST",
            "event_ticker": "KXUFCDISTANCE-26JUN27ABDNAS",
            "yes_sub_title": "Fight goes the distance",
            "yes_bid_dollars": "0.3000", "yes_ask_dollars": "0.3200",
            "last_price_dollars": "0.3100", "volume_fp": "100.0", "open_interest_fp": "90.0",
        }]}

    def test_one_quote_parsed_fight_level(self):
        quotes = _parse_distance_markets(self._payload())
        assert len(quotes) == 1
        assert quotes[0].market_kind == "distance"
        assert quotes[0].fighter_name == ""

    def test_prices_parsed(self):
        q = _parse_distance_markets(self._payload())[0]
        assert q.yes_bid == 0.30 and q.yes_ask == 0.32

    def test_non_dist_suffix_skipped(self):
        payload = {"markets": [{
            "ticker": "KXUFCDISTANCE-26JUN27ABDNAS-OTHERTHING",
            "event_ticker": "KXUFCDISTANCE-26JUN27ABDNAS",
            "yes_bid_dollars": "0.1", "yes_ask_dollars": "0.2",
            "last_price_dollars": "0.1", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        assert _parse_distance_markets(payload) == []

    def test_empty_markets_returns_empty(self):
        assert _parse_distance_markets({"markets": []}) == []


# ---------------------------------------------------------------------------
# _parse_mof_markets (KXUFCMOF — fight-level method, custom_strike.Method only)
# ---------------------------------------------------------------------------

class TestParseMofMarkets:
    def _payload(self):
        return {"markets": [
            {
                "ticker": "KXUFCMOF-26JUN27ABDNAS-SUB",
                "event_ticker": "KXUFCMOF-26JUN27ABDNAS",
                "custom_strike": {"Method": "Submission"},
                "yes_bid_dollars": "0.01", "yes_ask_dollars": "0.02",
                "last_price_dollars": "0.01", "volume_fp": "300.0", "open_interest_fp": "280.0",
            },
            {
                "ticker": "KXUFCMOF-26JUN27ABDNAS-KOTKODQ",
                "event_ticker": "KXUFCMOF-26JUN27ABDNAS",
                "custom_strike": {"Method": "KO/TKO/DQ"},
                "yes_bid_dollars": "0.60", "yes_ask_dollars": "0.62",
                "last_price_dollars": "0.61", "volume_fp": "500.0", "open_interest_fp": "400.0",
            },
            {
                "ticker": "KXUFCMOF-26JUN27ABDNAS-DEC",
                "event_ticker": "KXUFCMOF-26JUN27ABDNAS",
                "custom_strike": {"Method": "Decision"},
                "yes_bid_dollars": "0.30", "yes_ask_dollars": "0.32",
                "last_price_dollars": "0.31", "volume_fp": "200.0", "open_interest_fp": "150.0",
            },
            {
                "ticker": "KXUFCMOF-26JUN27ABDNAS-DRAW",
                "event_ticker": "KXUFCMOF-26JUN27ABDNAS",
                "custom_strike": {"Method": "Draw/No Contest"},
                "yes_bid_dollars": "0.0", "yes_ask_dollars": "0.01",
                "last_price_dollars": "0.0", "volume_fp": "0", "open_interest_fp": "0",
            },
        ]}

    def test_three_quotes_parsed_draw_skipped(self):
        quotes = _parse_mof_markets(self._payload())
        assert len(quotes) == 3

    def test_kinds_mapped_fight_level(self):
        quotes = _parse_mof_markets(self._payload())
        kinds = {q.market_kind for q in quotes}
        assert kinds == {"mof_sub", "mof_ko", "mof_dec"}
        assert all(q.fighter_name == "" for q in quotes)

    def test_suffix_fallback_when_custom_strike_missing(self):
        payload = {"markets": [{
            "ticker": "KXUFCMOF-26JUN27ABDNAS-KOTKODQ",
            "event_ticker": "KXUFCMOF-26JUN27ABDNAS",
            "yes_bid_dollars": "0.6", "yes_ask_dollars": "0.62",
            "last_price_dollars": "0.61", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        quotes = _parse_mof_markets(payload)
        assert len(quotes) == 1
        assert quotes[0].market_kind == "mof_ko"

    def test_suffix_fallback_draw_skipped(self):
        payload = {"markets": [{
            "ticker": "KXUFCMOF-26JUN27ABDNAS-DRAW",
            "event_ticker": "KXUFCMOF-26JUN27ABDNAS",
            "yes_bid_dollars": "0.0", "yes_ask_dollars": "0.01",
            "last_price_dollars": "0.0", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        assert _parse_mof_markets(payload) == []

    def test_empty_markets_returns_empty(self):
        assert _parse_mof_markets({"markets": []}) == []


# ---------------------------------------------------------------------------
# _parse_rounds_markets (KXUFCROUNDS — "ends before round r", any digit)
# ---------------------------------------------------------------------------

class TestParseRoundsMarkets:
    def _market(self, suffix):
        return {
            "ticker": f"KXUFCROUNDS-26JUN27ABDNAS-{suffix}",
            "event_ticker": "KXUFCROUNDS-26JUN27ABDNAS",
            "yes_bid_dollars": "0.10", "yes_ask_dollars": "0.12",
            "last_price_dollars": "0.11", "volume_fp": "100.0", "open_interest_fp": "90.0",
        }

    def test_round_2_and_3_parsed(self):
        payload = {"markets": [self._market("2"), self._market("3")]}
        quotes = _parse_rounds_markets(payload)
        kinds = {q.market_kind for q in quotes}
        assert kinds == {"end_before_r2", "end_before_r3"}
        assert all(q.fighter_name == "" for q in quotes)

    def test_any_digit_accepted_not_just_2_3(self):
        payload = {"markets": [self._market("4")]}
        quotes = _parse_rounds_markets(payload)
        assert quotes[0].market_kind == "end_before_r4"

    def test_non_digit_suffix_skipped(self):
        payload = {"markets": [self._market("OTHER")]}
        assert _parse_rounds_markets(payload) == []

    def test_empty_markets_returns_empty(self):
        assert _parse_rounds_markets({"markets": []}) == []


# ---------------------------------------------------------------------------
# _parse_vicround_markets (KXUFCVICROUND — per-fighter round win + OTHER)
# ---------------------------------------------------------------------------

class TestParseVicroundMarkets:
    def _fighter_market(self, participant="Jefferson Nascimento", round_="3"):
        return {
            "ticker": "KXUFCVICROUND-26JUN27ABDNAS-NAS3",
            "event_ticker": "KXUFCVICROUND-26JUN27ABDNAS",
            "custom_strike": {"Participant": participant, "Round": round_},
            "yes_sub_title": f"{participant} to win in Round {round_}",
            "yes_bid_dollars": "0.05", "yes_ask_dollars": "0.07",
            "last_price_dollars": "0.06", "volume_fp": "50.0", "open_interest_fp": "40.0",
        }

    def _other_market(self):
        return {
            "ticker": "KXUFCVICROUND-26JUN27ABDNAS-OTHER",
            "event_ticker": "KXUFCVICROUND-26JUN27ABDNAS",
            "custom_strike": {"Participant": "Other", "Round": "Decision / Draw / No Contest"},
            "yes_sub_title": "Decision / Draw / No Contest",
            "yes_bid_dollars": "0.01", "yes_ask_dollars": "0.02",
            "last_price_dollars": "0.01", "volume_fp": "10.0", "open_interest_fp": "8.0",
        }

    def test_fighter_round_from_custom_strike(self):
        quotes = _parse_vicround_markets({"markets": [self._fighter_market()]})
        assert len(quotes) == 1
        assert quotes[0].market_kind == "win_in_r3"
        assert quotes[0].fighter_name == "Jefferson Nascimento"

    def test_other_bucket_is_fight_level(self):
        quotes = _parse_vicround_markets({"markets": [self._other_market()]})
        assert len(quotes) == 1
        assert quotes[0].market_kind == "vicround_other"
        assert quotes[0].fighter_name == ""

    def test_both_together(self):
        quotes = _parse_vicround_markets({"markets": [self._fighter_market(), self._other_market()]})
        kinds = {q.market_kind for q in quotes}
        assert kinds == {"win_in_r3", "vicround_other"}

    def test_fallback_to_title_when_custom_strike_missing(self):
        payload = {"markets": [{
            "ticker": "KXUFCVICROUND-26JUN27ABDNAS-NAS3",
            "event_ticker": "KXUFCVICROUND-26JUN27ABDNAS",
            "yes_sub_title": "Jefferson Nascimento to win in Round 3",
            "yes_bid_dollars": "0.05", "yes_ask_dollars": "0.07",
            "last_price_dollars": "0.06", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        quotes = _parse_vicround_markets(payload)
        assert len(quotes) == 1
        assert quotes[0].market_kind == "win_in_r3"
        assert quotes[0].fighter_name == "Jefferson Nascimento"

    def test_fallback_other_suffix_when_custom_strike_missing(self):
        payload = {"markets": [{
            "ticker": "KXUFCVICROUND-26JUN27ABDNAS-OTHER",
            "event_ticker": "KXUFCVICROUND-26JUN27ABDNAS",
            "yes_sub_title": "Decision / Draw / No Contest",
            "yes_bid_dollars": "0.01", "yes_ask_dollars": "0.02",
            "last_price_dollars": "0.01", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        quotes = _parse_vicround_markets(payload)
        assert quotes[0].market_kind == "vicround_other"

    def test_unparseable_market_skipped(self):
        payload = {"markets": [{
            "ticker": "KXUFCVICROUND-26JUN27ABDNAS-WEIRD",
            "event_ticker": "KXUFCVICROUND-26JUN27ABDNAS",
            "yes_sub_title": "Something unexpected",
            "yes_bid_dollars": "0.01", "yes_ask_dollars": "0.02",
            "last_price_dollars": "0.01", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        assert _parse_vicround_markets(payload) == []

    def test_never_derives_fighter_from_ticker_abbreviation(self):
        # "NAS" is a truncated fragment — must never become the fighter name even
        # if title/custom_strike parsing fails; the market must be skipped instead.
        payload = {"markets": [{
            "ticker": "KXUFCVICROUND-26JUN27ABDNAS-NAS3",
            "event_ticker": "KXUFCVICROUND-26JUN27ABDNAS",
            "yes_sub_title": "",
            "yes_bid_dollars": "0.01", "yes_ask_dollars": "0.02",
            "last_price_dollars": "0.01", "volume_fp": "0", "open_interest_fp": "0",
        }]}
        quotes = _parse_vicround_markets(payload)
        assert quotes == []

    def test_empty_markets_returns_empty(self):
        assert _parse_vicround_markets({"markets": []}) == []


# ---------------------------------------------------------------------------
# is_fight_level_kind
# ---------------------------------------------------------------------------

class TestIsFightLevelKind:
    @pytest.mark.parametrize("kind", [
        "distance", "mof_ko", "mof_sub", "mof_dec", "vicround_other",
        "end_before_r2", "end_before_r3", "end_before_r5",
    ])
    def test_fight_level_kinds(self, kind):
        assert is_fight_level_kind(kind) is True

    @pytest.mark.parametrize("kind", [
        "winner", "method_ko", "method_sub", "method_dec", "win_in_r1", "win_in_r3",
    ])
    def test_fighter_level_kinds(self, kind):
        assert is_fight_level_kind(kind) is False


# ---------------------------------------------------------------------------
# resolve_markets_to_card — two-pass fight-level resolution via event code
# ---------------------------------------------------------------------------

class TestResolveFightLevelMarkets:
    def _card_matchups(self):
        return [
            ("Benoit Saint-Denis", "Paddy Pimblett", 3, False, date(2026, 7, 11),
             "Lightweight", "", "London, UK"),
        ]

    def _named_quote(self, fighter_name, event_ticker="KXUFCFIGHT-26JUL11SAIPIM"):
        return MarketQuote(
            platform="kalshi", event_ticker=event_ticker, market_ticker=event_ticker + "-X",
            market_kind="winner", fighter_name=fighter_name, yes_bid=0.5, yes_ask=0.51,
            last_price=0.5, volume=100.0, open_interest=100.0, fetched_at="2026-07-02T00:00:00Z",
        )

    def _fight_level_quote(self, market_kind="distance", event_ticker="KXUFCDISTANCE-26JUL11SAIPIM"):
        return MarketQuote(
            platform="kalshi", event_ticker=event_ticker, market_ticker=event_ticker + "-DIST",
            market_kind=market_kind, fighter_name="", yes_bid=0.3, yes_ask=0.32,
            last_price=0.31, volume=50.0, open_interest=40.0, fetched_at="2026-07-02T00:00:00Z",
        )

    def test_fight_level_resolves_via_event_code_when_named_quote_resolved(self):
        resolved, unresolved = resolve_markets_to_card(
            [self._named_quote("Benoit Saint-Denis"), self._fight_level_quote()],
            self._card_matchups())
        kinds = {r.market_kind for r in resolved}
        assert "distance" in kinds
        dist = [r for r in resolved if r.market_kind == "distance"][0]
        assert dist.corner == "fight"
        assert dist.fighter_name == ""
        assert dist.fight_idx == 0
        assert unresolved == []

    def test_fight_level_unresolved_without_a_matching_named_quote(self):
        resolved, unresolved = resolve_markets_to_card(
            [self._fight_level_quote(event_ticker="KXUFCDISTANCE-26AUG01NONEXIST")],
            self._card_matchups())
        assert resolved == []
        assert len(unresolved) == 1
        assert "fight-level" in unresolved[0]

    def test_fight_level_date_guard_still_applies(self):
        # Named quote resolves (event code "26JUL11SAIPIM" maps to fight 0), but the
        # fight-level quote's OWN event ticker is >1 day off — must not resolve.
        resolved, unresolved = resolve_markets_to_card(
            [self._named_quote("Benoit Saint-Denis"),
             self._fight_level_quote(event_ticker="KXUFCDISTANCE-26SEP05SAIPIM")],
            self._card_matchups())
        kinds = {r.market_kind for r in resolved}
        assert "distance" not in kinds

    def test_multiple_fight_level_kinds_resolve_together(self):
        quotes = [
            self._named_quote("Benoit Saint-Denis"),
            self._fight_level_quote(market_kind="distance"),
            self._fight_level_quote(market_kind="mof_dec", event_ticker="KXUFCMOF-26JUL11SAIPIM"),
            self._fight_level_quote(market_kind="vicround_other", event_ticker="KXUFCVICROUND-26JUL11SAIPIM"),
        ]
        resolved, _ = resolve_markets_to_card(quotes, self._card_matchups())
        kinds = {r.market_kind for r in resolved}
        assert kinds == {"winner", "distance", "mof_dec", "vicround_other"}
        assert all(r.corner == "fight" for r in resolved if r.market_kind != "winner")


# ---------------------------------------------------------------------------
# SERIES_TABLE / fetch_all_markets — series isolation, depth-fetch scoping
# ---------------------------------------------------------------------------

class TestSeriesTable:
    def test_winner_is_first_and_required(self):
        assert SERIES_TABLE[0].series_ticker == "KXUFCFIGHT"
        assert SERIES_TABLE[0].required is True

    def test_six_series_registered(self):
        tickers = {s.series_ticker for s in SERIES_TABLE}
        assert tickers == {
            "KXUFCFIGHT", "KXUFCMOV", "KXUFCDISTANCE",
            "KXUFCMOF", "KXUFCROUNDS", "KXUFCVICROUND",
        }

    def test_only_winner_and_method_fetch_depth(self):
        depth_on = {s.series_ticker for s in SERIES_TABLE if s.with_depth}
        assert depth_on == {"KXUFCFIGHT", "KXUFCMOV"}


class TestFetchAllMarketsIsolation:
    def test_new_series_failure_does_not_block_winner(self, monkeypatch):
        import ufc.ingest.market_lines as ml

        def fake_fetch_all_pages(session, series_ticker, status, timeout=10, max_pages=10):
            if series_ticker == "KXUFCROUNDS":
                raise RuntimeError("boom")
            if series_ticker == "KXUFCFIGHT":
                return {"markets": [{
                    "ticker": "KXUFCFIGHT-26JUL11SAIPIM-SAI",
                    "event_ticker": "KXUFCFIGHT-26JUL11SAIPIM",
                    "yes_sub_title": "Benoit Saint-Denis",
                    "yes_bid_dollars": "0.55", "yes_ask_dollars": "0.56",
                    "last_price_dollars": "0.56", "volume_fp": "100", "open_interest_fp": "90",
                }]}
            return {"markets": []}

        import ufc.io.paths as paths_mod
        import tempfile
        from pathlib import Path

        monkeypatch.setattr(ml, "_fetch_all_pages", fake_fetch_all_pages)
        monkeypatch.setattr(ml, "fetch_orderbook_depth", lambda *a, **k: None)
        monkeypatch.setattr(paths_mod, "external_market_lines", lambda: Path(tempfile.mkdtemp()))

        quotes, errors = fetch_all_markets(with_depth=False)
        assert any(q.market_kind == "winner" for q in quotes)
        assert any("rounds" in e for e in errors)

    def test_depth_only_fetched_for_winner_and_method(self, monkeypatch):
        import ufc.ingest.market_lines as ml

        def fake_fetch_all_pages(session, series_ticker, status, timeout=10, max_pages=10):
            suffix_map = {
                "KXUFCFIGHT": ("winner_ticker", "yes_sub_title", "Fighter One"),
                "KXUFCDISTANCE": ("dist_ticker", None, None),
            }
            if series_ticker == "KXUFCFIGHT":
                return {"markets": [{
                    "ticker": "KXUFCFIGHT-26JUL11SAIPIM-SAI",
                    "event_ticker": "KXUFCFIGHT-26JUL11SAIPIM",
                    "yes_sub_title": "Fighter One",
                    "yes_bid_dollars": "0.5", "yes_ask_dollars": "0.51",
                    "last_price_dollars": "0.5", "volume_fp": "0", "open_interest_fp": "0",
                }]}
            if series_ticker == "KXUFCDISTANCE":
                return {"markets": [{
                    "ticker": "KXUFCDISTANCE-26JUL11SAIPIM-DIST",
                    "event_ticker": "KXUFCDISTANCE-26JUL11SAIPIM",
                    "yes_bid_dollars": "0.3", "yes_ask_dollars": "0.32",
                    "last_price_dollars": "0.31", "volume_fp": "0", "open_interest_fp": "0",
                }]}
            return {"markets": []}

        depth_calls = []

        def fake_depth(market_ticker, best_ask, session=None, timeout=10):
            depth_calls.append(market_ticker)
            return 5000.0

        import ufc.io.paths as paths_mod
        import tempfile
        from pathlib import Path

        monkeypatch.setattr(ml, "_fetch_all_pages", fake_fetch_all_pages)
        monkeypatch.setattr(ml, "fetch_orderbook_depth", fake_depth)
        monkeypatch.setattr(paths_mod, "external_market_lines", lambda: Path(tempfile.mkdtemp()))

        quotes, errors = fetch_all_markets(with_depth=True)
        assert depth_calls == ["KXUFCFIGHT-26JUL11SAIPIM-SAI"]


# ---------------------------------------------------------------------------
# _divergence_flag (ufc.api.app — market-lines row builder flag logic)
# ---------------------------------------------------------------------------

def test_divergence_flag_threshold():
    """|model_p - ask| >= 0.20 flags the row; below stays unflagged."""
    from ufc.api.app import _divergence_flag
    assert _divergence_flag(model_p=0.636, ask=0.25) is True    # 38.6pp gap
    assert _divergence_flag(model_p=0.55, ask=0.45) is False    # 10pp gap
    assert _divergence_flag(model_p=0.30, ask=0.52) is True     # negative side too
    assert _divergence_flag(model_p=0.65, ask=0.45) is True     # exactly 0.20 -> flag
