"""Capture closing lines for pending upcoming prop rows.

Run on the day of the card (Saturday typically), AFTER 07_fetch_props.py has pulled
the current board into last_pull.json. Writes close_line_value + close_captured_at
to pending rows whose fight has not yet happened.

    python scripts/07c_capture_closing_lines.py

Safe to run multiple times — last-write-wins on close_line_value, only touches
pending rows for today or future dates. No model load; reads only from disk.

Match key: (matchup_key, market, corner, platform) — NOT line_value, because the
closing line number differs from the open line. Over and under rows for the same
prop share the same closing line value. Kalshi taker+maker rows share this key too
(same market/corner/platform) — both get the same fresh ask as their close, which
is correct: they're two prices for the same underlying quote.
"""
import json
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ufc.io import paths
from ufc.inference.card import parse_card
from ufc.inference.prediction_log import matchup_key
from ufc.inference.prop_prediction_log import load_log, save_log, now_iso
from ufc.ingest.prop_lines import LiveProp, resolve_to_card
from ufc.ingest.market_lines import fetch_all_markets, resolve_markets_to_card


def main():
    upcoming_dir = paths.upcoming_cards()
    card_paths = sorted(upcoming_dir.glob("*.json")) if upcoming_dir.exists() else []
    if not card_paths:
        print("[close-lines] No upcoming card files found.")
        return

    card_matchups = []
    for cp in card_paths:
        try:
            spec = parse_card(cp)
            for m in spec.matchups:
                card_matchups.append((
                    m.red, m.blue,
                    m.scheduled_rounds,
                    m.is_title,
                    spec.event_date,
                    m.weight_class,
                    m.referee or "",
                    spec.location or "",
                ))
        except Exception as e:
            print(f"  [warn] Could not parse {cp.name}: {e}")

    if not card_matchups:
        print("[close-lines] No matchups loaded from upcoming cards.")
        return

    # ── DFS lane (Power Play/Flat Multi) — optional, does not block Kalshi below ──
    resolved = []
    lines_path = paths.external_lines() / "last_pull.json"
    if not lines_path.exists():
        print("[close-lines] No DFS last_pull.json found — skipping DFS lane "
              "(run 07_fetch_props.py first if you want it).")
    else:
        raw = json.loads(lines_path.read_text(encoding="utf-8"))
        if not raw:
            print("[close-lines] DFS last_pull.json is empty — skipping DFS lane.")
        else:
            print(f"[close-lines] Loaded {len(raw)} lines from last_pull.json")
            live_props = [
                LiveProp(
                    platform=p["platform"],
                    player_name=p["player_name"],
                    market=p["market"],
                    line_value=float(p["line_value"]),
                    raw_stat=p.get("raw_stat", ""),
                    odds_type=p.get("odds_type", "standard"),
                    board_multiplier=p.get("board_multiplier"),
                    directional=bool(p.get("directional", False)),
                    under_only=bool(p.get("under_only", False)),
                    over_multiplier=p.get("over_multiplier"),
                    under_multiplier=p.get("under_multiplier"),
                )
                for p in raw
            ]
            resolved, _ = resolve_to_card(live_props, card_matchups)
            if not resolved:
                print("[close-lines] No DFS props resolved to card — check fighter name matching.")
            else:
                print(f"[close-lines] {len(resolved)} DFS props resolved to {len(card_matchups)} fights")

    # Build close lookup: (matchup_key, market, corner, platform) -> closing line_value.
    # Do NOT key on line_value — the close number differs from the open.
    close_lookup: dict[tuple, float] = {}
    for prop in resolved:
        fi = prop.fight_idx
        ev_date = str(card_matchups[fi][4])[:10]
        mk = matchup_key(ev_date, prop.card_red, prop.card_blue)
        lookup_key = (mk, prop.market, prop.corner, prop.platform)
        close_lookup[lookup_key] = prop.line_value  # last-write-wins if duplicates

    kalshi_quotes, kalshi_errors = fetch_all_markets(with_depth=False)
    for e in kalshi_errors:
        print(f"  [kalshi] {e}")
    if kalshi_quotes:
        kalshi_resolved, _ = resolve_markets_to_card(kalshi_quotes, card_matchups)
        for rq in kalshi_resolved:
            if rq.yes_ask is None:
                continue
            ev_date = str(card_matchups[rq.fight_idx][4])[:10]
            mk = matchup_key(ev_date, rq.card_red, rq.card_blue)
            close_lookup[(mk, rq.market_kind, rq.corner, "kalshi")] = rq.yes_ask
        print(f"[close-lines] {len(kalshi_resolved)} Kalshi quotes resolved for closing capture")

    log = load_log()
    today = str(date.today())
    # Guard: only touch pending rows for events that haven't happened yet.
    target_mask = (
        (log["status"] == "pending") &
        (log["event_date"].astype(str).str[:10] >= today)
    )

    updated = 0
    no_match = 0
    ts = now_iso()

    for idx in log.index[target_mask]:
        row = log.loc[idx]
        mk = matchup_key(
            str(row["event_date"])[:10],
            str(row["red"]),
            str(row["blue"]),
        )
        lookup_key = (mk, str(row["market"]), str(row["corner"]), str(row["platform"]))
        close_val = close_lookup.get(lookup_key)
        if close_val is not None:
            log.at[idx, "close_line_value"] = float(close_val)
            log.at[idx, "close_captured_at"] = ts
            updated += 1
        else:
            no_match += 1

    if updated:
        save_log(log)

    print(
        f"[close-lines] captured {updated} · "
        f"{no_match} pending upcoming rows had no close on the board"
    )


if __name__ == "__main__":
    main()
