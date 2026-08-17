#!/usr/bin/env python
"""UFC Prediction CLI.


Usage:
  # Single matchup
  python predict.py --red "Ilia Topuria" --blue "Max Holloway" \
      --rounds 5 --title --event-date 2026-05-16 \
      --prop sig_strikes:red:over:52.5 \
      --prop takedowns:red:under:0.5 \
      --prop duration:over:13.5min \
      --payout powerplay_power_3pick \
      --simulate 50000

  # Full card
  python predict.py --card path/to/card.json --payout powerplay_power_2pick
"""
import argparse
import sys
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from datetime import date, datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd


def _load_models():
    """Load all available trained models (delegates to predict_core)."""
    from ufc.inference.predict_core import load_models
    return load_models(verbose=True)


def predict_matchup(
    red_name: str,
    blue_name: str,
    rounds: int,
    is_title: bool,
    event_date: date,
    prop_strings: list[str],
    payout_type: str,
    n_simulate: int,
    models: dict,
    fighters_df: pd.DataFrame,
    pre_fight_state: pd.DataFrame,
    location: str = "",
    referee: str = "",
    weight_class: str | None = None,
    ref_history_df: pd.DataFrame | None = None,
) -> None:
    from ufc.inference.predict_core import predict_fight
    from ufc.inference import pretty
    from ufc.valuation.lines import CLILineSource
    from ufc.valuation.edge import evaluate_line
    from ufc.valuation.portfolio import evaluate_portfolio

    try:
        r = predict_fight(
            red_name, blue_name, rounds, is_title, event_date,
            models, fighters_df, pre_fight_state,
            n_simulate=n_simulate, location=location, referee=referee,
            ref_history_df=ref_history_df,
            run_simulation=True, verbose=True,
            weight_class=weight_class,
        )
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    pretty.print_matchup_header(r.red_name, r.blue_name, event_date, rounds, is_title)
    pretty.print_winner_probs(r.red_name, r.blue_name, r.prob_red)
    pretty.print_method_dist(r.method_probs)

    if r.display_dur_cdf is not None:
        pretty.print_duration_cdf(r.display_dur_cdf)
        pretty.print_rounds_distribution(r.display_dur_cdf, rounds)

    if r.ss_cdf_red is not None:
        pretty.print_prop_cdf("Sig Strikes", r.ss_cdf_red, r.red_name)
        pretty.print_prop_cdf("Sig Strikes", r.ss_cdf_blue, r.blue_name)

    if r.td_cdf_red is not None:
        pretty.print_prop_cdf("Takedowns", r.td_cdf_red, r.red_name)
        pretty.print_prop_cdf("Takedowns", r.td_cdf_blue, r.blue_name)

    if r.r1_cdf_red is not None:
        pretty.print_prop_cdf("R1 Sig Strikes", r.r1_cdf_red, r.red_name)
        pretty.print_prop_cdf("R1 Sig Strikes", r.r1_cdf_blue, r.blue_name)

    # ── Prop valuation ────────────────────────────────────────────────────
    if prop_strings:
        print(f"\n  PROP ANALYSIS ({payout_type})")
        pretty.print_separator()

        line_source = CLILineSource(
            prop_strings, payout_type,
            red_id=r.red_id, blue_id=r.blue_id,
            red_name=r.red_name, blue_name=r.blue_name,
        )
        lines = line_source.fetch()

        min_edge = 0.05

        from ufc.evaluation.prop_plot import plot_prop_distribution, _slug
        from ufc.io import paths as _paths

        pp_root = _paths.outputs_reports() / "prop_distributions"
        fight_dir = (pp_root
                     / f"{event_date.isoformat()}_{_slug(r.red_name)}_vs_{_slug(r.blue_name)}")
        plots_saved: list[str] = []

        edges = []
        for line in lines:
            if line.market == "sig_strikes":
                pred = r.ss_cdf_red if line.fighter_id == r.red_id else r.ss_cdf_blue
            elif line.market == "takedowns":
                pred = r.td_cdf_red if line.fighter_id == r.red_id else r.td_cdf_blue
            elif line.market in ("r1_sig_strikes", "r1_significant_strikes"):
                pred = r.r1_cdf_red if line.fighter_id == r.red_id else r.r1_cdf_blue
            elif line.market in ("duration_sec", "duration"):
                pred = r.dur_cdf
            elif line.market in ("rounds", "n_rounds", "round", "number_of_rounds"):
                pred = r.dur_cdf
            elif line.market == "winner":
                pred = r.prob_red if line.fighter_id == r.red_id else r.prob_blue
            else:
                pred = None

            if pred is None:
                print(f"  Skipping {line.market} (no model loaded)")
                continue

            edge = evaluate_line(line, pred)
            edges.append(edge)
            pretty.print_edge(edge, min_edge=min_edge)

            if not isinstance(pred, float):
                try:
                    fight_dir.mkdir(parents=True, exist_ok=True)
                    fname = (f"{line.market}_{_slug(line.fighter_name or 'fight')}"
                             f"_{line.side}_{line.line_value}.png")
                    plot_prop_distribution(
                        cdf=pred,
                        line=line.line_value,
                        market=line.market,
                        save_path=fight_dir / fname,
                        fighter_label=line.fighter_name or f"{r.red_name} vs {r.blue_name}",
                        edge=edge,
                        title_suffix=f"({line.payout_type})",
                    )
                    plots_saved.append(fname)
                except Exception as _plot_err:
                    print(f"  [warn] prop plot failed for {line.market}: {_plot_err}")

        if len(edges) > 1:
            portfolio = evaluate_portfolio(lines, edges, r.sim_samples, n_simulate)
            pretty.print_portfolio(portfolio)

        if plots_saved:
            print(f"\n  Distribution plots saved to: {fight_dir}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="UFC Fight & Prop Predictor")
    parser.add_argument("--red", type=str, help="Red corner fighter name")
    parser.add_argument("--blue", type=str, help="Blue corner fighter name")
    parser.add_argument("--rounds", type=int, default=3, choices=[3, 5])
    parser.add_argument("--title", action="store_true", help="Title fight")
    parser.add_argument("--event-date", type=str, default=str(date.today()),
                        help="Event date YYYY-MM-DD (default: today)")
    parser.add_argument("--prop", action="append", dest="props", default=[],
                        metavar="PROP",
                        help="Prop line: 'sig_strikes:red:over:52.5' (repeat for multiple)")
    parser.add_argument("--payout", type=str, default="powerplay_power_3pick",
                        help="Payout type (e.g. powerplay_power_3pick, flatmulti_standard_2pick)")
    parser.add_argument("--simulate", type=int, default=50000,
                        help="Monte Carlo simulation samples (default: 50000)")
    parser.add_argument("--location", type=str, default="",
                        help="Event city/location (used for altitude lookup)")
    parser.add_argument("--referee", type=str, default="",
                        help="Referee name (used for stoppage tendency feature)")
    parser.add_argument("--weight-class", type=str, default=None,
                        help="Override weight class (e.g. 'Lightweight'); inferred from fighters if omitted")
    parser.add_argument("--card", type=str, default=None,
                        help="Path to card JSON file for full event")

    args = parser.parse_args()

    if not args.red and not args.card:
        parser.print_help()
        sys.exit(1)

    from ufc.io import paths, parquet

    model_dir = paths.outputs_models()
    if not any(model_dir.glob("winner_ensemble_*.joblib")):
        print("ERROR: No trained winner model found.")
        print("Run: python scripts/03_train.py")
        sys.exit(1)

    print("Loading models...")
    models = _load_models()

    print("Loading fighter database...")
    fighters_df = parquet.read(paths.interim("fighters"))
    pre_fight_state = parquet.read(paths.processed("pre_fight_state"))
    from ufc.inference.ref_history import build_ref_history
    ref_history_df = build_ref_history()

    if args.card:
        from ufc.inference.card import parse_card
        card = parse_card(args.card)
        event_date = card.event_date
        payout_type = args.payout or card.default_payout

        print(f"\nEvent: {card.event_name} | {event_date}")
        for matchup in card.matchups:
            prop_strs = [
                f"{p.market}:{p.fighter}:{p.side}:{p.line}"
                for p in matchup.props
            ]
            predict_matchup(
                red_name=matchup.red,
                blue_name=matchup.blue,
                rounds=matchup.scheduled_rounds,
                is_title=matchup.is_title,
                event_date=event_date,
                prop_strings=prop_strs,
                payout_type=payout_type,
                n_simulate=args.simulate,
                models=models,
                fighters_df=fighters_df,
                pre_fight_state=pre_fight_state,
                location=args.location,
                referee=args.referee,
                weight_class=matchup.weight_class,
                ref_history_df=ref_history_df,
            )

    else:
        event_date = datetime.strptime(args.event_date, "%Y-%m-%d").date()
        predict_matchup(
            red_name=args.red,
            blue_name=args.blue,
            rounds=args.rounds,
            is_title=args.title,
            event_date=event_date,
            prop_strings=args.props,
            payout_type=args.payout,
            n_simulate=args.simulate,
            models=models,
            fighters_df=fighters_df,
            pre_fight_state=pre_fight_state,
            location=args.location,
            referee=args.referee,
            weight_class=args.weight_class,
            ref_history_df=ref_history_df,
        )


if __name__ == "__main__":
    main()
