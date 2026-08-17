"""Pretty-print prediction results to terminal."""
from __future__ import annotations

from ufc.valuation.edge import Edge
from ufc.valuation.portfolio import PortfolioResult


def print_matchup_header(red_name: str, blue_name: str, event_date=None, rounds: int = 3, is_title: bool = False):
    sep = "=" * 60
    title_str = " [TITLE FIGHT]" if is_title else ""
    date_str = f" | {event_date}" if event_date else ""
    print(f"\n{sep}")
    print(f"  {red_name} (RED)  vs.  {blue_name} (BLUE)")
    print(f"  {rounds} Rounds{title_str}{date_str}")
    print(sep)


def print_winner_probs(red_name: str, blue_name: str, prob_red: float):
    prob_blue = 1.0 - prob_red
    bar_len = 40
    red_bar = "█" * int(prob_red * bar_len)
    blue_bar = "█" * int(prob_blue * bar_len)
    print(f"\n  WINNER PROBABILITY")
    print(f"  {red_name:<25} {prob_red*100:5.1f}%  {red_bar}")
    print(f"  {blue_name:<25} {prob_blue*100:5.1f}%  {blue_bar}")


def print_method_dist(method_probs: dict[str, float]):
    print(f"\n  METHOD DISTRIBUTION")
    for method, prob in sorted(method_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {method:<12} {prob*100:5.1f}%  {bar}")


def print_prop_cdf(name: str, cdf, label: str = ""):
    if cdf is None:
        return
    p25 = cdf.quantile(0.25)
    p50 = cdf.quantile(0.50)
    p75 = cdf.quantile(0.75)
    print(f"\n  {name.upper()}{' — ' + label if label else ''}")
    print(f"  Median:  {p50:.1f}  |  25th: {p25:.1f}  |  75th: {p75:.1f}")


def print_duration_cdf(cdf):
    if cdf is None:
        return
    med_sec = cdf.median_sec
    med_rounds = med_sec / 300
    print(f"\n  FIGHT DURATION")
    print(f"  Median: {med_sec:.0f}s ({med_rounds:.1f} rounds)")
    for rounds in [1.5, 2.5, 3.5, 4.5]:
        p = cdf.p_over_rounds(rounds)
        print(f"  P(goes > {rounds}rds): {p*100:.1f}%")


def print_edge(edge: Edge, min_edge: float = 0.0):
    direction = "OVER" if edge.side == "over" else "UNDER" if edge.side == "under" else edge.side.upper()
    fighter_str = f" [{edge.fighter_name}]" if edge.fighter_name else ""
    line_str = f"{edge.line_value:.1f}" if edge.line_value else ""
    flag = " ★ BET" if edge.edge_pct >= min_edge else ""
    band_lo, band_hi = edge.confidence_band
    print(f"\n  {edge.market.upper()} {direction} {line_str}{fighter_str}")
    print(f"  Model: {edge.model_prob*100:.1f}%  |  Implied: {edge.implied_prob*100:.1f}%  |  Edge: {edge.edge_pct*100:+.1f}%{flag}")
    print(f"  Kelly: {edge.kelly_fraction*100:.1f}%  |  80% Band: [{band_lo*100:.1f}%, {band_hi*100:.1f}%]")


def print_portfolio(result: PortfolioResult):
    if result.n_legs == 0:
        return
    print(f"\n  PARLAY ({result.n_legs} legs)")
    print(f"  Naive joint prob (independence): {result.naive_joint_prob*100:.1f}%")
    print(f"  MC joint prob (correlated):      {result.mc_joint_prob*100:.1f}%")
    adj = result.correlation_adjustment * 100
    sign = "+" if adj >= 0 else ""
    print(f"  Correlation adjustment:          {sign}{adj:.1f}%")


def print_rounds_distribution(dur_cdf, scheduled_rounds: int):
    if dur_cdf is None:
        return
    print("\n  ROUNDS")
    for k in range(1, scheduled_rounds + 1):
        lo = (k - 1) * 300
        hi = k * 300
        p_in_k = dur_cdf.cdf(hi) - dur_cdf.cdf(lo)
        print(f"  Ends in R{k}: {p_in_k*100:5.1f}%")
    for line in [1.5, 2.5, 3.5, 4.5]:
        if line < scheduled_rounds + 0.5:
            p_over = 1 - dur_cdf.cdf(line * 300)
            print(f"  P(>{line} rounds): {p_over*100:5.1f}%")


def print_separator():
    print("-" * 60)
