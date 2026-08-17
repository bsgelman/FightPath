/* Presentational statistics helpers — server handles all MC computation. */

/* Find x where survival curve crosses a target probability (used for median line). */
export function survivalCrossing(curve, target) {
  for (let i = 1; i < curve.length; i++) {
    const [x0, p0] = curve[i - 1], [x1, p1] = curve[i];
    if ((p0 - target) * (p1 - target) <= 0 && p0 !== p1) {
      const f = (p0 - target) / (p0 - p1);
      return x0 + f * (x1 - x0);
    }
  }
  return null;
}

/* Linear interpolation over a precomputed [[x, survival]] curve. */
export function survivalAt2(curve, x) {
  if (!curve || curve.length === 0) return 0;
  if (x <= curve[0][0]) return curve[0][1];
  for (let i = 1; i < curve.length; i++) {
    if (x <= curve[i][0]) {
      const [x0, p0] = curve[i - 1], [x1, p1] = curve[i];
      const f = (x - x0) / (x1 - x0 || 1);
      return p0 + f * (p1 - p0);
    }
  }
  return curve[curve.length - 1][1];
}

/* DFS payout break-even per leg. */
export function breakEvenPerLeg(mult, legs) {
  return Math.pow(1 / mult, 1 / legs);
}

/* No-vig fair probability for the OVER side, de-vigged from the two per-side
 * break-evens. The break-evens already embed the house vig (they sum to >1); the
 * fair line normalizes them so the two sides sum to 1. For a flat pick'em where
 * beOver == beUnder this returns 0.50. Returns null when there is no opposite side
 * to de-vig against (one-sided demon/goblin/finish lines) — fair edge is undefined. */
export function noVigFairOver(beOver, beUnder) {
  if (!(beOver > 0) || !(beUnder > 0)) return null;
  return beOver / (beOver + beUnder);
}

/* Confidence-weighted pick score — ranks a likelier pick above a longshot with a
 * bigger raw edge. Heuristic: monotone in both edge and probability.
 * Only POSITIVE edges get the confidence weighting: scaling a negative edge by
 * probability inverts its order (-20pp @ 30% would outrank -10pp @ 90%), so
 * non-positive edges keep their raw value. Since a weighted positive edge stays
 * positive, every playable row still sorts above every negative-edge row — which
 * the Kalshi lane displays, as it shows the whole market, not just the picks. */
export function pickScore(r) {
  const edge = r.edgePct ?? 0;
  return edge > 0 ? edge * (r.modelP ?? 0) : edge;
}

/* Fractional Kelly for a single leg, capped. */
export function kellyFraction(p, breakeven, cap) {
  if (p <= breakeven) return 0;
  const b = (1 / breakeven) - 1;
  const f = (p * b - (1 - p)) / b;
  return Math.max(0, Math.min(cap, f));
}

/* 80% credible band on a probability estimate from n samples (Wald). */
export function probBand(p, n, z) {
  const se = Math.sqrt(Math.max(p * (1 - p), 1e-6) / n);
  return [Math.max(0, p - z * se), Math.min(1, p + z * se)];
}

/* Wilson score interval — better than Wald at small n / extreme p (early-event
 * cumulative accuracy has both). z=1.96 ≈ 95% CI. */
export function wilsonInterval(correct, total, z = 1.96) {
  if (!total) return [0, 1];
  const p = correct / total;
  const z2 = z * z;
  const denom = 1 + z2 / total;
  const center = (p + z2 / (2 * total)) / denom;
  const margin = (z * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)) / denom;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}
