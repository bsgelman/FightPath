/* Position selection and sizing for the Kalshi board.
 *
 * Two invariants this module exists to enforce, both measured as broken on the
 * 2026-07-11 card (see docs/superpowers/specs/2026-07-28-positions-board-design.md):
 *   1. ONE contract per fight. The old board picked Garbrandt to win AND Yanez to
 *      finish him in R1 - mutually exclusive - on 11 of 23 fights.
 *   2. Total stake <= card budget. Per-row quarter-Kelly asked for $122 on a $64
 *      bankroll because Kelly assumes sequential bets, not ~68 simultaneous ones.
 *
 * Pure functions over plain objects. No React imports - lib/positions.test.mjs
 * runs this directly under `node --test`.
 */
import { kellyFraction } from "./stats.js";

/* Pre-registered 2026-07-28. Deliberately NOT fitted: fitting it on 47 fights is
 * the overfitting this whole design exists to avoid. Neutral two-estimator prior.
 * Revisit only after 14 further cards, and record the change in the spec. */
export const SHRINK_W = 0.5;

const KELLY_FRACTION = 0.25;   // quarter-Kelly
const MIN_STAKE_USD = 1;

/* Confidence floor on the SHRUNK probability: never hold a position you think is
 * less likely than not once your belief has been reconciled with the market.
 *
 * Pre-registered 2026-07-28 on calibration evidence, not ROI. Measured on the
 * forward ledger (edge>=5pp, sentinels out, fight-clustered CIs), AFTER shrinking:
 *   p* < 0.35     -5.5pp  CI[-10.1, -0.7]   still significantly overstated
 *   p* 0.35-0.50 -18.1pp  CI[-35.6, +5.5]   large negative, underpowered
 *   p* >= 0.50    +3.9pp  CI[-15.7, +21.7]  no measurable effect
 * 0.50 is where the significant overstatement stops - a natural constant, not a
 * tuned one. ROI could not decide it: every variant's CI spanned zero at n=26.
 *
 * Known cost: this removes the cheap longshot contracts that carried the ledger's
 * entire realized ROI (+233.7% over 17 fights, at an 11.8% hit rate against a ~30%
 * prediction). That return is tail-driven and not distinguishable from zero. Those
 * contracts stay visible in the Explore view; they just aren't staked here. */
export const MIN_PSTAR = 0.50;

/* The low-data "no opinion" sentinel. predict_core pins debut fights at exactly
 * 0.5; subtracting an informed market price from it prints a phantom 30-40pp edge.
 * This is the bug that produced a spurious "0-for-14" in the July 2026 analysis. */
const SENTINEL_P = 0.5;
const isSentinel = (r) => Math.abs(r.modelP - SENTINEL_P) < 1e-9;

/** Pull the model probability halfway toward the market's. */
export function shrinkProb(modelP, feeAdjBE) {
  return SHRINK_W * modelP + (1 - SHRINK_W) * feeAdjBE;
}

/** One contract per fight: the highest quarter-Kelly fraction on the shrunk
 *  probability. Ties break to the cheaper contract. Rows with no positive edge
 *  after shrinking are dropped, so a fight can end up with no position at all.
 *
 *  kellyFraction() already returns 0 when p <= breakeven, so the f > 0 guard is
 *  belt-and-braces rather than the primary filter. */
export function selectPositions(rows) {
  const best = new Map();
  for (const r of rows) {
    if (isSentinel(r)) continue;
    if (!(r.feeAdjBE > 0) || !(r.feeAdjBE < 1)) continue;
    if (!(r.ask > 0)) continue;
    const pStar = shrinkProb(r.modelP, r.feeAdjBE);
    if (pStar < MIN_PSTAR) continue;
    const f = KELLY_FRACTION * kellyFraction(pStar, r.feeAdjBE, 1);
    if (!(f > 0)) continue;
    const cur = best.get(r.fightIdx);
    if (!cur || f > cur.f || (f === cur.f && r.ask < cur.ask)) {
      best.set(r.fightIdx, { ...r, pStar, f });
    }
  }
  return [...best.values()];
}

/** Kelly stake, then a proportional scale-down if the card budget is exceeded.
 *  The budget is a CAP, not a quota - a weak card deploys less than the budget
 *  rather than being topped up to it.
 *
 *  Rows worth less than a dollar (or less than one contract) are dropped after
 *  scaling, so the displayed total can sit slightly under the cap. */
export function sizePositions(selected, bankroll, cardRiskPct) {
  const budget = bankroll * cardRiskPct;
  const raw = selected.map((r) => ({ ...r, stake: r.f * bankroll }));
  const total = raw.reduce((s, r) => s + r.stake, 0);
  const scale = total > budget && total > 0 ? budget / total : 1;
  return raw
    .map((r) => {
      const stake = r.stake * scale;
      return { ...r, stake, contracts: Math.floor(stake / r.ask) };
    })
    .filter((r) => r.stake >= MIN_STAKE_USD && r.contracts >= 1);
}
