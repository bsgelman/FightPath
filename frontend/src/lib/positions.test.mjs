import { test } from "node:test";
import assert from "node:assert/strict";
import { shrinkProb, selectPositions, sizePositions, SHRINK_W } from "./positions.js";

// Real rows from the 2026-07-11 card. fightIdx 0 is the measured contradiction:
// the old board picked Garbrandt to WIN and Yanez to finish him in R1 and R2.
const ROWS = [
  { fightIdx: 0, fighter: "Cody Garbrandt", corner: "red",  marketKind: "winner",     modelP: 0.635657, ask: 0.25, feeAdjBE: 0.253281 },
  { fightIdx: 0, fighter: "Adrian Yanez",   corner: "blue", marketKind: "win_in_r1",  modelP: 0.129560, ask: 0.03, feeAdjBE: 0.030509 },
  { fightIdx: 0, fighter: "Adrian Yanez",   corner: "blue", marketKind: "win_in_r2",  modelP: 0.083740, ask: 0.03, feeAdjBE: 0.030509 },
  { fightIdx: 0, fighter: "Cody Garbrandt", corner: "red",  marketKind: "method_dec", modelP: 0.300240, ask: 0.13, feeAdjBE: 0.131979 },
  // fightIdx 1: the low-data sentinel. modelP is exactly 0.5 = "no opinion".
  // Subtracting an informed 13c price from it prints a phantom +37pp edge.
  { fightIdx: 1, fighter: "Elisha Ellison", corner: "blue", marketKind: "winner",     modelP: 0.500000, ask: 0.13, feeAdjBE: 0.131979 },
  // fightIdx 2: no positive-edge candidate at all -> must abstain.
  { fightIdx: 2, fighter: "Someone",        corner: "red",  marketKind: "winner",     modelP: 0.300000, ask: 0.60, feeAdjBE: 0.604000 },
  // fightIdx 3: the only row here that clears the 0.50 shrunk-probability floor.
  // Real contract off the 2026-08-01 card: model 59.6% at 40c shrinks to p* 50.6%.
  { fightIdx: 3, fighter: "Todorovic",      corner: "red",  marketKind: "winner",     modelP: 0.596000, ask: 0.40, feeAdjBE: 0.416000 },
];

test("shrinkProb pulls the model halfway to market", () => {
  assert.equal(SHRINK_W, 0.5);
  assert.ok(Math.abs(shrinkProb(0.6, 0.4) - 0.5) < 1e-9);
  assert.ok(Math.abs(shrinkProb(0.635657, 0.253281) - 0.444469) < 1e-6);
});

test("selects at most one contract per fight (the coherence fix)", () => {
  const sel = selectPositions(ROWS);
  const seen = new Set();
  for (const r of sel) {
    assert.ok(!seen.has(r.fightIdx), `two positions on fight ${r.fightIdx}`);
    seen.add(r.fightIdx);
  }
});

test("never picks both corners of the same fight", () => {
  const sel = selectPositions(ROWS);
  const byFight = new Map();
  for (const r of sel) {
    assert.ok(!byFight.has(r.fightIdx) || byFight.get(r.fightIdx) === r.corner);
    byFight.set(r.fightIdx, r.corner);
  }
});

test("excludes the 0.500000 low-data sentinel", () => {
  const sel = selectPositions(ROWS);
  assert.equal(sel.some((r) => r.fightIdx === 1), false);
  assert.equal(sel.some((r) => r.modelP === 0.5), false);
});

test("abstains when nothing has positive edge after shrink", () => {
  const sel = selectPositions(ROWS);
  assert.equal(sel.some((r) => r.fightIdx === 2), false);
});

test("the measured contradiction fight now abstains entirely", () => {
  // Every fightIdx 0 contract shrinks below the 0.50 floor (Garbrandt's winner
  // lands at p*=0.444), so the fight that produced the both-corners bug on
  // 2026-07-11 yields no position at all rather than a mis-stated one.
  assert.equal(selectPositions(ROWS).filter((r) => r.fightIdx === 0).length, 0);
});

test("picks the highest-Kelly contract among those clearing the floor", () => {
  const sel = selectPositions(ROWS);
  assert.equal(sel.length, 1);
  assert.equal(sel[0].fightIdx, 3);
  assert.equal(sel[0].fighter, "Todorovic");
  assert.ok(Math.abs(sel[0].pStar - 0.506) < 1e-9);
});

test("rejects contracts below the shrunk-probability floor", () => {
  // A 3c longshot: model 13% shrinks to ~8%, far under the 0.50 floor. The forward
  // ledger measured this band at 11.8% actual against a ~30% prediction.
  const longshot = [{ fightIdx: 9, fighter: "Longshot", corner: "blue",
                      marketKind: "win_in_r1", modelP: 0.1296, ask: 0.03, feeAdjBE: 0.030509 }];
  assert.equal(selectPositions(longshot).length, 0);
  // Everything that IS selected must clear the floor.
  for (const r of selectPositions(ROWS)) assert.ok(r.pStar >= 0.50, `p*=${r.pStar}`);
});

test("total stake never exceeds the card budget", () => {
  const sized = sizePositions(selectPositions(ROWS), 64, 0.10);
  const total = sized.reduce((s, r) => s + r.stake, 0);
  assert.ok(total <= 64 * 0.10 + 1e-9, `staked ${total} on a 6.40 budget`);
});

test("budget is a cap not a quota - a weak card deploys less", () => {
  const weak = [{ fightIdx: 0, fighter: "A", corner: "red", marketKind: "winner",
                  modelP: 0.52, ask: 0.50, feeAdjBE: 0.505 }];
  const sized = sizePositions(selectPositions(weak), 1000, 0.10);
  const total = sized.reduce((s, r) => s + r.stake, 0);
  assert.ok(total > 0, "weak-but-playable card produced no position at all");
  assert.ok(total < 1000 * 0.10, `weak card deployed the full budget: ${total}`);
});

test("a strong card is scaled down to exactly the budget", () => {
  // $200 bankroll asks ~$7.70 of Kelly; a 2% budget is $4.00 -> must scale to the cap.
  // (Deliberately not a tiny budget: below $1 the min-stake floor drops the row and
  // the total would be 0, which tests the floor rather than the cap.)
  const sized = sizePositions(selectPositions(ROWS), 200, 0.02);
  const total = sized.reduce((s, r) => s + r.stake, 0);
  assert.ok(Math.abs(total - 200 * 0.02) < 1e-9, `expected cap 4.00, got ${total}`);
});

test("contracts derive from raw ask, not fee-adjusted breakeven", () => {
  const sized = sizePositions(selectPositions(ROWS), 1000, 1.0);
  assert.ok(sized.length > 0);
  for (const r of sized) {
    assert.equal(r.contracts, Math.floor(r.stake / r.ask));
    assert.notEqual(r.contracts, Math.floor(r.stake / r.feeAdjBE));
  }
});

test("drops rows worth less than one dollar after the cap", () => {
  const sized = sizePositions(selectPositions(ROWS), 5, 0.10);
  assert.equal(sized.every((r) => r.stake >= 1 && r.contracts >= 1), true);
});
