// Regression check for the fights[] positional-join bug (see frontend/src/lib/fights.js).
// /api/cards drops fights whose predict failed, compacting the array, while
// fightIdx from /api/market-lines is the raw (uncompacted) card index. This
// asserts the OLD `fights[idx]` lookup breaks once a fight is dropped, and the
// NEW `fightByIdx` join stays correct.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { fightByIdx } from "../src/lib/fights.js";

const dir = path.dirname(fileURLToPath(import.meta.url));
const cards = JSON.parse(readFileSync(path.join(dir, "fixtures/cards.json"), "utf8"));
const marketlines = JSON.parse(readFileSync(path.join(dir, "fixtures/marketlines.json"), "utf8"));
const { fights } = cards;
const { rows } = marketlines;

assert.equal(fights.length, 10, "expected 10 fights (fight_7 dropped)");
assert.ok(!fights.some((f) => f.id === "fight_7"), "fight_7 should be dropped from the fixture");

// Old positional join breaks for every fight after the dropped one.
const brokenIdx = [8, 9, 10];
for (const idx of brokenIdx) {
  const oldMatch = fights[idx];
  assert.ok(
    oldMatch === undefined || oldMatch.id !== `fight_${idx}`,
    `expected OLD positional join for fightIdx ${idx} to be wrong-or-undefined, got ${oldMatch?.id}`
  );
}

// New join is correct for every row, regardless of dropped fights.
for (const r of rows) {
  const match = fightByIdx(fights, r.fightIdx);
  assert.ok(match, `fightByIdx found no fight for fightIdx ${r.fightIdx}`);
  assert.equal(match.id, `fight_${r.fightIdx}`, `fightByIdx mismatch for fightIdx ${r.fightIdx}`);
}

// fightIdx 10 (Melisano/Barbosa) is low-data — the default filter must hide it.
const f10 = fightByIdx(fights, 10);
assert.equal(f10.lowData, true, "fightIdx 10 should resolve to a lowData fight");

console.log("OK: fight join check passed (%d fights, %d rows)", fights.length, rows.length);
