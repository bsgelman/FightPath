// /api/cards compacts out fights whose predict failed, so positional lookup by
// the server's raw-card fightIdx is wrong after any dropped fight. Fight ids
// carry the raw index ("fight_10") — join on that.
export const fightByIdx = (fights, idx) => fights?.find((f) => f.id === `fight_${idx}`);
