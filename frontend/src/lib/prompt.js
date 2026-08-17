const pct = (v) => (v == null ? "N/A" : Math.round(v * 100) + "%");
const fix1 = (v) => (v == null ? "N/A" : v.toFixed(1));

// Only include a prop line if it adds narrative signal (non-trivial median + not almost always zero)
function propLine(label, prop) {
  if (!prop?.summary?.q) return null;
  const p50 = prop.summary.q.p50;
  if (p50 == null) return null;
  const pz = prop.pZero ?? 0;
  // Skip if median is negligible AND almost always zero — not worth mentioning
  if (p50 < 0.3 && pz > 0.8) return null;
  const zeroNote = pz > 0.4 ? ` (${pct(pz)} chance of zero)` : "";
  return `  ${label}: ~${fix1(p50)}${zeroNote}`;
}

// Core narrative props only — volume, wrestling, danger signals
const CORE_PROPS = [
  ["sig",    "Significant strikes"],
  ["td",     "Takedowns"],
  ["ctrl",   "Ground control time (minutes)"],
  ["kd",     "Knockdowns"],
  ["subAtt", "Submission attempts"],
];

function fighterBlock(f, fightMethodKo, fightMethodSub) {
  const lines = [`${f.name} — win probability: ${pct(f.pWin)}`];

  for (const [key, label] of CORE_PROPS) {
    const line = propLine(label, f[key]);
    if (line) lines.push(line);
  }

  // Directional finish — make clear these are "this fighter wins by X", not fight-level method
  const fin = f.finish || {};
  if (fin.ko_finish != null)
    lines.push(`  Chance ${f.name} specifically wins by KO/TKO: ${pct(fin.ko_finish)}`);
  if (fin.sub_finish != null)
    lines.push(`  Chance ${f.name} specifically wins by submission: ${pct(fin.sub_finish)}`);

  return lines.join("\n");
}

// Model's own win-probability attribution → plain-language "where the edges are".
// pp magnitudes are dropped (meaningless to a general chatbot, invite false
// precision); the "ratings" bucket is omitted (a black-box composite the AI
// can't turn into a fight mechanism without reaching for reputation, which this
// prompt forbids). Placed AFTER the raw per-fighter stats so the AI reasons
// from mechanism first and reads this as the model's summary, not a script.
function edgeLines(fight) {
  const drivers = (fight.winnerDrivers || [])
    .filter((d) => d && d.magnitude != null && d.key !== "ratings");
  if (!drivers.length) return [];
  const strength = (m) => {
    const pp = m * 100;
    return pp >= 5 ? "strong" : pp >= 2.5 ? "moderate" : pp >= 1 ? "slight" : "faint";
  };
  const sorted = [...drivers].sort((a, b) => b.magnitude - a.magnitude);
  const fmt = (side) =>
    sorted
      .filter((d) => d.favors === side)
      .map((d) => `${d.label.toLowerCase()} (${strength(d.magnitude)})`)
      .join(", ");
  const forA = fmt("a");
  const forB = fmt("b");
  const lines = [
    "WHERE THE MODEL SEES THE EDGES (the model's own ranked reasons for its pick — frame the analysis around these; do not invent others):",
  ];
  if (forA) lines.push(`  Favoring ${fight.a?.name || "Fighter A"}: ${forA}`);
  if (forB) lines.push(`  Favoring ${fight.b?.name || "Fighter B"}: ${forB}`);
  return lines.length > 1 ? lines : [];
}

export function buildFightPrompt(fight) {
  if (!fight) return "";

  const a = fight.a;
  const b = fight.b;
  const aName = a?.name || "Fighter A";
  const bName = b?.name || "Fighter B";
  const m = fight.method || {};

  // ── Instruction ──────────────────────────────────────────────────────────
  const instruction = [
    "You are an analytical MMA writer. Below are statistical model probabilities for a UFC bout.",
    "Write a clear, measured fight breakdown for a reader who follows MMA but is not a statistician.",
    "",
    "Your job:",
    "  1. Write 2–3 concise paragraphs covering: where the fight is likely decided",
    "     (striking / grappling / clinch), who holds the structural advantage, how the",
    "     fight evolves across its expected length, and how it most likely ends.",
    "     Describe in phases (early / mid / late) — cite specific rounds only when the",
    "     round finish distribution clearly concentrates probability there.",
    "  2. After the paragraphs, add two clearly labeled lines:",
    "       MOST LIKELY OUTCOME: [one sentence — who wins, how, roughly when]",
    "       SECOND MOST LIKELY: [one sentence — the alternative path]",
    "",
    "Rules:",
    "  - Tone: calm and analytical, not hype. No exclamation points, no TV-anchor energy.",
    "  - Translate numbers into plain language where it helps ('slight favorite', 'even odds',",
    "    'more likely than not') but stay grounded — avoid over-dramatizing.",
    "  - Do NOT add fighter reputation, nicknames, historical context, or biographical traits",
    "    (e.g. 'legendary cardio', 'power puncher', 'elite grappler') unless directly",
    "    supported by the numbers below. Base the narrative ONLY on the data provided.",
    "  - Do NOT reproduce the stats as a list. Integrate them into the analysis.",
    "  - HOW IT ENDS = how the fight ends regardless of who wins.",
    "    FIGHTER WIN-BY = which fighter specifically wins by that method. Do not conflate.",
    "  - ROUND FINISH CHANCES = probability the fight ends IN that specific round, not cumulative.",
  ].join("\n");

  // ── Bout ─────────────────────────────────────────────────────────────────
  const boutParts = [fight.weightClass || "Catchweight", `${fight.rounds || 3} rounds`];
  if (fight.isTitle) boutParts.push("TITLE FIGHT");
  const bout = `FIGHT: ${aName} vs ${bName} (${boutParts.join(", ")})`;

  // ── Winner ───────────────────────────────────────────────────────────────
  const conf = fight.confidence || {};
  const winnerLines = [
    `WHO WINS:`,
    `  ${aName}: ${pct(a?.pWin)} chance`,
    `  ${bName}: ${pct(b?.pWin)} chance`,
  ];
  if (fight.lowData) {
    winnerLines.push(`  NOTE: Limited UFC data — probabilities pulled toward 50/50. Treat with caution.`);
  }

  // ── Method ───────────────────────────────────────────────────────────────
  const methodLines = [
    `HOW IT ENDS (either fighter can cause this):`,
    `  Knockout or TKO: ${pct(m.ko)}`,
    `  Submission: ${pct(m.sub)}`,
    `  Goes to the judges: ${pct(m.dec)}`,
  ];

  // ── Duration + round dist ────────────────────────────────────────────────
  const durLines = [];
  if (fight.medianMin != null) {
    const medRound = Math.ceil((fight.medianMin * 60) / 300);
    durLines.push(`EXPECTED LENGTH: most likely ends around ${fix1(fight.medianMin)} minutes in (~Round ${medRound})`);
  }

  const rd = fight.roundDist || [];
  const rdLines = [];
  if (rd.length) {
    const nRounds = rd.length - 1;
    const parts = rd.slice(0, nRounds).map((p, i) => `R${i + 1}: ${pct(p)}`);
    parts.push(`Decision: ${pct(rd[rd.length - 1])}`);
    rdLines.push(`ROUND FINISH CHANCES (P fight ends IN that round): ${parts.join(" · ")}`);
  }

  // ── R1 finish ────────────────────────────────────────────────────────────
  const r1Lines = fight.r1Finish != null && fight.r1Finish > 0.05
    ? [`EARLY KNOCKOUT RISK: ${pct(fight.r1Finish)} chance someone gets stopped in Round 1`]
    : [];

  // ── Per-fighter breakdown ────────────────────────────────────────────────
  const projLines = [
    `FIGHTER BREAKDOWN (expected outputs for the full fight):`,
    "",
    fighterBlock(a, m.ko, m.sub),
    "",
    fighterBlock(b, m.ko, m.sub),
  ];

  // ── Combined pace (after per-fighter, so AI doesn't double-count) ────────
  const comboP50 = fight.sigCombo?.summary?.q?.p50;
  const paceLines = comboP50 != null
    ? [`COMBINED PACE: ~${fix1(comboP50)} total significant strikes from both fighters combined`]
    : [];

  // ── Model's edge attribution (plain language, after the raw stats) ───────
  const edges = edgeLines(fight);

  // ── Assemble ─────────────────────────────────────────────────────────────
  const sections = [
    instruction,
    "",
    "════════════════════════════════════════",
    "FIGHT DATA",
    "════════════════════════════════════════",
    bout,
    "",
    winnerLines.join("\n"),
    "",
    methodLines.join("\n"),
    "",
    ...(durLines.length ? [...durLines, ""] : []),
    ...(rdLines.length ? [...rdLines, ""] : []),
    ...(r1Lines.length ? [...r1Lines, ""] : []),
    projLines.join("\n"),
    ...(paceLines.length ? ["", ...paceLines] : []),
    ...(edges.length ? ["", ...edges] : []),
    "",
    "════════════════════════════════════════",
    "Now write the fight preview:",
  ];

  return sections.join("\n");
}
