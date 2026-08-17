import { MARKETS } from "../api/client.js";

// Fighter-or-market filter tokens shared by the Kalshi and Books/DFS lanes on
// the Positions page. Tokens of the same kind OR together; different kinds
// AND together — e.g. [Holloway]+[Takedowns] = Holloway's takedown lines only.
const _ROUND_RE = /^(end_before|win_in)_r(\d+)$/;

export const FILTER_FAMILIES = [
  { value: "winner",   label: "Winner",           kalshi: ["winner"], dfs: [] },
  { value: "ko",       label: "KO/TKO",           kalshi: ["method_ko", "mof_ko"], dfs: ["ko_finish", "r1_ko", "r2_ko", "r3_ko", "r4_ko", "r5_ko"] },
  { value: "sub",      label: "Submission",       kalshi: ["method_sub", "mof_sub"], dfs: ["sub_finish"] },
  { value: "decision", label: "Decision/Distance", kalshi: ["method_dec", "mof_dec", "distance", "vicround_other"], dfs: [] },
  { value: "rounds",   label: "Rounds",           kalshi: "round", dfs: ["rounds"] },
  { value: "duration", label: "Duration",         kalshi: [], dfs: ["duration"] },
  { value: "finish",   label: "Any Finish",       kalshi: [], dfs: ["finish", "r1_finish", "r2_finish", "r3_finish", "r4_finish", "r5_finish"] },
  { value: "sig",      label: "Sig Strikes",      kalshi: [], dfs: ["sig"] },
  { value: "r1sig",    label: "R1 Strikes",       kalshi: [], dfs: ["r1sig"] },
  { value: "bodySig",  label: "Body Strikes",     kalshi: [], dfs: ["bodySig"] },
  { value: "legSig",   label: "Leg Strikes",      kalshi: [], dfs: ["legSig"] },
  { value: "combo",    label: "Combined Strikes", kalshi: [], dfs: ["combo"] },
  { value: "td",       label: "Takedowns",        kalshi: [], dfs: ["td"] },
  { value: "r1td",     label: "R1 Takedowns",     kalshi: [], dfs: ["r1td"] },
  { value: "subAtt",   label: "Sub Attempts",     kalshi: [], dfs: ["subAtt"] },
  { value: "ctrl",     label: "Control Time",     kalshi: [], dfs: ["ctrl"] },
  { value: "kd",       label: "Knockdowns",       kalshi: [], dfs: ["kd"] },
];

const _FAMILY_BY_VALUE = new Map(FILTER_FAMILIES.map((f) => [f.value, f]));

function _norm(s) {
  return (s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

export function fighterOptionsFromFights(fights) {
  const seen = new Set();
  const out = [];
  (fights || []).forEach((f) => {
    [f.a, f.b].forEach((fighter) => {
      if (!fighter?.name || seen.has(fighter.name)) return;
      seen.add(fighter.name);
      out.push({ kind: "fighter", value: fighter.name, label: fighter.name });
    });
  });
  return out;
}

export function marketOptions() {
  return FILTER_FAMILIES.map((f) => ({
    kind: "market", value: f.value, label: f.label,
    accent: MARKETS[f.value]?.accent || "var(--m-exch)",
  }));
}

function _kalshiMarketMatch(marketKind, familyValue) {
  const fam = _FAMILY_BY_VALUE.get(familyValue);
  if (!fam) return false;
  if (fam.kalshi === "round") return _ROUND_RE.test(marketKind);
  return fam.kalshi.includes(marketKind);
}

function _dfsMarketMatch(market, familyValue) {
  const fam = _FAMILY_BY_VALUE.get(familyValue);
  return !!fam && fam.dfs.includes(market);
}

// Kalshi row: a fighter token matches if the row's fight involves that fighter
// (row.fighter is null/undefined for fight-level markets — those pass any
// fighter token for that fight, since e.g. "distance" concerns both fighters).
export function kalshiRowPassesFilter(tokens, row, fight) {
  if (!tokens.length) return true;
  const fighterTokens = tokens.filter((t) => t.kind === "fighter");
  const marketTokens = tokens.filter((t) => t.kind === "market");
  const fighterOk = fighterTokens.length === 0 ||
    fighterTokens.some((t) => fight && (fight.a?.name === t.value || fight.b?.name === t.value));
  const marketOk = marketTokens.length === 0 ||
    marketTokens.some((t) => _kalshiMarketMatch(row.marketKind, t.value));
  return fighterOk && marketOk;
}

// DFS row: fighter token matches the fight it belongs to (row.cardRed/cardBlue
// are the full fight names); market token matches row.market (short key for
// count/fight-level props, canonical key for finish props — see the picks surface).
export function dfsRowPassesFilter(tokens, row) {
  if (!tokens.length) return true;
  const fighterTokens = tokens.filter((t) => t.kind === "fighter");
  const marketTokens = tokens.filter((t) => t.kind === "market");
  const fighterOk = fighterTokens.length === 0 ||
    fighterTokens.some((t) => row.cardRed === t.value || row.cardBlue === t.value);
  const marketOk = marketTokens.length === 0 ||
    marketTokens.some((t) => _dfsMarketMatch(row.market, t.value));
  return fighterOk && marketOk;
}

export function matchesQuery(option, query) {
  const q = _norm(query);
  if (!q) return true;
  return _norm(option.label).includes(q) || _norm(option.value).includes(q);
}
