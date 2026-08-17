/* API client + static UI config (replaces data.jsx mock). */

// ── Static UI constants ──────────────────────────────────────────────────────

export const PAYOUTS = {
  pp_power_2: { label: "Power Play Power Play 2-pick", platform: "pp", legs: 2, mult: 3.0 },
  pp_power_3: { label: "Power Play Power Play 3-pick", platform: "pp", legs: 3, mult: 5.0 },
  pp_power_4: { label: "Power Play Power Play 4-pick", platform: "pp", legs: 4, mult: 10.0 },
  pp_power_5: { label: "Power Play Power Play 5-pick", platform: "pp", legs: 5, mult: 20.0 },
  ud_2:       { label: "Flat Multi 2-pick", platform: "ud", legs: 2, mult: 3.0 },
  ud_3:       { label: "Flat Multi 3-pick", platform: "ud", legs: 3, mult: 6.0 },
  ud_4:       { label: "Flat Multi 4-pick", platform: "ud", legs: 4, mult: 10.0 },
  ud_5:       { label: "Flat Multi 5-pick", platform: "ud", legs: 5, mult: 20.0 },
};

// Base entry multiplier for a (platform, legCount) standard/power play. Used by the
// Portfolio tab to auto-price from the number of legs ACTUALLY added (picks.length),
// instead of a manually-selected leg count that can desync. Per-line goblin/demon/boost
// modifiers are folded on top of this base by the caller. Returns null on table miss
// (caller falls back to the selected payout multiplier).
export function baseMultFor(platform, legCount) {
  for (const p of Object.values(PAYOUTS))
    if (p.platform === platform && p.legs === legCount) return p.mult;
  return null;
}

// trust: advisory reliability tier from configs/prop_trust.yaml (last scorecard
// 2026-06-22, scripts/05c_evaluate_prop_edge.py) — TRUST = edge proven on the eval
// test set, WATCH = real resolution but edge too thin/unproven, CUT = no resolution
// (AUC ~0.50), structurally unpredictable. Hand-maintained: re-sync after 05c reruns.
// rounds inherits duration's tier (RoundsPanel derives its curve directly from
// fight.durCurve — same model, reparametrized). Markets with no 05c/inherited
// coverage (r2-r5 finish) carry no tier.
export const MARKETS = {
  duration:   { label: "Duration",       unit: "min", accent: "var(--m-dur)",  trust: "TRUST" },
  rounds:     { label: "Rounds",         unit: "",    accent: "var(--m-rnd)",  trust: "TRUST" }, // reparametrized duration curve, not separately scored in 05c
  sig:        { label: "Sig Strikes",    unit: "",    accent: "var(--m-sig)",  trust: "TRUST" },
  r1sig:      { label: "R1 Sig Strikes", unit: "",    accent: "var(--m-r1)",   trust: "WATCH" },
  bodySig:    { label: "Body Strikes",   unit: "",    accent: "var(--m-body)", trust: "TRUST" }, // proven @4.5 line only
  legSig:     { label: "Leg Strikes",    unit: "",    accent: "var(--m-leg)",  trust: "TRUST" },
  combo:      { label: "Combined Strikes", unit: "",  accent: "var(--m-combo)", trust: "WATCH" },
  td:         { label: "Takedowns",      unit: "",    accent: "var(--m-td)",   trust: "WATCH" },
  r1td:       { label: "R1 Takedowns",   unit: "",    accent: "var(--m-r1td)", trust: "WATCH" },
  subAtt:     { label: "Sub Attempts",   unit: "",    accent: "var(--m-sub)",  trust: "CUT" },
  ctrl:       { label: "Control Time",   unit: "min", accent: "var(--m-ctrl)", trust: "WATCH" },
  kd:         { label: "Knockdowns",     unit: "",    accent: "var(--m-kd)",   trust: "CUT" },
  ko_finish:  { label: "KO Finish",      unit: "",    accent: "var(--neg)",   trust: "WATCH" },
  sub_finish: { label: "Sub Finish",     unit: "",    accent: "var(--m-td)",  trust: "WATCH" },
  finish:     { label: "Any Finish",     unit: "",    accent: "var(--gold-br)", trust: "WATCH" },
  r1_finish:  { label: "R1 Finish",      unit: "",    accent: "var(--m-r1)",  trust: "WATCH" },
  r2_finish:  { label: "R2 Finish",      unit: "",    accent: "var(--m-r1)"  },
  r3_finish:  { label: "R3 Finish",      unit: "",    accent: "var(--m-r1)"  },
  r4_finish:  { label: "R4 Finish",      unit: "",    accent: "var(--m-r1)"  },
  r5_finish:  { label: "R5 Finish",      unit: "",    accent: "var(--m-r1)"  },
};

export const N_MC = 50000;

// ── Fetch helpers ────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function apiFetch(path, opts = {}) {
  const res = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// ── Card endpoints ───────────────────────────────────────────────────────────

/** List scraped upcoming cards: [{id, label, event_date, n_fights}] */
export function getCards() {
  return apiFetch("/api/cards");
}

/** Re-scrape upcoming cards. Returns updated list. */
export function refreshCards() {
  return apiFetch("/api/cards/refresh", { method: "POST" });
}

/**
 * Get all fights for a card, serialized to the design data shape.
 * Returns { event, fights } where event = {code, name, venue, date}
 * and fights = array of FightOut objects.
 */
export function getCard(cardId) {
  return apiFetch(`/api/cards/${cardId}`);
}

// ── Predict (manual form) ────────────────────────────────────────────────────

/**
 * Run prediction for a manually specified matchup.
 * form = { red, blue, rounds, isTitle, eventDate, weightClass, referee, location }
 * Returns single fight-shape object.
 */
export function predictManual(form) {
  return apiFetch("/api/predict", {
    method: "POST",
    body: JSON.stringify(form),
  });
}

// ── Reference data ───────────────────────────────────────────────────────────

/** Fighter name list for autocomplete. */
export function getFighters() {
  return apiFetch("/api/fighters");
}

/** Known referee names. */
export function getReferees() {
  return apiFetch("/api/referees");
}

/** Valid weight class names. */
export function getWeightClasses() {
  return apiFetch("/api/weight-classes");
}

/**
 * Prop trust tiers keyed by frontend market key.
 * Returns { sig: "WATCH", td: "CUT", ... }
 * TRUST = real edge proven; WATCH = uncertain; CUT = no resolution / unpredictable.
 */
export function getPropTrust() {
  return apiFetch("/api/prop-trust");
}

// ── Portfolio ────────────────────────────────────────────────────────────────

/**
 * Grade a portfolio with MC correlation.
 * legs = [{fightId, modelP, label}]
 * Returns { naiveJoint, mcJoint, correlationAdj, ev, breakeven, kelly, verdict }
 */
export function gradePortfolio(legs, { payoutKey = "pp_power_2", mult = 3.0 } = {}) {
  return apiFetch("/api/portfolio", {
    method: "POST",
    body: JSON.stringify({ legs, payout_key: payoutKey, mult }),
  });
}

// ── History ──────────────────────────────────────────────────────────────────

/**
 * Get historical per-fight predicted-vs-actual data.
 * Returns array of events: [{id, event, date, correct, total, hitRate, fights:[...]}]
 */
export function getHistory() {
  return apiFetch("/api/history");
}

// ── Exchange (Kalshi) market lines ──────────────────────────────────────────

/**
 * Server-priced Kalshi winner/method quotes for a card (winner + method markets
 * only — round/duration are never advised). Priced backend-side against the same
 * predictions /api/cards serves; the frontend renders rows as-is, never reprices.
 * Returns { rows: [...], errors: [...] }. Each row's `paper` flag reflects
 * configs/market_advice.yaml; empty rows (no open Kalshi markets yet) is normal.
 */
export function getMarketLines(cardId, { fresh = false } = {}) {
  return apiFetch(`/api/market-lines/${cardId}${fresh ? "?fresh=1" : ""}`);
}

// ── Ledger summary (Model vs Market) ────────────────────────────────────────

/**
 * Read-only aggregation of the graded prop ledger: edge buckets, per-market
 * hit rate, and closing-line-value performance. Returns {available:false} when
 * no resolved prop rows exist yet (e.g. a fresh deploy).
 */
export function getLedgerSummary() {
  return apiFetch("/api/ledger_summary");
}

// ── Meta ─────────────────────────────────────────────────────────────────────

/**
 * Get model metadata.
 * Returns { version, lastSync, cardsAnalyzed, hitRate, roi, units, calib }
 */
export function getMeta() {
  return apiFetch("/api/meta");
}

// ── Export ───────────────────────────────────────────────────────────────────

export function exportCardCsv(cardId) {
  window.open(`/api/export/card/${cardId}.csv`);
}
