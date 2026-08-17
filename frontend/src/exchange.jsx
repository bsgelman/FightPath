import { useMemo, useState } from "react";
import { lastName, SrcIcon, agoStr, SegToggle } from "./components.jsx";
import { kalshiRowPassesFilter } from "./lib/filters.js";
import { fightByIdx } from "./lib/fights.js";
import { pickScore } from "./lib/stats.js";

// Shared grid so every card's columns line up page-wide.
const ROW_GRID = "minmax(120px,1.2fr) minmax(130px,1.3fr) 64px 56px 92px 52px 74px";

// Kalshi market_kind -> display label / accent color. Decision-family kinds
// (method_dec, mof_dec, distance, vicround_other) share one accent — they're
// the same underlying question ("does this go to a decision?") priced on
// three different venues, which is also why they collapse into one stack row.
const _ROUND_RE = /^(end_before|win_in)_r(\d+)$/;

export function kindMeta(marketKind) {
  switch (marketKind) {
    case "winner":         return { label: "WIN",         accent: "var(--gold-br)" };
    case "method_ko":      return { label: "KO/TKO",      accent: "var(--neg)" };
    case "method_sub":     return { label: "SUB",         accent: "var(--m-td)" };
    case "method_dec":     return { label: "DEC",         accent: "var(--m-port)" };
    case "mof_ko":         return { label: "KO/TKO · ANY", accent: "var(--neg)" };
    case "mof_sub":        return { label: "SUB · ANY",   accent: "var(--m-td)" };
    case "mof_dec":        return { label: "DEC · ANY",   accent: "var(--m-port)" };
    case "distance":       return { label: "DISTANCE",    accent: "var(--m-port)" };
    case "vicround_other": return { label: "DEC/DRAW",    accent: "var(--m-port)" };
    default: {
      const m = _ROUND_RE.exec(marketKind);
      if (m) {
        const r = m[2];
        return m[1] === "end_before"
          ? { label: `ENDS <R${r}`, accent: "var(--m-rnd)", round: Number(r), kind: "end_before" }
          : { label: `WIN R${r}`,   accent: "var(--m-rnd)", round: Number(r), kind: "win_in" };
      }
      return { label: marketKind, accent: "var(--text)" };
    }
  }
}

// Kinds that all price the same underlying question (P(decision)) on three
// different Kalshi venues — collapsed into one expandable stack row per fight.
const _DEC_STACK_KINDS = new Set(["distance", "mof_dec", "vicround_other"]);

/** Round-boundary strip for end_before_r{N} / win_in_r{N} markets — one filled
 *  cell per round the market resolves on. aria-hidden: the text label already
 *  states the same fact ("ENDS <R2", "WIN R1"), this is a glance-able echo. */
function RoundStrip({ round, kind, totalRounds }) {
  const n = totalRounds || 3;
  const cells = [];
  for (let i = 1; i <= n; i++) {
    const on = kind === "win_in" ? i === round : i < round;
    cells.push(<div key={i} className={"fp-round-cell" + (on ? " on" : "")} />);
  }
  const title = kind === "win_in"
    ? `Wins in round ${round} of ${n}`
    : `Fight ends before round ${round} of ${n}`;
  return <div className="fp-round-strip" aria-hidden="true" title={title}>{cells}</div>;
}

// Tooltip for the LOW DATA badge — names the thin fighter(s) (≤3 UFC bouts),
// mirroring the picks surface's _lowDataTitle for the DFS lane.
function _lowDataTitle(fight) {
  const thin = [];
  if (fight.nFightsRed != null && fight.nFightsRed < 4) thin.push(`${lastName(fight.a.name)} (${fight.nFightsRed})`);
  if (fight.nFightsBlue != null && fight.nFightsBlue < 4) thin.push(`${lastName(fight.b.name)} (${fight.nFightsBlue})`);
  return thin.length
    ? `Limited UFC data — ${thin.join(", ")} — prediction shrunk toward 50/50`
    : "Limited UFC data — prediction shrunk toward 50/50";
}

export function _liqBadge(r) {
  if (r.depthUsd == null) {
    return <span className="fp-liq na" title="Depth not fetched for this series — size manually">N/A</span>;
  }
  const liqCls = (r.liqTier || "THIN").toLowerCase();
  const title = liqCls === "thin"
    ? `~$${Math.round(r.depthUsd).toLocaleString()} within 3¢ of the ask — thin resting depth, not a bad edge. Size below the cap rather than skip the pick.`
    : `~$${Math.round(r.depthUsd).toLocaleString()} within 3¢ of the ask`;
  return (
    <span className={"fp-liq " + liqCls} title={title}>
      {r.liqTier || "THIN"}
    </span>
  );
}

function _divergenceBadge(r) {
  if (!r.divergence) return null;
  const gap = Math.abs((r.modelP ?? 0) - (r.ask ?? 0)) * 100;
  return (
    <span
      className="fp-diverge"
      title={`Model and market disagree by ${gap.toFixed(0)}pp. Gaps this large can mean real edge — or information the model can't see (late injury news, a last fight whose result hides a bad performance). Check recent tape before sizing.`}
    >
      ⚠ CHECK TAPE
    </span>
  );
}

/** One market row. `fight` may be undefined for a fight the card list hasn't
 *  loaded (the row still renders, just without the LOW DATA chip / round count).
 *  Liquidity (DEEP/OK/THIN/N/A) is a sizing signal, not a quality signal — it
 *  lives entirely in the badge; the row itself never dims for it, so a thin
 *  book doesn't read as "skip this pick." */
function ExchangeRow({ r, fight, indented, stackToggle }) {
  const meta = kindMeta(r.marketKind);
  const edge = r.edgePct ?? 0;
  const otherSideTitle = r.__otherSide
    ? `Other side: ${lastName(r.__otherSide.fighter)} ${Math.round(r.__otherSide.ask * 100)}¢ (${((r.__otherSide.edgePct ?? 0) * 100).toFixed(1)}pp)`
    : undefined;
  const askTitle = `Net cost incl. Kalshi fee: ${(r.feeAdjBE * 100).toFixed(1)}¢ — the true breakeven`;

  return (
    <div className={"fp-bbrow" + (indented ? " sub" : "")} style={{ gridTemplateColumns: ROW_GRID }}>
      <div className="fp-td-fighter" title={otherSideTitle}>
        {r.fighter
          ? lastName(r.fighter)
          : <span className="fp-mkt-tag fp-tag-fight">FIGHT</span>}
      </div>
      <div>
        {stackToggle ? (
          <button type="button" className="fp-dec-stack-btn"
                  aria-expanded={stackToggle.expanded} onClick={stackToggle.onToggle}>
            <span className="fp-mkt-tag" style={{ color: meta.accent }}>GOES DISTANCE</span>
            <span className="fp-dec-stack-count">▸ {stackToggle.count} venues</span>
            <span className={"fp-dec-stack-chevron" + (stackToggle.expanded ? " open" : "")}>›</span>
          </button>
        ) : (
          <>
            <span className="fp-mkt-tag" style={{ color: meta.accent }}>{meta.label}</span>
            {meta.round != null && <RoundStrip round={meta.round} kind={meta.kind} totalRounds={fight?.rounds} />}
          </>
        )}
      </div>
      <div className="fp-modelp">{(r.modelP * 100).toFixed(1)}%</div>
      <div title={askTitle} style={{ cursor: "help" }}>{Math.round(r.ask * 100)}¢</div>
      <div>
        <span className={"fp-edge-cell " + (edge >= 0 ? "pos" : "neg")}>{edge >= 0 ? "+" : ""}{(edge * 100).toFixed(1)}pp</span>
        {_divergenceBadge(r)}
      </div>
      <div>{(r.kelly * 100).toFixed(0)}%</div>
      <div>{_liqBadge(r)}</div>
    </div>
  );
}

/** Groups rows by fight, collapsing decision-family duplicates (distance /
 *  mof_dec / vicround_other price the same P(decision) on three venues) into
 *  one expandable stack row per fight, and deduplicating the two winner-market
 *  rows (same underlying question priced for each corner) down to the higher-
 *  edge side — the dropped side stays reachable via the fighter-cell tooltip.
 *  Fights are ordered by their own best edge, and rows within a fight stay
 *  edge-sorted. */
function useExchangeGroups(rows, fights, filterTokens) {
  return useMemo(() => {
    const filtered = !filterTokens?.length
      ? rows
      : rows.filter((r) => kalshiRowPassesFilter(filterTokens, r, fightByIdx(fights, r.fightIdx)));
    const byFight = new Map();
    for (const r of filtered) {
      if (!byFight.has(r.fightIdx)) byFight.set(r.fightIdx, []);
      byFight.get(r.fightIdx).push(r);
    }
    const groups = [];
    for (const [fightIdx, fightRows] of byFight) {
      let rowsIn = fightRows;
      const winRows = fightRows.filter((r) => r.marketKind === "winner");
      if (winRows.length > 1) {
        const best = winRows.reduce((a, b) => (pickScore(b) > pickScore(a) ? b : a));
        const other = winRows.find((r) => r !== best);
        rowsIn = fightRows.filter((r) => r.marketKind !== "winner");
        rowsIn.push({ ...best, __otherSide: other });
      }

      const decRows = rowsIn.filter((r) => _DEC_STACK_KINDS.has(r.marketKind));
      const plainRows = rowsIn.filter((r) => !_DEC_STACK_KINDS.has(r.marketKind));

      const display = [...plainRows];
      let stack = null;
      if (decRows.length > 1) {
        const best = decRows.reduce((a, b) => (pickScore(b) > pickScore(a) ? b : a));
        stack = { best, members: decRows };
        display.push({ ...best, __stack: stack });
      } else {
        display.push(...decRows);
      }
      display.sort((a, b) => pickScore(b) - pickScore(a));

      // bestEdge is the raw max edge, still shown in the group header badge unchanged;
      // bestScore is confidence-weighted and drives group ORDER only (ranking, not display).
      const bestEdge = Math.max(...fightRows.map((r) => r.edgePct ?? -1));
      const bestScore = Math.max(...fightRows.map((r) => pickScore(r)));
      groups.push({ fightIdx, rows: display, n: rowsIn.length, bestEdge, bestScore });
    }
    groups.sort((a, b) => b.bestScore - a.bestScore);
    return groups;
  }, [rows, fights, filterTokens]);
}

/** Positions tab, Exchange (Kalshi) section — server-priced quotes across six
 *  market families. Rendered as fight cards; the frontend never reprices these
 *  (see /api/market-lines). */
export function ExchangeSection({ fights, marketLines, loading, onRefresh, showLowData, filterTokens }) {
  const rows = marketLines?.rows || [];
  const errors = marketLines?.errors || [];
  const [minProb, setMinProb] = useState(0.55);
  const [showAllProb, setShowAllProb] = useState(false);
  const probFiltered = rows.filter((r) => showAllProb || (r.modelP ?? -Infinity) >= minProb);
  const hiddenProbN = showAllProb ? 0 : rows.filter((r) => (r.modelP ?? -Infinity) < minProb).length;
  const groups = useExchangeGroups(probFiltered, fights, filterTokens);
  const [expanded, setExpanded] = useState(() => new Set());

  const toggleStack = (fightIdx) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(fightIdx) ? next.delete(fightIdx) : next.add(fightIdx);
      return next;
    });
  };

  const visGroups = showLowData
    ? groups
    : groups.filter((g) => (fights?.length ? !(fightByIdx(fights, g.fightIdx)?.lowData ?? true) : true));
  const hiddenN = groups.length - visGroups.length;

  return (
    <div className="fp-exch">
      <div className="fp-exch-hd">
        <div>
          <span className="fp-exch-ttl">Prediction Markets — Kalshi</span>
          <span className="fp-exch-sub">Six market families, priced against the fee-adjusted ask — never the mid</span>
        </div>
        <div className="fp-exch-actions">
          {marketLines?.fetchedAt && <span className="fp-updated-ago">updated {agoStr(marketLines.fetchedAt)}</span>}
          <button className="fp-btn sm" onClick={onRefresh} disabled={loading || !onRefresh}>
            {SrcIcon.refresh}{loading ? "Refreshing…" : "Refresh quotes"}
          </button>
        </div>
      </div>

      <div className="fp-bb-controls compact">
        <span className="fp-cgroup-lbl">Min confidence</span>
        <div className="fp-stepper" style={{ width: 84, opacity: showAllProb ? 0.4 : 1 }}>
          <input type="number" min="0" max="100" step="0.5" inputMode="decimal"
            aria-label="Minimum confidence percent"
            value={showAllProb ? "" : +(minProb * 100).toFixed(1)}
            disabled={showAllProb}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setMinProb(isNaN(v) ? 0 : Math.min(100, Math.max(0, v)) / 100);
            }} />
          <span style={{ padding: "0 8px", color: "var(--text-dim)", fontFamily: "var(--f-mono)", fontSize: 12 }}>%</span>
        </div>
        <SegToggle value={showAllProb ? "all" : "min"} onChange={(v) => setShowAllProb(v === "all")}
          options={[{ value: "min", label: "Min" }, { value: "all", label: "All" }]} />
      </div>

      {loading ? (
        <div className="fp-exch-empty">Loading Kalshi quotes…</div>
      ) : rows.length === 0 ? (
        <div className="fp-exch-empty">
          {errors.length
            ? errors.join(" · ")
            : "No open Kalshi markets for this card yet — winner lists fight week; method, rounds and distance often later."}
        </div>
      ) : groups.length === 0 ? (
        <div className="fp-exch-empty">
          {`${hiddenProbN} line${hiddenProbN !== 1 ? "s" : ""} hidden by confidence floor — lower Min confidence or switch to All.`}
        </div>
      ) : visGroups.length === 0 ? (
        <div className="fp-exch-empty">
          {filterTokens?.length
            ? "No Kalshi lines match this filter — try removing a token."
            : "All fights on this card are low-data — flip Low data to Show."}
        </div>
      ) : (
        <>
          <div className="fp-bbcards">
            {visGroups.map((g) => {
              const fight = fightByIdx(fights, g.fightIdx);
              const isExpanded = expanded.has(g.fightIdx);
              return (
                <div className="fp-bbcard exch" key={g.fightIdx}>
                  <div className="fp-bbcard-hd">
                    <span className="fp-bbcard-names">
                      <em className="r">{fight ? lastName(fight.a.name) : `Fight ${g.fightIdx + 1}`}</em>
                      {fight && <><i>vs</i><em className="b">{lastName(fight.b.name)}</em></>}
                      {fight?.lowData && <span className="fp-chip-lowdata" title={_lowDataTitle(fight)}>LOW DATA</span>}
                    </span>
                    <span className="fp-bbcard-meta">
                      {g.n} market{g.n !== 1 ? "s" : ""} · best <b className={g.bestEdge >= 0 ? "pos" : "neg"}>{g.bestEdge >= 0 ? "+" : ""}{(g.bestEdge * 100).toFixed(1)}pp</b>
                    </span>
                  </div>
                  <div className="fp-bbcard-colhd" style={{ gridTemplateColumns: ROW_GRID }}>
                    <span>Fighter</span><span>Market</span><span>Model P</span><span>Ask</span><span>Edge</span><span>Kelly</span><span>Liquidity</span>
                  </div>
                  {g.rows.map((r, i) => {
                    if (r.__stack) {
                      const rowsToShow = isExpanded
                        ? [r, ...r.__stack.members.filter((m) => m !== r.__stack.best)]
                        : [r];
                      return rowsToShow.map((mr, j) => (
                        <ExchangeRow
                          key={mr.ticker || `${g.fightIdx}:stack:${j}`}
                          r={mr}
                          fight={fight}
                          indented={j > 0}
                          stackToggle={j === 0 ? {
                            count: r.__stack.members.length,
                            expanded: isExpanded,
                            onToggle: () => toggleStack(g.fightIdx),
                          } : undefined}
                        />
                      ));
                    }
                    return <ExchangeRow key={r.ticker || `${g.fightIdx}:${r.marketKind}:${r.fighter}:${i}`} r={r} fight={fight} />;
                  })}
                </div>
              );
            })}
          </div>
          <span className="fp-bb-hint">
            Liquidity is depth within 3¢ of the ask, in yes-equivalent dollars — a sizing guide, not a fill guarantee.
            {hiddenN > 0 && ` · ${hiddenN} low-data fight${hiddenN !== 1 ? "s" : ""} hidden`}
            {hiddenProbN > 0 && ` · ${hiddenProbN} line${hiddenProbN !== 1 ? "s" : ""} hidden by confidence floor`}
          </span>
        </>
      )}
    </div>
  );
}
