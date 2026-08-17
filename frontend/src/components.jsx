import { useState, useEffect, useRef, useMemo } from "react";
import { motion, useReducedMotion } from "motion/react";
import { EASE_REVEAL } from "./charts/motion.js";
import { MARKETS } from "./api/client.js";
import { matchesQuery } from "./lib/filters.js";

// Smoothly interpolates a numeric readout toward its latest value instead of
// snapping — used on Prop Lab's live-adjusted probability readouts. Tracks
// the in-flight animated value (not just the last committed one) so rapid
// re-triggers (e.g. dragging a line stepper) re-target without a snap-back.
export function useCountUp(value, duration = 0.45) {
  const reduced = useReducedMotion();
  const [display, setDisplay] = useState(value);
  const displayRef = useRef(value);
  const rafRef = useRef();
  useEffect(() => {
    if (reduced || typeof value !== "number") { setDisplay(value); displayRef.current = value; return; }
    if (displayRef.current === value) return;
    cancelAnimationFrame(rafRef.current);
    const from = typeof displayRef.current === "number" ? displayRef.current : value;
    const to = value;
    const start = performance.now();
    const dur = duration * 1000;
    const tick = (now) => {
      const t = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = from + (to - from) * eased;
      displayRef.current = v;
      setDisplay(v);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
      else displayRef.current = to;
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value, reduced, duration]);
  return (reduced || typeof value !== "number") ? value : display;
}

// Suffixes that should stay attached to the preceding surname token.
const _SUFFIXES = new Set(["jr", "sr", "ii", "iii", "iv", "v"]);
export function lastName(name) {
  if (!name) return "";
  const parts = name.trim().split(/\s+/);
  if (parts.length <= 1) return name;
  const tail = parts[parts.length - 1].toLowerCase().replace(/\.$/, "");
  if (_SUFFIXES.has(tail) && parts.length >= 2) return parts.slice(-2).join(" ");
  return parts[parts.length - 1];
}

export const TabIcon = {
  duration: <svg viewBox="0 0 20 20"><circle cx="10" cy="11" r="7"/><path d="M10 11V7"/><path d="M7.5 2.5h5"/></svg>,
  finishes: <svg viewBox="0 0 20 20"><path d="M11.5 2L4 11h5.5l-1 7L17 9h-5.5z"/></svg>,
  rounds:   <svg viewBox="0 0 20 20"><rect x="3" y="3" width="6" height="6" rx="1"/><rect x="11" y="3" width="6" height="6" rx="1"/><rect x="3" y="11" width="6" height="6" rx="1"/><rect x="11" y="11" width="6" height="6" rx="1"/></svg>,
  sig:      <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><circle cx="10" cy="10" r="3"/><path d="M10 1v2M10 17v2M1 10h2M17 10h2"/></svg>,
  r1sig:    <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="6.5"/><path d="M10 6v4l2.5 1.5"/></svg>,
  td:       <svg viewBox="0 0 20 20"><path d="M3 7l4-4M3 7h4M3 7v-4"/><path d="M17 13l-4 4M17 13h-4M17 13v4"/><path d="M6 14l8-8"/></svg>,
  bodySig:  <svg viewBox="0 0 20 20"><circle cx="10" cy="4.5" r="2.2"/><path d="M10 6.7v7M6 9h8M7 17l3-3.3L13 17"/></svg>,
  legSig:   <svg viewBox="0 0 20 20"><path d="M8 2v7l-2.5 8M8 9l3.5 1.5L14 17"/><circle cx="8" cy="9" r="0.6" fill="currentColor" stroke="none"/></svg>,
  combo:    <svg viewBox="0 0 20 20"><circle cx="7" cy="10" r="4"/><circle cx="13" cy="10" r="4"/></svg>,
  r1td:     <svg viewBox="0 0 20 20"><path d="M3 7l4-4M3 7h4M3 7v-4"/><path d="M6 14l8-8"/><text x="12" y="17" fontSize="8" fill="currentColor" stroke="none">1</text></svg>,
  subAtt:   <svg viewBox="0 0 20 20"><path d="M7 3a4 4 0 0 1 0 8H5"/><path d="M13 17a4 4 0 0 1 0-8h2"/></svg>,
  ctrl:     <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><path d="M10 6v4l3 2"/></svg>,
  kd:       <svg viewBox="0 0 20 20"><path d="M10 2v5M10 13v5M2 10h5M13 10h5M5 5l3 3M15 5l-3 3M5 15l3-3M15 15l-3-3"/></svg>,
  best:     <svg viewBox="0 0 20 20"><path d="M10 2l2.2 4.6L17 7.3l-3.5 3.4.8 4.9L10 13.3 5.7 15.6l.8-4.9L3 7.3l4.8-.7z"/></svg>,
  portfolio:<svg viewBox="0 0 20 20"><path d="M10 3l7 4-7 4-7-4z"/><path d="M3 11l7 4 7-4"/></svg>,
};

export const SrcIcon = {
  refresh:  <svg viewBox="0 0 20 20"><path d="M16 6a7 7 0 1 0 1.5 4"/><path d="M16 2v4h-4"/></svg>,
  bolt:     <svg viewBox="0 0 20 20"><path d="M11 2 4 11h5l-1 7 7-9h-5z"/></svg>,
  download: <svg viewBox="0 0 20 20"><path d="M10 3v9M6 9l4 4 4-4M4 16h12"/></svg>,
};

// "12s ago" / "3m ago" / "2h ago" from a fetchedAt epoch-seconds timestamp.
// Shared by every "updated ___ ago" readout next to a data-source refresh button.
export function agoStr(epochSec) {
  const s = Math.max(0, Math.round(Date.now() / 1000 - epochSec));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}

export function EdgePill({ edge, label }) {
  const pos = edge >= 0;
  return (
    <span className="fp-fchip-edge">
      <span style={{
        fontFamily: "var(--f-mono)", fontSize: 9, fontWeight: 600, letterSpacing: ".5px",
        padding: "3px 6px", borderRadius: 5, whiteSpace: "nowrap",
        color: pos ? "var(--pos)" : "var(--text-faint)",
        background: pos ? "rgba(70,188,132,.12)" : "var(--panel-3)",
      }}>{pos ? "+" : ""}{(edge * 100).toFixed(0)}%{label ? " " + label : ""}</span>
    </span>
  );
}

export function FightChip({ fight, selected, onSelect }) {
  const aFav = fight.a.pWin >= fight.b.pWin;
  const aLast = lastName(fight.a.name);
  const bLast = lastName(fight.b.name);
  return (
    <div className={"fp-fchip" + (selected ? " sel" : "")} onClick={() => onSelect(fight.id)}>
      <div className="fp-fchip-top">
        <span className={"fp-slot" + (fight.isTitle ? " title" : "")}>{fight.slot}</span>
        <span className="fp-fchip-wc">{fight.rounds}R</span>
        {fight.lowData && <span className="fp-chip-lowdata" title="Limited UFC data — prediction shrunk toward 50/50">LOW DATA</span>}
      </div>
      <div className="fp-fchip-names">
        <div className="fp-fchip-row">
          <span className={"fp-fchip-name" + (aFav ? " fav" : "")}>{aLast}</span>
          <span className={"fp-fchip-p" + (aFav ? " fav" : "")}>{(fight.a.pWin * 100).toFixed(0)}%</span>
        </div>
        <div className="fp-fchip-row">
          <span className={"fp-fchip-name" + (!aFav ? " fav" : "")}>{bLast}</span>
          <span className={"fp-fchip-p" + (!aFav ? " fav" : "")}>{(fight.b.pWin * 100).toFixed(0)}%</span>
        </div>
      </div>
      <div className="fp-fchip-split">
        <i style={{ width: fight.a.pWin * 100 + "%", background: aFav ? "var(--gold-br)" : "var(--other)" }}></i>
        <i style={{ width: fight.b.pWin * 100 + "%", background: !aFav ? "var(--gold-br)" : "var(--other)" }}></i>
      </div>
    </div>
  );
}

export function UnavailableChip({ fight }) {
  const aLast = lastName(fight.red);
  const bLast = lastName(fight.blue);
  return (
    <div className="fp-fchip unavail" aria-disabled="true"
         title={`Not enough fighter history to price this bout — ${fight.reason}`}>
      <div className="fp-fchip-top">
        <span className={"fp-slot" + (fight.isTitle ? " title" : "")}>{fight.slot}</span>
        <span className="fp-fchip-wc">{fight.rounds}R</span>
        <span className="fp-chip-nodata">NO DATA</span>
      </div>
      <div className="fp-fchip-names">
        <div className="fp-fchip-row">
          <span className="fp-fchip-name">{aLast}</span>
          <span className="fp-fchip-p">&mdash;</span>
        </div>
        <div className="fp-fchip-row">
          <span className="fp-fchip-name">{bLast}</span>
          <span className="fp-fchip-p">&mdash;</span>
        </div>
      </div>
      <div className="fp-fchip-split nodata"></div>
    </div>
  );
}

// marketRows: resolved Kalshi winner-market rows for this fight ({corner, ask, modelP, ...}).
// Each fighter's own row shows its own corner's quote — never a fallback to the
// other corner's — so there is no attribution to get wrong. When a corner has no
// resolved quote of its own but the OTHER corner does, we show an implied price
// (100¢ minus the other corner's ask), clearly labeled "IMPLIED" so it's never
// mistaken for a real Kalshi quote on this fighter. If neither corner resolved,
// nothing renders (empty state is invisible, not a placeholder).
export function WinSplit({ a, b, marketRows, marketLinesLoading }) {
  const reduced = useReducedMotion();
  const aFav = a.pWin >= b.pWin;
  const aLast = lastName(a.name);
  const bLast = lastName(b.name);
  const fighters = [
    { f: a, last: aLast, color: aFav  ? "var(--gold-br)" : "var(--other)", corner: "red" },
    { f: b, last: bLast, color: !aFav ? "var(--gold-br)" : "var(--other)", corner: "blue" },
  ];
  return (
    <div className="fp-method">
      {fighters.map(({ f, last, color, corner }) => {
        const mkt = marketRows?.find((r) => r.corner === corner) || null;
        const otherMkt = !mkt
          ? marketRows?.find((r) => r.corner !== corner) || null
          : null;
        const impliedAsk = otherMkt ? 1 - otherMkt.ask : null;
        return (
          <div className="fp-mrow" key={last}>
            <div className="fp-mrow-hd">
              <span className="fp-mrow-lbl"><i style={{ background: color }}></i>{last}</span>
              <span className="fp-mrow-right">
                <span className="fp-mrow-val">{(f.pWin * 100).toFixed(1)}%</span>
                {mkt && (
                  <span className="fp-mrow-mkt" title={`Kalshi ask ${Math.round(mkt.ask * 100)}¢ vs model ${(mkt.modelP * 100).toFixed(0)}%`}>
                    <span className="fp-gap-bar">
                      <span className="fp-gap-bar-fill" style={{
                        left: Math.min(mkt.modelP, mkt.ask) * 100 + "%",
                        width: Math.max(Math.abs(mkt.modelP - mkt.ask) * 100, 1.5) + "%" }} />
                    </span>
                    KALSHI {Math.round(mkt.ask * 100)}¢
                  </span>
                )}
                {!mkt && otherMkt && (
                  <span
                    className="fp-mrow-mkt fp-mrow-mkt--implied"
                    title={`No direct Kalshi quote resolved for ${last} — ${Math.round(impliedAsk * 100)}¢ is implied as 100¢ minus the other corner's ${Math.round(otherMkt.ask * 100)}¢ ask, not a tradeable price for this fighter`}
                  >
                    ~{Math.round(impliedAsk * 100)}¢ IMPLIED
                  </span>
                )}
                {!mkt && !otherMkt && marketLinesLoading && (
                  <span className="fp-mrow-mkt fp-mrow-mkt--pending">KALSHI PRICING…</span>
                )}
              </span>
            </div>
            <div className="fp-mrow-track">
              <motion.div
                className="fp-mrow-fill"
                style={{ background: color, transition: "none" }}
                initial={{ width: reduced ? f.pWin * 100 + "%" : 0 }}
                animate={{ width: f.pWin * 100 + "%" }}
                transition={{ duration: reduced ? 0 : 0.6, ease: EASE_REVEAL }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

const TRUST_META = {
  TRUST: { cls: "trust", label: "Trusted", title: "Edge proven on the eval test set — model_prob beat break-even at a statistically confirmed rate." },
  WATCH: { cls: "watch", label: "Watch",   title: "Real signal, but the edge is thin or unproven — treat picks as lower-confidence." },
  CUT:   { cls: "cut",   label: "Low signal", title: "No resolution on the eval test set (AUC ~0.50) — edges here are likely noise." },
};

function _renderTab(t, active, onChange) {
  const tm = TRUST_META[t.trust];
  return (
    <button key={t.key} role="tab" aria-selected={active === t.key}
      className={"fp-tab" + (active === t.key ? " on" : "")}
      style={{ "--tabacc": t.accent }}
      onClick={() => onChange(t.key)}>
      <span className="fp-tab-ic">{TabIcon[t.icon]}</span>{t.label}
      {tm && <span className={"fp-tab-trust " + tm.cls} title={tm.title}>{tm.label}</span>}
    </button>
  );
}

// When tabs carry a `group` field, cluster them under a small caption label
// per group (e.g. "STRIKING", "GRAPPLING") — purely visual, same buttons/handlers.
export function MarketTabs({ tabs, active, onChange }) {
  const hasGroups = tabs.some((t) => t.group);
  if (!hasGroups) {
    return (
      <div className="fp-tabs" role="tablist">
        {tabs.map((t) => _renderTab(t, active, onChange))}
      </div>
    );
  }
  const groups = [];
  const byGroup = new Map();
  tabs.forEach((t) => {
    const g = t.group || "";
    if (!byGroup.has(g)) { byGroup.set(g, []); groups.push(g); }
    byGroup.get(g).push(t);
  });
  return (
    <div className="fp-tabs fp-tabs-grouped" role="tablist">
      {groups.map((g) => (
        <div className="fp-mtab-group" key={g || "_"}>
          {g && <span className="fp-mtab-group-lbl">{g}</span>}
          <div className="fp-mtab-group-tabs">
            {byGroup.get(g).map((t) => _renderTab(t, active, onChange))}
          </div>
        </div>
      ))}
    </div>
  );
}

export function LineStepper({ value, onChange, step = 0.5, placeholder = "set line", width = 180, min = 0, max = Infinity }) {
  const has = value != null && value !== "";
  const clamp = (v) => Math.min(max, Math.max(min, v));
  return (
    <div className="fp-lineinput" style={{ width }}>
      <button className="fp-li-btn" aria-label="Decrease" onClick={() => onChange(clamp((+value || min) - step))}>−</button>
      <input type="number" step={step} value={has ? value : ""} placeholder={placeholder}
        onChange={(e) => onChange(e.target.value === "" ? null : clamp(+e.target.value))} />
      <button className="fp-li-btn" aria-label="Increase" onClick={() => onChange(clamp((+value || min) + step))}>+</button>
    </div>
  );
}

/** Numeric field you can actually clear before typing a new value.
 *
 *  The naive handler (`isNaN(v) ? 0 : v`) writes 0 back the instant the box is
 *  emptied, so the 0 reappears under the cursor and you can never type a fresh
 *  number — you have to select-all and overwrite. This keeps the in-progress text
 *  in local state and only commits parseable values upstream, snapping the display
 *  back to the committed value on blur. Same contract as LineStepper's empty->null.
 */
export function NumField({ value, onChange, disabled, ariaLabel, prefix, suffix,
                           min = 0, max = 100, step = 0.5, width = 84, decimals = 1 }) {
  const [draft, setDraft] = useState(null);   // non-null only while editing
  const committed = value == null ? "" : +Number(value).toFixed(decimals);
  const shown = disabled ? "" : (draft != null ? draft : committed);
  const unitStyle = { padding: "0 8px", color: "var(--text-dim)",
                      fontFamily: "var(--f-mono)", fontSize: 12 };
  return (
    <div className="fp-stepper" style={{ width, opacity: disabled ? 0.4 : 1 }}>
      {prefix && <span style={unitStyle}>{prefix}</span>}
      <input
        type="number" min={min} max={max} step={step} inputMode="decimal"
        aria-label={ariaLabel} disabled={disabled} value={shown}
        onChange={(e) => {
          const raw = e.target.value;
          setDraft(raw);                       // let the box be empty while typing
          const v = parseFloat(raw);
          if (!Number.isNaN(v)) onChange(Math.min(max, Math.max(min, v)));
        }}
        onBlur={() => setDraft(null)}
      />
      {suffix && <span style={unitStyle}>{suffix}</span>}
    </div>
  );
}

export function SegToggle({ value, options, onChange, dim }) {
  return (
    <div className="fp-toggle">
      {options.map((o) => (
        <button key={o.value} className={"fp-toggle-btn" + (value === o.value ? " on" : "") + (dim ? " dim" : "")}
          onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

/** Shared fight-group header — a broadcast "lower-third" band used by both the
 *  Kalshi and DFS lanes so a fight boundary reads identically in either view. */
export function FightGroupHeader({ colSpan, redName, blueName, lowData, lowDataTitle, meta }) {
  return (
    <tr className="fp-fight-hd">
      <td colSpan={colSpan}>
        <div className="fp-fight-hd-inner">
          <span className="fp-fight-hd-names">
            <em className="r">{redName}</em>
            {blueName ? <><i>vs</i><em className="b">{blueName}</em></> : null}
            {lowData && <span className="fp-chip-lowdata" title={lowDataTitle}>LOW DATA</span>}
          </span>
          {meta && <span className="fp-fight-hd-meta">{meta}</span>}
        </div>
      </td>
    </tr>
  );
}

export function SideProbPair({ pOver, breakeven, side }) {
  const pOverAnim = useCountUp(pOver);
  const rows = [
    { key: "over",  label: "OVER",  p: pOverAnim },
    { key: "under", label: "UNDER", p: 1 - pOverAnim },
  ];
  return (
    <div className="fp-sidepair">
      {rows.map((r) => {
        const edge = r.p - breakeven;
        return (
          <div key={r.key} className={"fp-sideprob" + (side === r.key ? " on" : "")}>
            <span className="fp-sideprob-lbl">{r.label}</span>
            <span className="fp-sideprob-val">{(r.p * 100).toFixed(1)}%</span>
            <span className={"fp-sideprob-edge " + (edge >= 0 ? "pos" : "neg")}>
              {edge >= 0 ? "+" : ""}{(edge * 100).toFixed(1)}% vs break-even
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function QuantileTable({ rows }) {
  return (
    <div className="fp-qtable">
      {rows.map((r) => (
        <div key={r.label} className={"fp-qrow" + (r.hl ? " hl" : "")}>
          <span className="fp-qrow-lbl">{r.label}</span>
          <span className="fp-qrow-val">{r.value}</span>
        </div>
      ))}
    </div>
  );
}

export function AddToPortfolio({ added, onClick, label }) {
  return (
    <button className={"fp-addbtn" + (added ? " added" : "")} onClick={onClick}>
      {added ? (
        <><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10l4 4 8-9"/></svg>In portfolio · remove</>
      ) : (
        <><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 4v12M4 10h12"/></svg>{label || "Add leg to portfolio"}</>
      )}
    </button>
  );
}

export function buildCountLeg(fight, fighterName, fighterLast, marketKey, line, side, modelP, accent) {
  return {
    key: fight.id + ":" + fighterLast + ":" + marketKey + ":" + side + ":" + line,
    fightId: fight.id, accent, modelP,
    label: fighterLast + " · " + (MARKETS[marketKey]?.label || marketKey) + " " + side + " " + line,
    sub: (lastName(fight.a.name)) + " v " + (lastName(fight.b.name)),
  };
}

export function buildDurLeg(fight, line, side, modelP) {
  const aL = lastName(fight.a.name);
  const bL = lastName(fight.b.name);
  return {
    key: fight.id + ":dur:" + side + ":" + line,
    fightId: fight.id, accent: "var(--m-dur)", modelP,
    label: "Duration " + side + " " + line + " min",
    sub: aL + " v " + bL,
  };
}

export function buildRoundsLeg(fight, rdLine, side, modelP) {
  const aL = lastName(fight.a.name);
  const bL = lastName(fight.b.name);
  return {
    key: fight.id + ":rounds:" + side + ":" + rdLine,
    fightId: fight.id, accent: "var(--m-rnd)", modelP,
    label: "Fight length " + side + " " + rdLine + " rds",
    sub: aL + " v " + bL,
  };
}

export function buildInsideLeg(fight, modelP) {
  const aL = lastName(fight.a.name);
  const bL = lastName(fight.b.name);
  return {
    key: fight.id + ":inside",
    fightId: fight.id, accent: "var(--m-rnd)", modelP,
    label: "Ends inside distance",
    sub: aL + " v " + bL,
  };
}

export function buildFinishLeg(fight, fighterLast, mktKey, modelP) {
  const aL = lastName(fight.a.name);
  const bL = lastName(fight.b.name);
  const labels = { finish: "any finish", ko_finish: "KO/TKO", sub_finish: "submission" };
  return {
    key: fight.id + ":" + fighterLast + ":finish:" + mktKey,
    fightId: fight.id, accent: "var(--m-r1)", modelP,
    label: fighterLast + " · " + (labels[mktKey] || mktKey),
    sub: aL + " v " + bL,
  };
}

export function buildFinishRndLeg(fight, fighterLast, round, modelP) {
  const aL = lastName(fight.a.name);
  const bL = lastName(fight.b.name);
  return {
    key: fight.id + ":" + fighterLast + ":finish:r" + round,
    fightId: fight.id, accent: "var(--m-r1)", modelP,
    label: fighterLast + " · R" + round + " finish",
    sub: aL + " v " + bL,
  };
}

// Type-to-filter combobox: stacks fighter/market tokens (OR within a kind, AND
// across kinds). Shared by the Positions Kalshi + Books/DFS lanes so filter
// state and UI stay identical across both.
export function FilterCombobox({ tokens, onChange, fighterOptions, marketOptions, placeholder }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const wrapRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    function onDocDown(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocDown);
    return () => document.removeEventListener("mousedown", onDocDown);
  }, []);

  const selectedKeys = useMemo(() => new Set(tokens.map((t) => t.kind + ":" + t.value)), [tokens]);
  const fMatches = useMemo(() =>
    fighterOptions.filter((o) => !selectedKeys.has("fighter:" + o.value) && matchesQuery(o, query)),
    [fighterOptions, selectedKeys, query]);
  const mMatches = useMemo(() =>
    marketOptions.filter((o) => !selectedKeys.has("market:" + o.value) && matchesQuery(o, query)),
    [marketOptions, selectedKeys, query]);
  const flat = [...fMatches, ...mMatches];

  function addToken(opt) {
    onChange([...tokens, { kind: opt.kind, value: opt.value, label: opt.label }]);
    setQuery("");
    setHi(0);
    inputRef.current?.focus();
  }
  function removeToken(i) {
    onChange(tokens.filter((_, idx) => idx !== i));
  }
  function onKeyDown(e) {
    if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); setHi((h) => Math.min(h + 1, flat.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setHi((h) => Math.max(h - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); if (flat[hi]) { addToken(flat[hi]); setOpen(false); } }
    else if (e.key === "Escape") { setOpen(false); }
    else if (e.key === "Backspace" && query === "" && tokens.length) { removeToken(tokens.length - 1); }
  }

  return (
    <div className="fp-filter" ref={wrapRef}>
      <div className="fp-filter-box"
        onClick={() => { inputRef.current?.focus(); setOpen(true); }}>
        {tokens.map((t, i) => (
          <span key={t.kind + ":" + t.value} className={"fp-filter-chip " + t.kind}>
            {t.kind === "fighter" ? lastName(t.label) : t.label}
            <button type="button" aria-label={`Remove ${t.label} filter`}
              onClick={(e) => { e.stopPropagation(); removeToken(i); }}>×</button>
          </span>
        ))}
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-label="Filter by fighter or market"
          value={query}
          placeholder={tokens.length ? "" : (placeholder || "Filter by fighter or market…")}
          onFocus={() => setOpen(true)}
          onChange={(e) => { setQuery(e.target.value); setOpen(true); setHi(0); }}
          onKeyDown={onKeyDown}
        />
      </div>
      {open && (fMatches.length > 0 || mMatches.length > 0) && (
        <div className="fp-filter-menu" role="listbox">
          {fMatches.length > 0 && (
            <div className="fp-filter-section">
              <span className="fp-filter-section-lbl">Fighters</span>
              {fMatches.map((o, i) => (
                <div key={"f:" + o.value} role="option" aria-selected={hi === i}
                  className={"fp-filter-opt" + (hi === i ? " hi" : "")}
                  onMouseDown={(e) => { e.preventDefault(); addToken(o); }}>
                  {o.label}
                </div>
              ))}
            </div>
          )}
          {mMatches.length > 0 && (
            <div className="fp-filter-section">
              <span className="fp-filter-section-lbl">Markets</span>
              {mMatches.map((o, i) => {
                const idx = fMatches.length + i;
                return (
                  <div key={"m:" + o.value} role="option" aria-selected={hi === idx}
                    className={"fp-filter-opt" + (hi === idx ? " hi" : "")}
                    onMouseDown={(e) => { e.preventDefault(); addToken(o); }}>
                    <i className="fp-filter-dot" style={{ background: o.accent }} />{o.label}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
