import { useState, useMemo, useEffect, useRef } from "react";
import { motion, useReducedMotion } from "motion/react";
import { buildFightPrompt } from "./lib/prompt.js";
import { POverCurve, CountHistogram, MethodBars, WinDriversBar, Sparkline, DurationPDFChart, RoundBars, EdgeBucketChart, CLVLineChart, ClvCentsChart, ModelVsMarketScatter, survivalAt2 } from "./charts/charts.jsx";
import { survivalCrossing } from "./lib/stats.js";
import { ChartReveal } from "./charts/ChartReveal.jsx";
import { cornerSlideVariants } from "./charts/motion.js";
import {
  TabIcon, SrcIcon, MarketTabs, LineStepper, SegToggle, FilterCombobox,
  SideProbPair, QuantileTable, AddToPortfolio, useCountUp,
  buildCountLeg, buildDurLeg, buildInsideLeg, buildRoundsLeg,
  buildFinishLeg, buildFinishRndLeg, WinSplit, FightChip, UnavailableChip, lastName,
} from "./components.jsx";
import { MARKETS, PAYOUTS } from "./api/client.js";
import { ExchangeSection } from "./exchange.jsx";
import { PositionsPanel } from "./positions.jsx";
import { PortfolioRail } from "./portfolio.jsx";
import { fighterOptionsFromFights, marketOptions } from "./lib/filters.js";

const NavIcon = {
  card:      <svg viewBox="0 0 20 20"><polygon points="10,2 18,7 18,13 10,18 2,13 2,7" fill="none" strokeWidth="1.7"/><circle cx="10" cy="10" r="2.5"/></svg>,
  proplab:   <svg viewBox="0 0 20 20"><rect x="3" y="9" width="3" height="8" rx="1"/><rect x="8.5" y="5" width="3" height="12" rx="1"/><rect x="14" y="11" width="3" height="6" rx="1"/></svg>,
  positions: <svg viewBox="0 0 20 20"><path d="M10 2l2.3 5 5.7.8-4 4 1 5.5L10 15l-5 2.3 1-5.5-4-4 5.7-.8z"/></svg>,
  portfolio: <svg viewBox="0 0 20 20"><path d="M10 3l7 4-7 4-7-4z"/><path d="M3 11l7 4 7-4"/></svg>,
  market:    <svg viewBox="0 0 20 20"><path d="M2.5 12l4-5 3.5 3 5-6.5" fill="none"/><path d="M2.5 17h15" fill="none"/><circle cx="6.5" cy="7" r="1.2" fill="currentColor" stroke="none"/><circle cx="10" cy="10" r="1.2" fill="currentColor" stroke="none"/><circle cx="15" cy="3.5" r="1.2" fill="currentColor" stroke="none"/></svg>,
  settings:  <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="2.5"/><path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M5 5l1.4 1.4M13.6 13.6l1.4 1.4M5 15l1.4-1.4M13.6 6.4l1.4-1.4"/></svg>,
  about:     <svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7.5" fill="none"/><line x1="10" y1="9" x2="10" y2="14.5" strokeLinecap="round"/><circle cx="10" cy="6.2" r="1" fill="currentColor" stroke="none"/></svg>,
};

/* ══ Sidebar ════════════════════════════════════════════════════════ */

export function SidebarV3({ nav, setNav, picks, onCollapse, meta, cardSource }) {
  // Positions is card-only — live lines can't be matched to a hypothetical manual matchup.
  const showPositions = cardSource !== "manual";
  const navItems = [
    { key: "card",      label: "Fight Card" },
    { key: "proplab",   label: "Prop Lab" },
    ...(showPositions ? [{ key: "positions", label: "Positions" }] : []),
    { key: "portfolio", label: "Portfolio" },
    { key: "market",    label: "Performance" },
    { key: "about",     label: "About" },
    { key: "settings",  label: "Settings" },
  ];
  return (
    <aside className="fp-side">
      <div className="fp-side-pad">
        <div className="fp-brand" style={{ position: "relative" }}>
          <img src="/assets/fightpath-logo.png" alt="FightPath" className="fp-logo" />
          <div>
            <span className="fp-brand-name">FIGHTPATH</span>
            <span className="fp-brand-sub">PREDICTION ENGINE</span>
          </div>
          <button onClick={onCollapse} title="Collapse sidebar" aria-label="Collapse sidebar" style={{
            position: "absolute", right: 0, top: "50%", transform: "translateY(-50%)",
            width: 28, height: 28, borderRadius: 7, background: "var(--panel-3)",
            color: "var(--text-faint)", display: "grid", placeItems: "center", transition: "all .14s", flexShrink: 0,
          }}>
            <svg viewBox="0 0 20 20" style={{ width: 14, height: 14, fill: "none", stroke: "currentColor", strokeWidth: 2 }}>
              <path d="M13 4l-6 6 6 6"/>
            </svg>
          </button>
        </div>
        <nav className="fp-navitems">
          {navItems.map(({ key, label }) => (
            <button key={key} className={"fp-navitem" + (nav === key ? " on" : "")} onClick={() => setNav(key)}>
              <span className="fp-nav-ic">{NavIcon[key]}</span>
              <span>{label}</span>
              {key === "portfolio" && picks.length > 0 && (
                <span className="fp-nav-badge">{picks.length}</span>
              )}
            </button>
          ))}
        </nav>
      </div>
      <div className="fp-side-foot">
        {meta && (
          <div className="fp-model-status">
            <div className="fp-ms-row">
              <span className="fp-ms-dot"></span>
              <span className="fp-ms-lbl">Model {meta.version} · synced {meta.lastSync}</span>
            </div>
            {meta.calib?.length > 0 && <Sparkline data={meta.calib} width={232} height={28} baseline={0.5} />}
          </div>
        )}
      </div>
    </aside>
  );
}

/* ══ TopNav ═════════════════════════════════════════════════════════ */
export function TopNav({ nav, setNav, picks, onExpand, cards, selectedCardId, onCardChange, cardSource }) {
  const isManual = cardSource === "manual";
  // Positions is card-only — live lines can't be matched to a hypothetical manual matchup.
  const showPositions = !isManual;
  const navItems = [
    { key: "card",      label: "Fight Card" },
    { key: "proplab",   label: "Prop Lab" },
    ...(showPositions ? [{ key: "positions", label: "Positions" }] : []),
    { key: "portfolio", label: "Portfolio" },
    { key: "market",    label: "Performance" },
    { key: "about",     label: "About" },
    { key: "settings",  label: "Settings" },
  ];
  return (
    <header className="fp-topnav">
      {onExpand && (
        <button className="fp-tnav-expand" onClick={onExpand} title="Show sidebar" aria-label="Show sidebar" style={{
          width: 32, height: 32, borderRadius: 8, background: "var(--panel-3)",
          color: "var(--text-dim)", display: "grid", placeItems: "center",
          border: "1px solid var(--line-2)", transition: "all .15s", flexShrink: 0, marginRight: 10,
        }}>
          <svg viewBox="0 0 20 20" style={{ width: 15, height: 15, fill: "none", stroke: "currentColor", strokeWidth: 1.8 }}>
            <rect x="3" y="4" width="14" height="12" rx="2"/>
            <line x1="7.5" y1="4" x2="7.5" y2="16"/>
          </svg>
        </button>
      )}
      <div className="fp-tnav-brand">
        <img src="/assets/fightpath-logo.png" alt="FightPath" className="fp-tnav-logo" />
        <span className="fp-tnav-name">FIGHTPATH</span>
      </div>
      <nav className="fp-tnav-items">
        {navItems.map(({ key, label }) => (
          <button key={key} className={"fp-tnav-item" + (nav === key ? " on" : "")} onClick={() => setNav(key)}>
            <span className="fp-tnav-ic">{NavIcon[key]}</span>
            <span className="fp-tnav-label">{label}</span>
            {key === "portfolio" && picks.length > 0 && (
              <span className="fp-tnav-badge">{picks.length}</span>
            )}
          </button>
        ))}
      </nav>
      <div className="fp-tnav-right">
        {!isManual && cards?.length > 0 && (
          <div className="fp-tnav-evsel">
            <select value={selectedCardId || ""} onChange={(e) => onCardChange(e.target.value)}>
              {cards.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
            </select>
          </div>
        )}
        <div className="fp-tnav-live">
          <span className="fp-ms-dot"></span>
          <span>Live</span>
        </div>
      </div>
    </header>
  );
}

/* Multiplier stepper with a local text draft so the field can be fully
   cleared while typing (e.g. backspace "3.0" → "" → "0.5"); the parsed
   value commits only when it's a positive number, and blur restores the
   last committed value if the draft is empty/invalid. */
function MultInput({ mult, setMult }) {
  const [txt, setTxt] = useState(String(mult));
  const editingRef = useRef(false);
  useEffect(() => { if (!editingRef.current) setTxt(String(mult)); }, [mult]);
  return (
    <div className="fp-stepper">
      <input type="number" step="0.5" value={txt}
        onFocus={() => { editingRef.current = true; }}
        onChange={(e) => {
          setTxt(e.target.value);
          const v = parseFloat(e.target.value);
          if (v > 0) setMult(v);
        }}
        onBlur={() => {
          editingRef.current = false;
          const v = parseFloat(txt);
          setTxt(String(v > 0 ? v : mult));
        }} />
      <button className="fp-step-btn" aria-label="Decrease" onClick={() => setMult((m) => Math.max(0.5, +(m - 0.5).toFixed(2)))}>−</button>
      <button className="fp-step-btn" aria-label="Increase" onClick={() => setMult((m) => +(m + 0.5).toFixed(2))}>+</button>
    </div>
  );
}

/* ══ BetStrip ═══════════════════════════════════════════════════════ */
export function BetStrip({ payoutKey, setPayoutKey, mult, setMult }) {
  return (
    <div className="fp-betstrip">
      <div style={{ flex: 3, minWidth: 0, display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="fp-cgroup-lbl">Payout type</span>
        <div className="fp-select">
          <select value={payoutKey} onChange={(e) => setPayoutKey(e.target.value)}>
            {Object.entries(PAYOUTS).map(([k, p]) => <option key={k} value={k}>{p.label}</option>)}
          </select>
        </div>
      </div>
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="fp-cgroup-lbl">Multiplier</span>
        <MultInput mult={mult} setMult={setMult} />
      </div>
    </div>
  );
}

/* ══ Hero ═══════════════════════════════════════════════════════════ */
export function Hero({ event }) {
  if (!event) return <div className="fp-hero" style={{ minHeight: 120 }}></div>;
  return (
    <header className="fp-hero">
      <div className="fp-hero-bg"></div>
      <div className="fp-hero-content">
        <div className="fp-hero-tag"><b></b>{event.code}</div>
        <h1 className="fp-hero-title">{event.name}</h1>
        <div className="fp-hero-meta">
          <span>{event.venue}</span>
          <i></i>
          <span>{event.date}</span>
        </div>
      </div>
    </header>
  );
}

/* ══ MatchupHeader ══════════════════════════════════════════════════ */
function fmtRecord(rec) {
  if (!rec || rec.length < 2) return null;
  const [w, l, d] = rec;
  return (d > 0) ? `${w}-${l}-${d}` : `${w}-${l}`;
}

export function MatchupHeader({ fight, marketRows, marketLinesLoading }) {
  const reduced = useReducedMotion();
  if (!fight) return null;
  const aL = lastName(fight.a.name);
  const bL = lastName(fight.b.name);
  const aFav = fight.a.pWin >= fight.b.pWin;
  const winnerLast = aFav ? aL : bL;
  const otherLast = aFav ? bL : aL;
  const aRec = fmtRecord(fight.a.record);
  const bRec = fmtRecord(fight.b.record);
  return (
    <div className="fp-matchup">
      <div className="fp-mu-head">
        <motion.div className="fp-mu-side a" key={"a:" + fight.id} {...cornerSlideVariants("a", reduced)}>
          <span className={"fp-mu-name" + (aFav ? " fav" : "")}>{fight.a.name}</span>
          {aRec && <span className="fp-mu-rec">{aRec}</span>}
        </motion.div>
        <div className="fp-mu-center">
          <span className="fp-mu-vs">VS</span>
          {(fight.weightClass || fight.rounds) && (
            <span className="fp-mu-bout">
              <span className="fp-mu-bout-txt">
                {fight.weightClass || "Catchweight"}
                {fight.rounds ? ` · ${fight.rounds}R` : ""}
                {fight.isTitle ? " · TITLE" : ""}
              </span>
            </span>
          )}
        </div>
        <motion.div className="fp-mu-side b" key={"b:" + fight.id} {...cornerSlideVariants("b", reduced, 0.05)}>
          <span className={"fp-mu-name" + (!aFav ? " fav" : "")}>{fight.b.name}</span>
          {bRec && <span className="fp-mu-rec">{bRec}</span>}
        </motion.div>
      </div>
      {fight.lowData && (
        <div className="fp-lowdata-banner">
          <span className="fp-lowdata-icon">⚠</span>
          <span className="fp-lowdata-msg">
            <b>Limited UFC data</b> — {fight.nFightsRed === 0 ? fight.a.name : fight.nFightsBlue === 0 ? fight.b.name : "one fighter"} has 3 or fewer UFC bouts on record.
            Win probability is shrunk toward 50/50. Treat with caution.
            {fight.nFightsRed !== undefined && ` (${lastName(fight.a.name)}: ${fight.nFightsRed} fight${fight.nFightsRed !== 1 ? "s" : ""} · ${lastName(fight.b.name)}: ${fight.nFightsBlue} fight${fight.nFightsBlue !== 1 ? "s" : ""})`}
          </span>
        </div>
      )}
      <div className="fp-mu-grid">
        <div>
          <div className="fp-mu-block-ttl" style={{ "--acc": "var(--gold-br)" }}>WINNER PROBABILITY · P(WIN)</div>
          <div key={"ws:" + fight.id}>
            <WinSplit a={fight.a} b={fight.b} marketRows={marketRows} marketLinesLoading={marketLinesLoading} />
          </div>
        </div>
        <div>
          <div className="fp-mu-block-ttl" style={{ "--acc": "var(--m-port)" }}>METHOD DISTRIBUTION · P(METHOD)</div>
          <ChartReveal replayKey={fight.id} delay={0.15}>
            <MethodBars method={fight.method} />
          </ChartReveal>
        </div>
      </div>
      {fight.winnerDrivers?.length > 0 && (
        <div className="fp-drv-wrap">
          <div className="fp-mu-block-ttl" style={{ "--acc": "var(--gold-br)" }}>
            WHAT'S DRIVING THIS PICK · TOP FACTORS
          </div>
          <p className="fp-drv-hint">
            Each bar shows how hard one area pushes the model's pick toward a fighter — a longer bar means a bigger pull.
            <span className="fp-drv-unit"><b>pts</b> = percentage points it adds to that fighter's win chance.</span>
          </p>
          <ChartReveal replayKey={fight.id} delay={0.25}>
            <WinDriversBar drivers={fight.winnerDrivers} aLast={aL} bLast={bL} aFav={aFav} />
          </ChartReveal>
          <div className="fp-drv-legend">
            <span className="fp-drv-leg-item"><i style={{ background: "var(--gold-br)" }} />{winnerLast}</span>
            <span className="fp-drv-leg-item"><i style={{ background: "var(--other)" }} />{otherLast}</span>
          </div>
        </div>
      )}
    </div>
  );
}

/* ══ AI Fight-Path Prompt Panel ══════════════════════════════════════ */

function AiPromptPanel({ fight }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [copied, setCopied] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => {
    setText(buildFightPrompt(fight));
    setCopied(false);
  }, [fight?.id]);

  function handleCopy() {
    if (timerRef.current) clearTimeout(timerRef.current);
    const doCopy = async () => {
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        // fallback for non-https / older browsers
        const el = document.createElement("textarea");
        el.value = text;
        el.style.position = "fixed";
        el.style.opacity = "0";
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
      }
      setCopied(true);
      timerRef.current = setTimeout(() => setCopied(false), 1500);
    };
    doCopy();
  }

  return (
    <div className="fp-aiprompt">
      <div className="fp-aiprompt-hd">
        <button
          className="fp-aiprompt-toggle"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-controls="fp-aiprompt-body"
        >
          <svg className={"fp-aiprompt-chevron" + (open ? " open" : "")} viewBox="0 0 16 16" aria-hidden="true">
            <path d="M4 6l4 4 4-4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span className="fp-aiprompt-title">AI FIGHT-PATH PROMPT</span>
        </button>
        <button
          className={"fp-aiprompt-copy" + (copied ? " copied" : "")}
          onClick={handleCopy}
          aria-label={copied ? "Prompt copied" : "Copy prompt to clipboard"}
        >
          {copied ? (
            <>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8l3.5 3.5 7-7" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
              Copied!
            </>
          ) : (
            <>
              <svg viewBox="0 0 16 16" aria-hidden="true"><rect x="5" y="5" width="9" height="10" rx="1.5" fill="none" strokeWidth="1.4"/><path d="M2 11V3a1 1 0 011-1h8" strokeWidth="1.4" strokeLinecap="round"/></svg>
              Copy prompt
            </>
          )}
        </button>
      </div>
      {open && (
        <div id="fp-aiprompt-body" className="fp-aiprompt-body">
          <p className="fp-aiprompt-hint">Paste into ChatGPT, Claude, Gemini, or any AI chat to get the predicted fight narrative.</p>
          <textarea
            className="fp-aiprompt-box"
            value={text}
            onChange={(e) => setText(e.target.value)}
            aria-label="Generated AI fight-path prompt"
            spellCheck={false}
          />
        </div>
      )}
    </div>
  );
}

/* ══ Pages ══════════════════════════════════════════════════════════ */

export function FightCardPage({ fights, unavailFights = [], selId, setSelId, event, marketLines, marketLinesLoading }) {
  const fight = fights?.find((f) => f.id === selId) || fights?.[0];
  const fightIdx = fight?.id?.startsWith("fight_") ? Number(fight.id.slice(6)) : -1;
  const marketRows = (marketLines?.rows || []).filter(
    (r) => r.marketKind === "winner" && r.fightIdx === fightIdx
  );
  const merged = useMemo(() => {
    const avail = (fights || []).map((f) => ({
      idx: f.id?.startsWith("fight_") ? Number(f.id.slice(6)) : -1,
      node: <FightChip key={f.id} fight={f} selected={f.id === selId} onSelect={setSelId} />,
    }));
    const unavail = (unavailFights || []).map((f) => ({
      idx: f.idx,
      node: <UnavailableChip key={f.id} fight={f} />,
    }));
    return [...avail, ...unavail].sort((a, b) => a.idx - b.idx);
  }, [fights, unavailFights, selId, setSelId]);
  const total = (fights?.length || 0) + (unavailFights?.length || 0);
  const sub = unavailFights?.length > 0
    ? `${total} bouts · ${unavailFights.length} not priced · select a fight to analyze`
    : `${total} bouts · select a fight to analyze`;
  return (
    <div className="fp-page fp-page-card">
      <Hero event={event} />
      <div className="fp-card-body">
        <div>
          <div className="fp-section-hd">
            <div>
              <h2 className="fp-section-ttl">Fight Card</h2>
              <span className="fp-section-sub">{sub}</span>
            </div>
          </div>
          {merged.length > 0 && (
            <div className="fp-fightrail">
              {merged.map((m) => m.node)}
            </div>
          )}
        </div>
        {fight && <MatchupHeader fight={fight} marketRows={marketRows} marketLinesLoading={marketLinesLoading} />}
        {fight && <AiPromptPanel fight={fight} />}
      </div>
    </div>
  );
}

export function PropLabPage({ fight, fights, selId, setSelId, tab, setTab, durLine, setDurLine, durSide, setDurSide, cp, patchCp, breakeven, picks, onToggle, onNavigate }) {
  const [sigSub, setSigSub] = useState("sig");   // strike-target sub-tab: sig | bodySig | legSig
  if (!fight) return <div className="fp-page"><div className="fp-page-body" style={{ color: "var(--text-faint)" }}>Select a fight to analyze.</div></div>;
  const aFav = fight.a.pWin >= fight.b.pWin;
  const aL = lastName(fight.a.name);
  const bL = lastName(fight.b.name);
  const ACC = {
    sig:     { accent: "var(--m-sig)",   soft: "rgba(236,155,60,.12)", line: "rgba(236,155,60,.32)" },
    r1sig:   { accent: "var(--m-r1)",    soft: "rgba(236,97,73,.12)",  line: "rgba(236,97,73,.32)" },
    bodySig: { accent: "var(--m-body)",  soft: "rgba(240,168,104,.12)", line: "rgba(240,168,104,.32)" },
    legSig:  { accent: "var(--m-leg)",   soft: "rgba(217,138,78,.12)",  line: "rgba(217,138,78,.32)" },
    combo:   { accent: "var(--m-combo)", soft: "rgba(245,192,75,.12)",  line: "rgba(245,192,75,.32)" },
    td:      { accent: "var(--m-td)",    soft: "rgba(70,188,132,.12)", line: "rgba(70,188,132,.32)" },
    r1td:    { accent: "var(--m-r1td)",  soft: "rgba(95,208,160,.12)", line: "rgba(95,208,160,.32)" },
    subAtt:  { accent: "var(--m-sub)",   soft: "rgba(54,168,155,.12)", line: "rgba(54,168,155,.32)" },
    ctrl:    { accent: "var(--m-ctrl)",  soft: "rgba(63,174,143,.12)", line: "rgba(63,174,143,.32)" },
    kd:      { accent: "var(--m-kd)",    soft: "rgba(224,96,142,.12)", line: "rgba(224,96,142,.32)" },
  };
  // Body/leg sig strikes fold into the Sig Strikes tab as a strike-target sub-toggle.
  const SIG_SUBS = [
    { value: "sig",     label: "All" },
    { value: "bodySig", label: "Body" },
    { value: "legSig",  label: "Leg" },
  ];
  // Grouped left→right: fight shape · striking · grappling · power/finish.
  const TABS = [
    { key: "duration", icon: "duration", label: "Duration",    accent: "var(--m-dur)", trust: MARKETS.duration.trust, group: "FIGHT SHAPE" },
    { key: "rounds",   icon: "rounds",   label: "Rounds",      accent: "var(--m-rnd)", trust: MARKETS.rounds.trust, group: "FIGHT SHAPE" },
    { key: "sig",      icon: "sig",      label: "Sig Strikes", accent: "var(--m-sig)", trust: MARKETS.sig.trust, group: "STRIKING" },
    { key: "r1sig",    icon: "r1sig",    label: "R1 Strikes",  accent: "var(--m-r1)",  trust: MARKETS.r1sig.trust, group: "STRIKING" },
    { key: "combo",    icon: "combo",    label: "Combined",    accent: "var(--m-combo)", trust: MARKETS.combo.trust, group: "STRIKING" },
    { key: "td",       icon: "td",       label: "Takedowns",   accent: "var(--m-td)",  trust: MARKETS.td.trust, group: "GRAPPLING" },
    { key: "r1td",     icon: "r1td",     label: "R1 TD",       accent: "var(--m-r1td)", trust: MARKETS.r1td.trust, group: "GRAPPLING" },
    { key: "subAtt",   icon: "subAtt",   label: "Sub Att",     accent: "var(--m-sub)", trust: MARKETS.subAtt.trust, group: "GRAPPLING" },
    { key: "ctrl",     icon: "ctrl",     label: "Control",     accent: "var(--m-ctrl)", trust: MARKETS.ctrl.trust, group: "GRAPPLING" },
    { key: "kd",       icon: "kd",       label: "Knockdowns",  accent: "var(--m-kd)",  trust: MARKETS.kd.trust, group: "POWER / FINISH" },
    { key: "finishes", icon: "finishes", label: "Finishes",    accent: "var(--m-r1)",  trust: MARKETS.finish.trust, group: "POWER / FINISH" },
  ];
  function renderTab() {
    if (tab === "duration") return <DurationPanel fight={fight} line={durLine} setLine={setDurLine} side={durSide} setSide={setDurSide} breakeven={breakeven} picks={picks} onToggle={onToggle} leg={(L, s, p) => buildDurLeg(fight, L, s, p)} />;
    if (tab === "finishes") return <FinishesPanel fight={fight} breakeven={breakeven} picks={picks} onToggle={onToggle} />;
    if (tab === "rounds")   return <RoundsPanel   fight={fight} breakeven={breakeven} picks={picks} onToggle={onToggle} />;
    if (tab === "combo") {
      const ac = ACC.combo, sc = cp.combo;
      return <CountPropPanel fight={fight} fightLevel propData={fight.sigCombo} fighterLast="Combined" marketKey="combo"
        accent={ac.accent} accSoft={ac.soft} accLine={ac.line}
        line={sc.line} setLine={(v) => patchCp("combo", { line: v })}
        side={sc.side} setSide={(v) => patchCp("combo", { side: v })}
        breakeven={breakeven} picks={picks} onToggle={onToggle}
        aLabel={aL} bLabel={bL}
        leg={(_f, _fl, mk, L, side, p, acc) => buildCountLeg(fight, "Combined", aL + "+" + bL, mk, L, side, p, acc)} />;
    }
    if (tab === "sig") {
      const mk = sigSub;                      // "sig" | "bodySig" | "legSig"
      const a = ACC[mk], st = cp[mk];
      const fighter = st.f === "a" ? fight.a : fight.b;
      const fLast = lastName(fighter.name);
      // Keep the chosen corner consistent across all three strike targets.
      const setFighter = (v) => SIG_SUBS.forEach((s) => patchCp(s.value, { f: v }));
      return <CountPropPanel fight={fight} fighter={fighter} fighterLast={fLast} marketKey={mk}
        accent={a.accent} accSoft={a.soft} accLine={a.line}
        fighterSide={st.f} setFighterSide={setFighter}
        line={st.line} setLine={(v) => patchCp(mk, { line: v })}
        side={st.side} setSide={(v) => patchCp(mk, { side: v })}
        target={{ value: sigSub, onChange: setSigSub, options: SIG_SUBS }}
        breakeven={breakeven} picks={picks} onToggle={onToggle}
        aLabel={aL} bLabel={bL}
        leg={(fighter, fLast, m, L, side, p, acc) => buildCountLeg(fight, fighter.name, fLast, m, L, side, p, acc)} />;
    }
    const a = ACC[tab], st = cp[tab];
    const fighter = st.f === "a" ? fight.a : fight.b;
    const fLast = lastName(fighter.name);
    return <CountPropPanel fight={fight} fighter={fighter} fighterLast={fLast} marketKey={tab}
      accent={a.accent} accSoft={a.soft} accLine={a.line}
      fighterSide={st.f} setFighterSide={(v) => patchCp(tab, { f: v })}
      line={st.line} setLine={(v) => patchCp(tab, { line: v })}
      side={st.side} setSide={(v) => patchCp(tab, { side: v })}
      breakeven={breakeven} picks={picks} onToggle={onToggle}
      aLabel={aL} bLabel={bL}
      leg={(fighter, fLast, mk, L, side, p, acc) => buildCountLeg(fight, fighter.name, fLast, mk, L, side, p, acc)} />;
  }
  return (
    <div className="fp-page fp-page-proplab">
      <div className="fp-page-hd">
        <div className="fp-page-ttl">Prop Lab</div>
        <span className="fp-page-sub">
          Deep-dive any market — type a line, get the model's P(over/under).
          {onNavigate && <> Ranked picks vs live lines → <button type="button" className="fp-page-sub-link" onClick={() => onNavigate("positions")}>Positions</button></>}
        </span>
      </div>
      <div className="fp-fight-sel-bar">
        <div className="fp-fsel-wrap">
          <select value={selId} onChange={(e) => setSelId(e.target.value)}>
            {fights?.map((f) => {
              const mu = `${lastName(f.a.name)} vs ${lastName(f.b.name)}`;
              // Drop empty weight class and the internal "manual" slot tag.
              const meta = [f.weightClass, f.slot && f.slot !== "manual" ? f.slot : null].filter(Boolean).join(" · ");
              return <option key={f.id} value={f.id}>{meta ? `${mu} · ${meta}` : mu}</option>;
            })}
          </select>
        </div>
        <div className="fp-fsb-probs">
          <div className="fp-fsb-side">
            <span className="fp-fsb-name" style={{ color: aFav ? "var(--gold-br)" : "var(--text)" }}>{aL}</span>
            <span className="fp-fsb-pct">{(fight.a.pWin * 100).toFixed(0)}% P(WIN)</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 5, alignItems: "center" }}>
            <div className="fp-fsb-bar-wrap">
              <div className="fp-fsb-bar-a" style={{ width: fight.a.pWin * 100 + "%", background: aFav ? "var(--gold-br)" : "var(--other)" }}></div>
              <div className="fp-fsb-bar-b" style={{ width: fight.b.pWin * 100 + "%", background: !aFav ? "var(--gold-br)" : "var(--other)" }}></div>
            </div>
          </div>
          <div className="fp-fsb-side r">
            <span className="fp-fsb-name" style={{ color: !aFav ? "var(--gold-br)" : "var(--text)" }}>{bL}</span>
            <span className="fp-fsb-pct">{(fight.b.pWin * 100).toFixed(0)}% P(WIN)</span>
          </div>
        </div>
      </div>
      <div className="fp-proplab-ws">
        <MarketTabs tabs={TABS} active={tab} onChange={setTab} />
        {renderTab()}
      </div>
    </div>
  );
}

export function PositionsPage({ fights, marketLines, marketLinesLoading, onRefreshMarketLines }) {
  // "kalshi" was the old id for the full contract list; it is now split into
  // "positions" (the decision board) and "explore" (the same full list). Migrate a
  // stored "kalshi" to positions so returning users don't land on a blank pane.
  const [view, setView] = useState(() => {
    try {
      const v = localStorage.getItem("fp-bb-view");
      return v === "kalshi" ? "positions" : (v || "positions");
    } catch { return "positions"; }
  });
  const pickView = (v) => { setView(v); try { localStorage.setItem("fp-bb-view", v); } catch {} };
  const [showLowData, setShowLowData] = useState(false);   // default: hidden
  const lowDataCount = (fights || []).filter((f) => f?.lowData).length;

  // Filter tokens live here (not in either lane) so they persist across the
  // Kalshi/Books-DFS toggle — both lanes stay mounted via display:none.
  const [filterTokens, setFilterTokens] = useState([]);
  useEffect(() => { setFilterTokens([]); }, [fights]);
  const fighterOpts = useMemo(() => fighterOptionsFromFights(fights), [fights]);
  const marketOpts = useMemo(() => marketOptions(), []);

  return (
    <div className="fp-page fp-page-positions">
      <div className="fp-page-hd">
        <div>
          <div className="fp-page-ttl">Positions</div>
          <span className="fp-page-sub">One position per fight, staked inside your card budget</span>
        </div>
        <div className="fp-bb-viewbar">
          <SegToggle value={view} onChange={pickView}
            options={[{ value: "positions", label: "Positions" },
                      { value: "explore", label: "Explore" }]} />
          <div className="fp-cgroup">
            <span className="fp-cgroup-lbl">Low data{lowDataCount ? ` (${lowDataCount})` : ""}</span>
            <SegToggle value={showLowData ? "show" : "hide"} onChange={(v) => setShowLowData(v === "show")}
              options={[{ value: "hide", label: "Hide" }, { value: "show", label: "Show" }]} />
          </div>
        </div>
      </div>
      <div className="fp-page-body">
        {/* Positions is a whole-card decision surface (one row per bout, ~12 rows)
            and takes no filter tokens — showing an inert filter box above it would
            be a control that controls nothing. */}
        {view !== "positions" && (
          <FilterCombobox tokens={filterTokens} onChange={setFilterTokens}
            fighterOptions={fighterOpts} marketOptions={marketOpts} />
        )}
        <div style={{ display: view === "positions" ? "" : "none" }}>
          <PositionsPanel rows={marketLines?.rows || []} fights={fights} />
        </div>
        <div style={{ display: view === "explore" ? "" : "none" }}>
          <ExchangeSection fights={fights} marketLines={marketLines} loading={marketLinesLoading}
            onRefresh={onRefreshMarketLines} showLowData={showLowData} filterTokens={filterTokens} />
        </div>
      </div>
    </div>
  );
}

export function PortfolioPage({ picks, onRemove, onClear, payout, mult, multDirty, breakeven }) {
  return (
    <div className="fp-page fp-page-portfolio">
      <div className="fp-page-hd">
        <div className="fp-page-ttl">Portfolio</div>
        <span className="fp-page-sub">Build your pick slip · grade combined probability · assess EV</span>
      </div>
      <div className="fp-page-body">
        <div className="fp-port-page">
          <PortfolioRail picks={picks} onRemove={onRemove} onClear={onClear} payout={payout} mult={mult} multDirty={multDirty} breakeven={breakeven} />
          <div className="fp-port-info">
            <div className="fp-port-how">
              <h3>How it works</h3>
              <p>Add prop legs from <b>Prop Lab</b> or <b>Positions</b>. Each leg shows the model's estimated P(hit) vs the break-even probability.</p>
              <ul>
                <li>Leg count is automatic — the parlay prices off the legs you actually add, no manual selection</li>
                <li>Combined hit % = product of all leg model probabilities</li>
                <li>Effective multiplier = base payout (by platform + leg count) × each leg's own line multiplier (Flat Multi goblin/demon/boost folded in)</li>
                <li>Break-even = 1 ÷ effective multiplier · EV = (combined hit × effective multiplier) − 1</li>
              </ul>
            </div>
            <div className="fp-port-how">
              <h3>Payout structure</h3>
              <p style={{ fontFamily: "var(--f-mono)", fontSize: 13, color: "var(--gold-br)" }}>{payout.platform === "ud" ? "Flat Multi" : "Power Play"}</p>
              <p>Pick the platform in the control deck — leg count and effective multiplier are derived from your slip automatically. Power Play demon/goblin multipliers are not yet folded in (priced at flat break-even).</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function HistoryPageSkeleton() {
  return (
    <div className="fp-page fp-page-history" style={{ position: "relative", overflow: "hidden" }}>
      <div className="fp-cl-progress" aria-hidden="true"><div className="fp-cl-progress-bar" /></div>
      <div className="fp-page-hd">
        <div className="fp-skeleton" style={{ width: 110, height: 34, borderRadius: 6 }} />
        <div className="fp-skeleton" style={{ width: 260, height: 10, borderRadius: 4, marginTop: 12 }} />
      </div>
      <div className="fp-page-body">
        <div className="fp-hist-summary">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="fp-hist-sum-tile">
              <div className="fp-skeleton" style={{ width: 58, height: 28, borderRadius: 5 }} />
              <div className="fp-skeleton" style={{ width: 82, height: 9, borderRadius: 3, marginTop: 7 }} />
            </div>
          ))}
        </div>
        <div className="fp-hist-grid">
          {[6, 5, 4].map((rows, ci) => (
            <div key={ci} className="fp-ev-card">
              <div className="fp-ev-hd">
                <div className="fp-skeleton" style={{ width: 230, height: 18, borderRadius: 5 }} />
                <div className="fp-skeleton" style={{ width: 68, height: 10, borderRadius: 3 }} />
                <div className="fp-skeleton" style={{ width: 64, height: 22, borderRadius: 7, marginLeft: "auto" }} />
              </div>
              {[...Array(rows)].map((_, ri) => (
                <div key={ri} style={{ padding: "10px 16px", borderBottom: ri < rows - 1 ? "1px solid var(--line)" : "none", display: "flex", gap: 20, alignItems: "center" }}>
                  <div className="fp-skeleton" style={{ flex: "0 0 200px", height: 11, borderRadius: 3 }} />
                  <div className="fp-skeleton" style={{ flex: "0 0 110px", height: 11, borderRadius: 3 }} />
                  <div className="fp-skeleton" style={{ flex: "0 0 90px", height: 11, borderRadius: 3 }} />
                  <div className="fp-skeleton" style={{ flex: "0 0 24px", height: 24, borderRadius: 6 }} />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="fp-cl-status">
        <span className="fp-cl-dot" aria-hidden="true" />
        Loading history…
      </div>
    </div>
  );
}

// Current streak from the shown events: flattens fights most-recent-first (events[]
// is already most-recent-event-first; within an event, fight_id_to_pos position 0 is
// the main event — UFCStats lists the main card top-down, main event first — so the
// raw fight order within an event is already main-event-first / most-recent-first,
// with no reversal needed) and counts the leading run of same-result picks.
function computeStreak(events) {
  const flat = [];
  for (const ev of events) {
    const fights = ev.fights || [];
    for (const f of fights) flat.push(f.correct);
  }
  if (!flat.length) return null;
  const win = flat[0];
  let n = 0;
  for (const c of flat) { if (c === win) n++; else break; }
  return { win, n };
}

// Best night among the shown events (min 5 fights, so a 2/2 card can't win on noise).
function computeBestNight(events) {
  const eligible = events.filter((e) => e.total >= 5);
  if (!eligible.length) return null;
  return eligible.reduce((best, e) => (e.hitRate > best.hitRate ? e : best));
}

export function WinnerRecordView({ history }) {
  if (!history) return <HistoryPageSkeleton />;
  const events  = history.events  || [];
  const totals  = history.totals  || {};
  const dbEvents = totals.dbEvents ?? events.length;
  const dbFights = totals.dbFights ?? events.reduce((s, e) => s + e.total, 0);
  // Single accumulating tally: eval-model backtest (through "now") + live prod model after.
  const correct = totals.correct ?? totals.testCorrect ?? events.reduce((s, e) => s + e.correct, 0);
  const wrong   = totals.wrong   ?? totals.testWrong   ?? events.reduce((s, e) => s + (e.total - e.correct), 0);
  const fights  = totals.fights  ?? (correct + wrong);
  const hitRate = totals.hitRate ?? totals.testHitRate ?? (fights > 0 ? correct / fights : 0);
  const livePending = totals.livePending ?? 0;
  const streak = computeStreak(events);
  const bestNight = computeBestNight(events);

  // events[] is most-recent-first; live (prod forward-record) events are contiguous
  // at the top. seamAfter = index of the last live event, or -1 if none are shown.
  let seamAfter = -1;
  for (let i = 0; i < events.length; i++) {
    if (events[i].live) seamAfter = i; else break;
  }
  const showSeam = seamAfter >= 0 && seamAfter < events.length - 1 && totals.liveSince;

  const cards = [];
  events.forEach((ev, i) => {
    if (showSeam && i === seamAfter + 1) {
      cards.push(
        <div className="fp-seam" key="seam">
          <div className="fp-seam-chip"><span className="fp-seam-chip-txt">Live record begins · {totals.liveSince}</span></div>
          <div className="fp-seam-note">Above: the served model, logged before each fight. Below: the eval model's backtest on held-out fights.</div>
        </div>
      );
    }
    cards.push(<EventCard key={ev.id} ev={ev} />);
  });

  return (
    <>
      <div className="fp-hist-tape">
        <div className="fp-hist-tape-id">
          <span className="fp-hist-tape-eyebrow">The model's record</span>
          <span className="fp-hist-tape-name">FightPath</span>
        </div>
        <div className="fp-hist-tape-rows">
          <div className="fp-hist-tape-row">
            <span className="fp-hist-tape-val">{correct}–{wrong}</span>
            <span className="fp-hist-tape-lbl">Record</span>
          </div>
          <div className="fp-hist-tape-row">
            <span className="fp-hist-tape-val" style={{ color: hitRate >= 0.6 ? "var(--pos)" : "var(--gold-br)" }}>{(hitRate * 100).toFixed(1)}%</span>
            <span className="fp-hist-tape-lbl">Hit rate</span>
          </div>
          {streak && (
            <div className="fp-hist-tape-row">
              <span className={"fp-hist-tape-val " + (streak.win ? "pos" : "neg")}>{streak.win ? "W" : "L"}{streak.n}</span>
              <span className="fp-hist-tape-lbl">Current streak</span>
            </div>
          )}
          {bestNight && (
            <div className="fp-hist-tape-row">
              <span className="fp-hist-tape-val">{bestNight.correct}/{bestNight.total}</span>
              <span className="fp-hist-tape-lbl">Best night — {bestNight.event}</span>
            </div>
          )}
        </div>
        <div className="fp-hist-tape-db"><span className="fp-hist-tape-db-txt">{dbFights} fights · {dbEvents} events in DB</span></div>
      </div>
      <div className="fp-hist-grid">
        {cards}
      </div>
    </>
  );
}

function ResultStamp({ f }) {
  const conf = f.pRed != null ? (f.predWinner === f.red ? f.pRed : 1 - f.pRed) : null;
  const tag = f.correct
    ? (conf != null && conf < 0.55 ? "CALLED IT" : null)
    : (conf != null && conf >= 0.65 ? "UPSET" : null);
  return (
    <span className="fp-stamp-wrap">
      <span className={"fp-stamp " + (f.correct ? "hit" : "miss")}>{f.correct ? "Hit" : "Miss"}</span>
      {tag && <span className={"fp-stamp-tag " + (tag === "UPSET" ? "upset" : "called")}>{tag}</span>}
    </span>
  );
}

export function EventCard({ ev }) {
  return (
    <div className="fp-ev-card">
      <div className="fp-ev-hd">
        <span className="fp-ev-name">{ev.event}</span>
        {ev.live && <span className="fp-ev-live-tag"><span className="fp-ev-live-tag-txt">Live</span></span>}
        <span className="fp-ev-date">{ev.date}</span>
        <span className={"fp-ev-verdict " + (ev.hitRate >= 0.64 ? "good" : "ok")}>
          <span className="fp-ev-verdict-frac">{ev.correct}/{ev.total}</span>
          <span className="fp-ev-verdict-pct">{(ev.hitRate * 100).toFixed(0)}%</span>
        </span>
      </div>
      <table className="fp-ev-table">
        <thead><tr><th>Fight</th><th>Model lean</th><th>Actual winner</th><th>Result</th></tr></thead>
        <tbody>
          {ev.fights.map((f, i) => (
            <tr key={i}>
              <td className="fp-hist-fight">{f.red} vs {f.blue}</td>
              <td className="fp-hist-pred">{f.predWinner} <span style={{ color: "var(--gold-br)", fontWeight: 600 }}>{f.pRed != null ? ((f.predWinner === f.red ? f.pRed : 1 - f.pRed) * 100).toFixed(0) + "%" : ""}</span></td>
              <td className="fp-hist-winner">{f.actualWinner}</td>
              <td><ResultStamp f={f} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ══ Model vs Market ═════════════════════════════════════════════════ */

// Mirrors configs/prop_trust.yaml — same source as Prop Lab's tab chips
// (api/client.js MARKETS), keyed by the ledger's snake_case market names
// instead of the UI's abbreviated tab keys.
const LEDGER_TRUST = {
  sig_strikes: "TRUST", leg_sig_strikes: "TRUST", duration: "TRUST", body_sig_strikes: "TRUST",
  takedowns: "WATCH", r1_sig_strikes: "WATCH", ctrl_time: "WATCH", r1_takedowns: "WATCH",
  sig_strikes_combo: "WATCH", ko_finish: "WATCH", sub_finish: "WATCH", finish: "WATCH", r1_finish: "WATCH",
  knockdowns: "CUT", sub_attempts: "CUT",
};
const TRUST_LABEL = { TRUST: "Trusted", WATCH: "Watch", CUT: "Low signal" };
const TRUST_CLS   = { TRUST: "trust",   WATCH: "watch", CUT: "cut" };

function ModelVsMarketSkeleton() {
  return (
    <div className="fp-page fp-page-mvm" style={{ position: "relative", overflow: "hidden" }}>
      <div className="fp-cl-progress" aria-hidden="true"><div className="fp-cl-progress-bar" /></div>
      <div className="fp-page-hd">
        <div className="fp-skeleton" style={{ width: 220, height: 34, borderRadius: 6 }} />
        <div className="fp-skeleton" style={{ width: 320, height: 10, borderRadius: 4, marginTop: 12 }} />
      </div>
      <div className="fp-page-body">
        <div className="fp-skeleton" style={{ height: 220, borderRadius: 12 }} />
      </div>
    </div>
  );
}

export function ModelVsMarketPage({ ledger, history }) {
  const [perfView, setPerfView] = useState("record");
  if (!ledger) return <ModelVsMarketSkeleton />;

  const dfs = ledger.available ? ledger : null;
  const exch = ledger.exchange?.available ? ledger.exchange : null;

  return (
    <div className="fp-page fp-page-mvm">
      <div className="fp-page-hd">
        <div className="fp-page-ttl">Performance</div>
        <span className="fp-page-sub">The model's graded track record — win/loss on past cards, and how its bets priced against the market</span>
      </div>
      <div className="fp-page-body">
        <div className="fp-perf-toggle">
          <SegToggle
            value={perfView}
            onChange={setPerfView}
            options={[{ value: "record", label: "Record" }, { value: "betting", label: "Betting" }]}
          />
        </div>
        {perfView === "record" ? (
          <WinnerRecordView history={history} />
        ) : (
        <>
        <div className="fp-exch-hd" style={{ marginTop: 0 }}>
          <span className="fp-exch-ttl">Exchange Ledger — Kalshi</span>
          <span className="fp-exch-sub">Every graded taker pick across all Kalshi markets — winner, method, rounds, distance — priced vs the fee-adjusted ask</span>
        </div>
        {!exch ? (
          <div className="fp-mvm-empty" style={{ marginBottom: "var(--gap)" }}>
            No graded Kalshi picks yet. This section fills in once the exchange lane
            (see Positions → Explore) has resolved rows — check back after a Kalshi-listed
            card has been logged and graded.
          </div>
        ) : (
          <>
            <div className="fp-hist-summary">
              <div className="fp-hist-sum-tile" title="Total graded taker picks across every Kalshi market, not just the winner market.">
                <b style={{ color: "var(--text)" }}>{exch.overall.n}</b><span>GRADED PICKS</span>
              </div>
              <div className="fp-hist-sum-tile" title="Blended strike rate across all markets. Most picks are deliberate longshots (round / method props), so this runs far below the winner-only rate — it is not the model's winner accuracy. Judge the lane by CLV, not hit rate.">
                <b style={{ color: exch.overall.hitRate == null ? "var(--text-faint)" : exch.overall.hitRate >= 0.5 ? "var(--pos)" : "var(--gold-br)" }}>{exch.overall.hitRate != null ? (exch.overall.hitRate * 100).toFixed(1) + "%" : "—"}</b><span>HIT RATE</span>
              </div>
              <div className="fp-hist-sum-tile" title="Average cents the price moved in our favor between when we locked the pick and the market close. Positive = we got in ahead of the move.">
                <b style={{ color: exch.avgClvCents != null && exch.avgClvCents >= 0 ? "var(--pos)" : "var(--neg)" }}>{exch.avgClvCents != null ? (exch.avgClvCents >= 0 ? "+" : "") + exch.avgClvCents.toFixed(1) + "c" : "—"}</b><span>AVG CLV</span>
              </div>
              <div className="fp-hist-sum-tile" title="Of the picks whose price moved, the share where we locked a BETTER price than the market's closing price. Above 50% means we were consistently ahead of the market — the strongest long-run profit signal.">
                <b style={{ color: exch.beatClose != null && exch.beatClose >= 0.5 ? "var(--pos)" : "var(--text-faint)" }}>
                  {exch.beatClose != null ? (exch.beatClose * 100).toFixed(0) + "%" : "—"}
                </b>
                <span>BEAT CLOSE{exch.beatCloseN ? ` (n=${exch.beatCloseN})` : ""}</span>
              </div>
            </div>
            <p className="fp-exch-note">
              These numbers cover <b>all</b> Kalshi markets, so most picks are deliberate longshots (round / method props) — a low <b>hit rate</b> is expected and is <b>not</b> the model's winner accuracy. The lane's real scorecard is <b>CLV</b>: <b>beat close</b> is how often we locked a better price than the market's final one (above 50% is good).
            </p>

            <div className="fp-exch-charts">
              <div className="fp-card">
                <div className="fp-card-hd">
                  <span className="fp-card-ttl">Exchange CLV Over Time</span>
                  <span className="fp-card-tag">Avg Cents Moved In Our Favor</span>
                </div>
                {/* The card stretches to match the scatter (see .fp-exch-charts) and
                    centres this plot, so the height here only sets the CLV chart's own
                    aspect — it no longer has to pixel-match the neighbouring card. */}
                {exch.clvSeries?.length >= 2 ? (
                  <ChartReveal replayKey="mvm-exch-clv"><ClvCentsChart series={exch.clvSeries} height={200} /></ChartReveal>
                ) : (
                  <div className="fp-mvm-empty small">No closing lines captured yet for the exchange lane — fills in once at least two events have a captured close.</div>
                )}
              </div>
              <div className="fp-card">
                <div className="fp-card-hd">
                  <span className="fp-card-ttl">Model vs Market</span>
                  <span className="fp-card-tag">Graded Picks</span>
                </div>
                {exch.points?.length ? (
                  <ChartReveal replayKey="mvm-exch-scatter"><ModelVsMarketScatter points={exch.points} /></ChartReveal>
                ) : (
                  <div className="fp-mvm-empty small">No graded picks yet.</div>
                )}
              </div>
              {/* Third grid child: sits under the CLV card in column 1 while the
                  scatter spans both rows in column 2. Fills what used to be a dead
                  column, and puts the explainer beside the chart it describes. */}
              {exch.points?.length ? (
                <div className="fp-mvm-guide">
                  <span className="fp-mvm-guide-hd">How to read the Model vs Market chart</span>
                  <p>Each dot is one graded pick — placed by <b>our win probability</b> (up) against <b>the market's ask</b> (right). The dashed line is where the two agree.</p>
                  <p><b>Example:</b> a dot in the <b>top-left</b> means the model gave a fighter a high chance (~70%) while the market's ask was low (~35%) — a big disagreement in our favor. Dots <b>above</b> the line are picks where the model was more confident than the market; green ones won.</p>
                </div>
              ) : null}
            </div>

            <div className="fp-about-section-hd">By Market (Kalshi)</div>
            {exch.byMarket?.length ? (
              <div className="fp-table-wrap" style={{ marginBottom: "var(--gap)" }}>
                <table className="fp-table">
                  <thead><tr><th>Market</th><th>N</th><th>Hit Rate</th><th>Avg Edge</th><th>ROI (net fees)</th></tr></thead>
                  <tbody>
                    {exch.byMarket.map((r) => (
                      <tr key={r.market}>
                        <td className="fp-td-fight">{r.market}</td>
                        <td>{r.n}</td>
                        <td className={"fp-edge-cell" + (r.hitRate >= 0.5 ? " pos" : "")}>{(r.hitRate * 100).toFixed(1)}%</td>
                        <td className={"fp-edge-cell" + (r.avgEdge >= 0 ? " pos" : " neg")}>{r.avgEdge >= 0 ? "+" : ""}{(r.avgEdge * 100).toFixed(1)}pp</td>
                        <td className={"fp-edge-cell" + (r.roiNetFees >= 0 ? " pos" : " neg")}>{r.roiNetFees >= 0 ? "+" : ""}{(r.roiNetFees * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="fp-mvm-empty small" style={{ marginBottom: "var(--gap)" }}>No graded picks in any market yet.</div>
            )}
          </>
        )}

        <div className="fp-exch-hd">
          <span className="fp-page-ttl" style={{ fontSize: 20 }}>DFS Props Ledger</span>
        </div>
        {!dfs ? (
          <div className="fp-mvm-empty">
            No graded prop picks on this deployment yet. This section fills in once the
            forward ledger (pre-fight log → post-fight grade) has resolved rows to
            aggregate — check back after a card has been logged and graded.
          </div>
        ) : (() => {
          const { overall, edgeBuckets, byMarket, clvSeries } = dfs;
          return (
            <>
              <span className="fp-page-sub" style={{ display: "block", marginBottom: 12 }}>{overall.picks} edge picks (of {overall.graded} graded props) · edge = model probability minus break-even</span>
              <div className="fp-hist-summary">
                <div className="fp-hist-sum-tile"><b style={{ color: "var(--text)" }}>{overall.picks}</b><span>EDGE PICKS</span></div>
                <div className="fp-hist-sum-tile"><b style={{ color: overall.hitRate == null ? "var(--text-faint)" : overall.hitRate >= 0.55 ? "var(--pos)" : "var(--gold-br)" }}>{overall.hitRate != null ? (overall.hitRate * 100).toFixed(1) + "%" : "—"}</b><span>HIT RATE</span></div>
                <div className="fp-hist-sum-tile"><b style={{ color: "var(--m-port)" }}>{overall.avgEdge != null ? (overall.avgEdge >= 0 ? "+" : "") + (overall.avgEdge * 100).toFixed(1) + "pp" : "—"}</b><span>AVG EDGE</span></div>
                <div className="fp-hist-sum-tile">
                  <b style={{ color: overall.beatClose != null && overall.beatClose >= 0.5 ? "var(--pos)" : "var(--text-faint)" }}>
                    {overall.beatClose != null ? (overall.beatClose * 100).toFixed(0) + "%" : "—"}
                  </b>
                  <span>BEAT CLOSE{overall.beatCloseN ? ` (n=${overall.beatCloseN})` : ""}</span>
                </div>
              </div>

              <div className="fp-card" style={{ marginBottom: "var(--gap)", maxWidth: 880 }}>
                <div className="fp-card-hd">
                  <span className="fp-card-ttl">Edge Buckets — Expected vs Realized</span>
                  <span className="fp-card-tag">Model Claim vs Reality</span>
                </div>
                {edgeBuckets?.length ? (
                  <ChartReveal replayKey="mvm-buckets"><EdgeBucketChart buckets={edgeBuckets} /></ChartReveal>
                ) : (
                  <div className="fp-mvm-empty small">Not enough graded picks yet to bucket by edge.</div>
                )}
              </div>

              <div className="fp-card" style={{ marginBottom: "var(--gap)", maxWidth: 880 }}>
                <div className="fp-card-hd">
                  <span className="fp-card-ttl">Closing-Line Value Over Time</span>
                  <span className="fp-card-tag">% Of Moved Lines That Closed In Our Favor</span>
                </div>
                {clvSeries?.length >= 2 ? (
                  <ChartReveal replayKey="mvm-clv"><CLVLineChart series={clvSeries} /></ChartReveal>
                ) : (
                  <div className="fp-mvm-empty small">No closing lines captured yet — the Saturday closing-line job hasn't run against enough cards. This chart fills in once at least two events have a captured close.</div>
                )}
              </div>

              <div className="fp-about-section-hd">By Market</div>
              {byMarket.length ? (
                <div className="fp-table-wrap">
                  <table className="fp-table">
                    <thead><tr><th>Market</th><th>Trust</th><th>N</th><th>Hit Rate</th><th>Avg Edge</th><th>Beat Close</th></tr></thead>
                    <tbody>
                      {byMarket.map((r) => {
                        const tier = LEDGER_TRUST[r.market];
                        return (
                          <tr key={r.market}>
                            <td className="fp-td-fight">{r.market}</td>
                            <td>{tier ? <span className={"fp-tab-trust " + TRUST_CLS[tier]}>{TRUST_LABEL[tier]}</span> : "—"}</td>
                            <td>{r.n}</td>
                            <td className={"fp-edge-cell" + (r.hitRate >= 0.55 ? " pos" : "")}>{(r.hitRate * 100).toFixed(1)}%</td>
                            <td className={"fp-edge-cell" + (r.avgEdge >= 0 ? " pos" : " neg")}>{r.avgEdge >= 0 ? "+" : ""}{(r.avgEdge * 100).toFixed(1)}pp</td>
                            <td>{r.beatClose != null ? (r.beatClose * 100).toFixed(0) + "%" : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="fp-mvm-empty small">No edge picks graded yet in any market.</div>
              )}
            </>
          );
        })()}
        </>
        )}
      </div>
    </div>
  );
}

export function SettingsPage({
  cardSource, setCardSource, cards, selectedCardId, onCardChange,
  payoutKey, setPayoutKey, mult, setMult, breakeven, legs, onRefresh,
  manualForm, setManualForm, referees, weightClasses, onManualPredict, manualLoading,
}) {
  const patchManual = (k, v) => setManualForm((f) => ({ ...f, [k]: v }));
  return (
    <div className="fp-page fp-page-settings">
      <div className="fp-page-hd">
        <div className="fp-page-ttl">Settings</div>
        <span className="fp-page-sub">Card source, event selection, and payout configuration</span>
      </div>
      <div className="fp-page-body">
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: "var(--gap)", maxWidth: 760 }}>
          <div className="fp-card">
            <div className="fp-card-hd"><span className="fp-card-ttl">Card source</span></div>
            <div className="fp-radios" style={{ marginBottom: 14 }}>
              {[["scraped", "Upcoming card (scraped)"], ["manual", "Manual matchup"]].map(([k, lbl]) => (
                <button key={k} className={"fp-radio" + (cardSource === k ? " on" : "")} onClick={() => setCardSource(k)}>
                  <span className="fp-radio-dot"></span>{lbl}
                </button>
              ))}
            </div>
            {cardSource === "scraped" && (<>
              <button className="fp-btn" style={{ width: "100%", marginBottom: 14 }} onClick={onRefresh}>
                {SrcIcon.refresh}Refresh upcoming cards
              </button>
              {cards?.length > 0 && (
                <div className="fp-select">
                  <select value={selectedCardId || ""} onChange={(e) => onCardChange(e.target.value)}>
                    {cards.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
                  </select>
                </div>
              )}
            </>)}
          </div>
          <div className="fp-card fp-payout-card">
            <div className="fp-payout-row">
              <div style={{ flex: 3, minWidth: 0, display: "flex", flexDirection: "column", gap: 4 }}>
                <span className="fp-cgroup-lbl">Payout type</span>
                <div className="fp-select">
                  <select value={payoutKey} onChange={(e) => setPayoutKey(e.target.value)}>
                    {Object.entries(PAYOUTS).map(([k, p]) => <option key={k} value={k}>{p.label}</option>)}
                  </select>
                </div>
              </div>
              <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 4 }}>
                <span className="fp-cgroup-lbl">Multiplier</span>
                <MultInput mult={mult} setMult={setMult} />
              </div>
            </div>
          </div>
        </div>

        {cardSource === "manual" && (
          <div className="fp-manual-section">
            <div className="fp-card" style={{ "--acc": "var(--gold-br)", maxWidth: 760 }}>
              <div className="fp-card-hd">
                <span className="fp-card-ttl">Manual Matchup</span>
                <span className="fp-card-tag">Custom prediction</span>
              </div>
              <form onSubmit={onManualPredict}>
                {/* Corners */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--gap)", marginBottom: "var(--gap)" }}>
                  <div className="fp-cgroup fp-corner-input red">
                    <span className="fp-corner-badge red">
                      <svg viewBox="0 0 8 8" width="7" height="7" style={{ fill: "currentColor" }}><circle cx="4" cy="4" r="4"/></svg>
                      Red corner
                    </span>
                    <input className="fp-input" type="text" value={manualForm.red}
                      placeholder="Fighter name e.g. Islam Makhachev"
                      onChange={(e) => patchManual("red", e.target.value)} required />
                  </div>
                  <div className="fp-cgroup fp-corner-input blue">
                    <span className="fp-corner-badge blue">
                      <svg viewBox="0 0 8 8" width="7" height="7" style={{ fill: "currentColor" }}><circle cx="4" cy="4" r="4"/></svg>
                      Blue corner
                    </span>
                    <input className="fp-input" type="text" value={manualForm.blue}
                      placeholder="Fighter name e.g. Charles Oliveira"
                      onChange={(e) => patchManual("blue", e.target.value)} required />
                  </div>
                </div>

                {/* Secondary fields */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: "var(--gap)", marginBottom: "var(--gap)" }}>
                  <div className="fp-cgroup">
                    <span className="fp-cgroup-lbl">Rounds</span>
                    <div className="fp-toggle">
                      {[3, 5].map((r) => (
                        <button key={r} type="button"
                          className={"fp-toggle-btn" + (manualForm.rounds === r ? " on" : "")}
                          onClick={() => patchManual("rounds", r)}>
                          {r} rds
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="fp-cgroup">
                    <span className="fp-cgroup-lbl">Title fight</span>
                    <div className="fp-toggle">
                      {[["No", false], ["Yes", true]].map(([lbl, val]) => (
                        <button key={lbl} type="button"
                          className={"fp-toggle-btn" + (manualForm.isTitle === val ? " on" : "")}
                          onClick={() => patchManual("isTitle", val)}>
                          {lbl}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="fp-cgroup">
                    <span className="fp-cgroup-lbl">Event date</span>
                    <input className="fp-input" type="date" value={manualForm.eventDate}
                      onChange={(e) => patchManual("eventDate", e.target.value)} />
                  </div>
                  <div className="fp-cgroup">
                    <span className="fp-cgroup-lbl">Weight class</span>
                    <div className="fp-select">
                      <select value={manualForm.weightClass} onChange={(e) => patchManual("weightClass", e.target.value)}>
                        <option value="">Auto-detect</option>
                        {weightClasses.map((wc) => <option key={wc} value={wc}>{wc}</option>)}
                      </select>
                    </div>
                  </div>
                  <div className="fp-cgroup">
                    <span className="fp-cgroup-lbl">Referee</span>
                    <div className="fp-select">
                      <select value={manualForm.referee} onChange={(e) => patchManual("referee", e.target.value)}>
                        <option value="">Any / unknown</option>
                        {referees.map((r) => <option key={r} value={r}>{r}</option>)}
                      </select>
                    </div>
                  </div>
                  <div className="fp-cgroup">
                    <span className="fp-cgroup-lbl">Location</span>
                    <input className="fp-input" type="text" value={manualForm.location}
                      placeholder="e.g. Las Vegas, NV"
                      onChange={(e) => patchManual("location", e.target.value)} />
                  </div>
                </div>

                <button type="submit" className="fp-btn gold" disabled={manualLoading}
                  style={{ width: "100%", padding: "14px", fontSize: 14, letterSpacing: ".5px" }}>
                  {manualLoading ? (
                    <>
                      <svg viewBox="0 0 20 20" style={{ width: 16, height: 16, animation: "spin 1s linear infinite", fill: "none", stroke: "currentColor", strokeWidth: 2 }}>
                        <circle cx="10" cy="10" r="7" strokeDasharray="32" strokeDashoffset="8" strokeLinecap="round"/>
                      </svg>
                      Running prediction…
                    </>
                  ) : (
                    <>
                      <svg viewBox="0 0 20 20" style={{ width: 16, height: 16, fill: "none", stroke: "currentColor", strokeWidth: 2 }}>
                        <polygon points="5,3 17,10 5,17"/>
                      </svg>
                      Run Prediction
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ══ Prop panels ════════════════════════════════════════════════════ */

export function DurationPanel({ fight, line, setLine, side, setSide, breakeven, leg, picks, onToggle }) {
  const acc = "var(--m-dur)";
  const curve = fight.durCurve || [];
  const xMax = fight.schedSec ? fight.schedSec / 60 : fight.rounds * 5;
  const L = line == null ? (fight.medianMin || xMax / 2) : line;
  const pOver = curve.length ? survivalAt2(curve, L) : 0.5;
  const sideP = side === "over" ? pOver : 1 - pOver;
  const myLeg = leg(L, side, sideP);
  const added = picks.some((x) => x.key === myLeg.key);
  const medianMinAnim = useCountUp(fight.medianMin);
  const pDecAnim = useCountUp(fight.pDec != null ? fight.pDec * 100 : fight.pDec);
  const insideAnim = useCountUp(fight.inside != null ? fight.inside * 100 : fight.inside);
  return (
    <div className="fp-tabbody" style={{ "--acc": acc, "--acc-soft": "rgba(54,190,201,.12)", "--line-acc": "rgba(54,190,201,.32)" }}>
      <div className="fp-controls">
        <div className="fp-cgroup"><span className="fp-cgroup-lbl">Line (minutes)</span><LineStepper value={line} onChange={setLine} step={0.25} placeholder={fight.medianMin?.toFixed(1) || "—"} /></div>
        <div className="fp-cgroup"><span className="fp-cgroup-lbl">Side</span><SegToggle value={side} onChange={setSide} options={[{ value: "over", label: "Over" }, { value: "under", label: "Under" }]} /></div>
      </div>
      <div className="fp-cpanel">
        <div className="fp-card">
          <div className="fp-card-hd"><span className="fp-card-ttl">FIGHT DURATION — P({side.toUpperCase()}) CURVE</span><span className="fp-card-tag">SURVIVAL FUNCTION</span></div>
          <ChartReveal replayKey={fight.id}>
            {curve.length > 0
              ? <POverCurve curve={curve} xMax={xMax} lineX={line} side={side} accent={acc} xLabel="DURATION (MINUTES)" lineLabel={line != null ? "@ " + line : ""} />
              : <div className="fp-skeleton" style={{ height: 240, margin: "4px 0" }} />}
          </ChartReveal>
          {fight.durPdf?.length > 0 && (
            <>
              <div className="fp-card-hd" style={{ marginTop: 6 }}><span className="fp-card-ttl">PROBABILITY DENSITY (PDF)</span><span className="fp-card-tag">WHERE IT ENDS</span></div>
              <ChartReveal replayKey={fight.id} delay={0.08}>
                <DurationPDFChart pdfCurve={fight.durPdf} xMax={xMax} lineX={line} pOver={line != null ? pOver : null} accent={acc} />
              </ChartReveal>
            </>
          )}
          <div className="fp-readouts">
            <div className="fp-readout"><span className="fp-readout-val">{fight.medianMin != null ? medianMinAnim.toFixed(1) : "—"}</span><span className="fp-readout-lbl">MEDIAN MINUTES</span></div>
            <div className="fp-readout"><span className="fp-readout-val">{fight.pDec != null ? pDecAnim.toFixed(0) + "%" : "—"}</span><span className="fp-readout-lbl">P(DECISION)</span></div>
            <div className="fp-readout"><span className="fp-readout-val">{fight.inside != null ? insideAnim.toFixed(0) + "%" : "—"}</span><span className="fp-readout-lbl">P(INSIDE DIST.)</span></div>
          </div>
        </div>
        <div className="fp-card">
          <div className="fp-card-hd"><span className="fp-card-ttl">SIDE PROBABILITY @ {L} MIN</span><span className="fp-card-tag">vs {(breakeven * 100).toFixed(1)}% BE</span></div>
          <SideProbPair pOver={pOver} breakeven={breakeven} side={side} />
          <QuantileTable rows={[
            { label: "P(over " + L + " min)",  value: (pOver * 100).toFixed(1) + "%", hl: side === "over" },
            { label: "P(under " + L + " min)", value: ((1 - pOver) * 100).toFixed(1) + "%", hl: side === "under" },
            { label: "Median duration", value: fight.medianMin ? fight.medianMin.toFixed(1) + " min" : "—" },
            ...(fight.durQuantiles || []).map((q) => ({ label: q.label, value: q.value != null ? q.value.toFixed(1) + " min" : "—" })),
          ]} />
          <AddToPortfolio added={added} onClick={() => onToggle(myLeg)} />
        </div>
      </div>
    </div>
  );
}

export function RoundsPanel({ fight, breakeven, picks, onToggle }) {
  const acc = "var(--m-rnd)";
  const durCurve = fight.durCurve || [];
  const [rdLineSel, setRdLine] = useState(null); // null = auto (model median), until user picks a line
  const [rdSide, setRdSide] = useState("over");

  const roundsCurve = useMemo(() => {
    if (!durCurve.length) return [];
    const pts = [];
    for (let r = 0; r <= fight.rounds; r += 0.25) {
      pts.push([r, survivalAt2(durCurve, r * 5)]);
    }
    pts.push([fight.rounds, 0]); // bell-drop mirrors durCurve so survivalCrossing finds median for decision-likely fights
    return pts;
  }, [durCurve, fight.rounds]);

  // Default to the model's predicted median round (curve crossing P=0.5), snapped to
  // the nearest half-integer betting line; falls back to half the scheduled rounds
  // until the curve has loaded.
  const autoLine = useMemo(() => {
    const raw = roundsCurve.length ? survivalCrossing(roundsCurve, 0.5) : null;
    const snapped = raw != null ? Math.round(raw - 0.5) + 0.5 : fight.rounds / 2;
    return Math.min(fight.rounds - 0.5, Math.max(0.5, snapped));
  }, [roundsCurve, fight.rounds]);
  const rdLine = rdLineSel == null ? autoLine : rdLineSel;

  const rdPOver = roundsCurve.length && rdLine != null ? survivalAt2(durCurve, rdLine * 5) : 0.5;
  const rdSideP = rdSide === "over" ? rdPOver : 1 - rdPOver;
  const rdLeg = buildRoundsLeg(fight, rdLine, rdSide, rdSideP);
  const rdAdded = picks.some((x) => x.key === rdLeg.key);

  // Coherent "which round it ends" distribution combining BOTH models. The duration
  // model and the method model each carry their own decision probability and disagree;
  // showing the duration model's round masses raw left a giant DEC-inflated last bar
  // that contradicted the Method tab (e.g. 49% "ends R5" next to KO 57%). Fix: let the
  // method model own the KO/SUB/DEC split (DEC bar = method dec, matches Method tab) and
  // the duration model own only the finish TIMING — rescale its per-round finish masses
  // so they sum to method P(finish)=ko+sub. Bars = [R1..RN finishes, DEC], sum to 100%.
  const roundDistCoherent = useMemo(() => {
    const rd = fight.roundDist;
    if (!rd?.length) return [];
    const m = fight.method || {};
    const pFinish = (m.ko || 0) + (m.sub || 0);
    const pDec = m.dec != null ? m.dec : Math.max(0, 1 - pFinish);
    const finishMasses = rd.slice(0, fight.rounds);
    const finishSum = finishMasses.reduce((a, b) => a + b, 0);
    const scaled = finishSum > 1e-9
      ? finishMasses.map((f) => +((f * pFinish) / finishSum).toFixed(4))
      : finishMasses;
    return [...scaled, +pDec.toFixed(4)];
  }, [fight.roundDist, fight.rounds, fight.method]);

  return (
    <div className="fp-tabbody" style={{ "--acc": acc, "--acc-soft": "rgba(145,131,242,.12)", "--line-acc": "rgba(145,131,242,.32)" }}>
      <div className="fp-controls" style={{ marginBottom: 8 }}>
        <div className="fp-cgroup"><span className="fp-cgroup-lbl">Line (rounds)</span>
          <LineStepper value={rdLineSel} onChange={setRdLine} step={1.0} min={0.5} max={fight.rounds - 0.5} placeholder={rdLine.toFixed(1)} /></div>
        <div className="fp-cgroup"><span className="fp-cgroup-lbl">Side</span>
          <SegToggle value={rdSide} onChange={setRdSide} options={[{ value: "over", label: "Over" }, { value: "under", label: "Under" }]} /></div>
      </div>
      <div className="fp-cpanel">
        <div className="fp-card">
          <div className="fp-card-hd"><span className="fp-card-ttl">FIGHT LENGTH — P({rdSide.toUpperCase()}) CURVE</span><span className="fp-card-tag">ROUNDS O/U</span></div>
          <ChartReveal replayKey={fight.id}>
            {roundsCurve.length > 0
              ? <POverCurve curve={roundsCurve} xMax={fight.rounds} lineX={rdLine} side={rdSide}
                  accent={acc} xLabel="ROUNDS" lineLabel={"@ " + rdLine}
                  xticks={fight.rounds * 2} xTickFmt={(x) => x % 1 === 0 ? Math.round(x) : x.toFixed(1)}
                  height={220} />
              : <div className="fp-skeleton" style={{ height: 220, margin: "4px 0" }} />}
          </ChartReveal>
          {roundDistCoherent.length > 0 && (
            <>
              <div className="fp-card-hd" style={{ marginTop: 6 }}><span className="fp-card-ttl">WHICH ROUND IT ENDS</span><span className="fp-card-tag">FINISH + DECISION</span></div>
              <ChartReveal replayKey={fight.id} delay={0.08}>
                <RoundBars dist={roundDistCoherent} rounds={fight.rounds} accent={acc} />
              </ChartReveal>
            </>
          )}
        </div>
        <div className="fp-card">
          <div className="fp-card-hd"><span className="fp-card-ttl">ROUNDS O/U @ {rdLine}</span><span className="fp-card-tag">vs {(breakeven * 100).toFixed(1)}% BE</span></div>
          <SideProbPair pOver={rdPOver} breakeven={breakeven} side={rdSide} />
          <QuantileTable rows={[
            { label: "P(over " + rdLine + " rds)",  value: (rdPOver * 100).toFixed(1) + "%",       hl: rdSide === "over" },
            { label: "P(under " + rdLine + " rds)", value: ((1 - rdPOver) * 100).toFixed(1) + "%", hl: rdSide === "under" },
          ]} />
          <AddToPortfolio added={rdAdded} onClick={() => onToggle(rdLeg)} label="Add rounds O/U leg" />
        </div>
      </div>
    </div>
  );
}

// Per-market display config: x-axis max, fallback line, stepper step, decimals,
// and whether the quantity is discrete (integer counts) vs continuous (ctrl minutes).
const MK_CFG = {
  sig:     { xMax: 100, line: 19.5, step: 0.5,  dec: 0, discrete: true  },
  r1sig:   { xMax: 60,  line: 14.5, step: 0.5,  dec: 0, discrete: true  },
  bodySig: { xMax: 60,  line: 14.5, step: 0.5,  dec: 0, discrete: true  },
  legSig:  { xMax: 50,  line: 9.5,  step: 0.5,  dec: 0, discrete: true  },
  combo:   { xMax: 160, line: 76.5, step: 0.5,  dec: 0, discrete: true  },
  td:      { xMax: 15,  line: 1.5,  step: 0.5,  dec: 0, discrete: true  },
  r1td:    { xMax: 5,   line: 0.5,  step: 0.5,  dec: 0, discrete: true  },
  subAtt:  { xMax: 6,   line: 0.5,  step: 0.5,  dec: 0, discrete: true  },
  kd:      { xMax: 4,   line: 0.5,  step: 0.5,  dec: 0, discrete: true  },
  ctrl:    { xMax: 15,  line: 1.75, step: 0.25, dec: 1, discrete: false },
};

export function CountPropPanel({ fight, fighter, fighterLast, marketKey, accent, accSoft, accLine, fighterSide, setFighterSide, line, setLine, side, setSide, breakeven, leg, picks, onToggle, aLabel, bLabel, fightLevel = false, propData = null, target = null }) {
  const prop = propData || fighter?.[marketKey];
  const mkt = MARKETS[marketKey] || { label: marketKey, unit: "" };
  const cfg = MK_CFG[marketKey] || { xMax: 100, line: 19.5, step: 0.5, dec: 0, discrete: true };
  const curve = prop?.curve || [];
  const hist = prop?.hist || [];
  const summary = prop?.summary || {};
  // Control time x-axis tracks the scheduled fight length (minutes); others fixed.
  const xMax = marketKey === "ctrl" && fight?.schedSec ? Math.round(fight.schedSec / 60) : cfg.xMax;
  const unit = mkt.unit ? " " + mkt.unit : "";
  const ppLine = (summary?.q?.p50 != null ? summary.q.p50 : cfg.line);
  const L = line == null ? ppLine : line;
  // Discrete counts: snap line to ceiling for the correct integer survival; ctrl is continuous.
  const effectiveL = cfg.discrete ? Math.ceil(L) : L;
  const pOver = curve.length ? survivalAt2(curve, effectiveL) : 0.5;
  const sideP = side === "over" ? pOver : 1 - pOver;
  const myLeg = leg(fighter, fighterLast, marketKey, L, side, sideP, accent);
  const added = picks.some((x) => x.key === myLeg.key);
  const who = fightLevel ? "BOTH FIGHTERS" : fighterLast.toUpperCase();
  const fmtL = (cfg.discrete ? L : Number(L).toFixed(cfg.dec));
  return (
    <div className="fp-tabbody" style={{ "--acc": accent, "--acc-soft": accSoft, "--line-acc": accLine }}>
      <div className="fp-controls">
        {target && (
          <div className="fp-cgroup"><span className="fp-cgroup-lbl">Strike target</span>
            <SegToggle value={target.value} onChange={target.onChange} options={target.options} /></div>
        )}
        {!fightLevel && (
          <div className="fp-cgroup"><span className="fp-cgroup-lbl">Fighter</span>
            <SegToggle value={fighterSide} onChange={setFighterSide} dim options={[{ value: "a", label: aLabel }, { value: "b", label: bLabel }]} /></div>
        )}
        <div className="fp-cgroup"><span className="fp-cgroup-lbl">Line{mkt.unit ? " (" + mkt.unit + ")" : ""}</span><LineStepper value={line} onChange={setLine} step={cfg.step} placeholder={ppLine.toFixed(cfg.dec)} /></div>
        <div className="fp-cgroup"><span className="fp-cgroup-lbl">Side</span><SegToggle value={side} onChange={setSide} options={[{ value: "over", label: "Over" }, { value: "under", label: "Under" }]} /></div>
      </div>
      {mkt.trust === "CUT" && (
        <div className="fp-card-note" style={{ marginBottom: 12 }}>This market has not beaten closing lines historically — treat edges here as low-signal.</div>
      )}
      {marketKey === "bodySig" && (
        <div className="fp-card-note" style={{ marginBottom: 12 }}>Trust is validated at the 4.5-strike line only; other body-strike lines are unproven.</div>
      )}
      <div className="fp-cpanel">
        <div className="fp-card">
          <div className="fp-card-hd"><span className="fp-card-ttl">{who} — {mkt.label.toUpperCase()} · P({side.toUpperCase()})</span><span className="fp-card-tag">SURVIVAL CURVE</span></div>
          <ChartReveal replayKey={`${fight.id}:${marketKey}`}>
            {curve.length > 0
              ? <POverCurve curve={curve} xMax={xMax} lineX={line} side={side} accent={accent} xLabel={(mkt.label + unit).toUpperCase()} lineLabel={line != null ? "@ " + fmtL + unit : ""} discrete={cfg.discrete} />
              : prop
                ? <div className="fp-skeleton" style={{ height: 200, margin: "4px 0" }} />
                : <div className="fp-prop-unavail" style={{ height: 200, margin: "4px 0", display: "grid", placeItems: "center", textAlign: "center", color: "var(--text-faint)", border: "1px dashed var(--line-2)", borderRadius: 10, padding: 16, gap: 6 }}>
                    <span style={{ fontSize: 22, opacity: .5 }}>—</span>
                    <span style={{ fontSize: 13, color: "var(--text-dim)" }}>{mkt.label} not available for this matchup</span>
                    <span style={{ fontSize: 11 }}>The prediction API hasn't returned this market. If it persists, the deployed API needs the latest model build.</span>
                  </div>}
          </ChartReveal>
          {hist.length > 0 && (
            <>
              <div className="fp-card-hd" style={{ marginTop: 4 }}><span className="fp-card-ttl" style={{ color: "var(--text-dim)" }}>Sample distribution</span></div>
              <ChartReveal replayKey={`${fight.id}:${marketKey}`} delay={0.08}>
                <CountHistogram hist={hist} lineX={line} accent={accent} xMax={xMax} />
              </ChartReveal>
            </>
          )}
        </div>
        <div className="fp-card">
          <div className="fp-card-hd"><span className="fp-card-ttl">SIDE PROBABILITY @ {fmtL}{unit}</span><span className="fp-card-tag">vs {(breakeven * 100).toFixed(1)}% BE</span></div>
          <SideProbPair pOver={pOver} breakeven={breakeven} side={side} />
          <QuantileTable rows={[
            { label: "Median",    value: summary?.q?.p50 != null ? summary.q.p50.toFixed(cfg.dec) + unit : "—", hl: true },
            { label: "Mean",      value: summary?.mean != null ? summary.mean.toFixed(Math.max(1, cfg.dec)) + unit : "—" },
            { label: "25th pct",  value: summary?.q?.p25 != null ? summary.q.p25.toFixed(cfg.dec) + unit : "—" },
            { label: "75th pct",  value: summary?.q?.p75 != null ? summary.q.p75.toFixed(cfg.dec) + unit : "—" },
            ...(marketKey !== "td" ? [{ label: "P(zero)", value: summary?.p0 != null ? (summary.p0 * 100).toFixed(1) + "%" : "—" }] : []),
          ]} />
          <AddToPortfolio added={added} onClick={() => onToggle(myLeg)} />
        </div>
      </div>
    </div>
  );
}

export function FinishesPanel({ fight, breakeven, picks, onToggle }) {
  const acc = "var(--m-r1)";
  const [fighter, setFighter] = useState("a");
  const aL = lastName(fight.a.name);
  const bL = lastName(fight.b.name);
  const f = fighter === "a" ? fight.a : fight.b;
  const fLast = fighter === "a" ? aL : bL;

  const finishData = f.finish || {};
  const finishMarkets = [
    { key: "finish",     label: "Any finish",  mAcc: "var(--m-r1)",  p: finishData.finish     ?? null },
    { key: "ko_finish",  label: "KO / TKO",    mAcc: "var(--m-sig)", p: finishData.ko_finish  ?? null },
    { key: "sub_finish", label: "Submission",   mAcc: "var(--m-td)",  p: finishData.sub_finish ?? null },
  ];

  const roundDist = fight.roundDist || [];
  const roundFinishes = [];
  for (let r = 0; r < fight.rounds; r++) {
    const rp = roundDist[r];
    if (rp == null) continue;
    roundFinishes.push({ round: r + 1, p: +(f.pWin * rp).toFixed(4) });
  }

  return (
    <div className="fp-tabbody" style={{ "--acc": acc, "--acc-soft": "rgba(236,97,73,.12)", "--line-acc": "rgba(236,97,73,.3)" }}>
      <div className="fp-controls" style={{ marginBottom: 8 }}>
        <div className="fp-cgroup"><span className="fp-cgroup-lbl">Fighter</span>
          <SegToggle value={fighter} onChange={setFighter} dim options={[{ value: "a", label: aL }, { value: "b", label: bL }]} /></div>
      </div>
      <div className="fp-cpanel">
        <div className="fp-card">
          <div className="fp-card-hd"><span className="fp-card-ttl">{fLast.toUpperCase()} — FINISH MARKETS</span><span className="fp-card-tag">OVER 0.5</span></div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 8 }}>
            {finishMarkets.map(({ key, label, mAcc, p }) => {
              if (p == null) return null;
              const edge = p - breakeven;
              const leg = buildFinishLeg(fight, fLast, key, p);
              const added = picks.some((x) => x.key === leg.key);
              return (
                <div key={key} style={{ background: "var(--panel-3)", borderRadius: 11, padding: "13px 15px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
                    <span style={{ fontFamily: "var(--f-ui)", fontWeight: 700, fontSize: 12, color: "var(--text-dim)" }}>{label}</span>
                    <span style={{ fontFamily: "var(--f-disp)", fontWeight: 800, fontSize: 20, color: mAcc }}>{(p * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 4, background: "var(--panel-hi)", borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
                    <div style={{ height: "100%", width: Math.min(100, p * 100) + "%", background: mAcc, borderRadius: 3, transition: "width .55s cubic-bezier(.2,.7,.2,1)" }} />
                  </div>
                  <span style={{ fontFamily: "var(--f-mono)", fontSize: 11, color: edge >= 0 ? "var(--pos)" : "var(--neg)", display: "block", marginBottom: 8 }}>
                    {edge >= 0 ? "+" : ""}{(edge * 100).toFixed(1)}% vs {(breakeven * 100).toFixed(1)}% BE
                  </span>
                  <AddToPortfolio added={added} onClick={() => onToggle(leg)} />
                </div>
              );
            })}
          </div>
        </div>
        <div className="fp-card">
          <div className="fp-card-hd"><span className="fp-card-ttl">{fLast.toUpperCase()} — ROUND FINISH</span><span className="fp-card-tag">{fight.rounds}R BOUT</span></div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 8 }}>
            {roundFinishes.length > 0 ? roundFinishes.map(({ round, p }) => {
              const edge = p - breakeven;
              const leg = buildFinishRndLeg(fight, fLast, round, p);
              const added = picks.some((x) => x.key === leg.key);
              return (
                <div key={round} style={{ background: "var(--panel-3)", borderRadius: 11, padding: "13px 15px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
                    <span style={{ fontFamily: "var(--f-ui)", fontWeight: 700, fontSize: 12, color: "var(--text-dim)" }}>Round {round} finish</span>
                    <span style={{ fontFamily: "var(--f-disp)", fontWeight: 800, fontSize: 20, color: acc }}>{(p * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 4, background: "var(--panel-hi)", borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
                    <div style={{ height: "100%", width: Math.min(100, p * 100) + "%", background: acc, borderRadius: 3, transition: "width .55s cubic-bezier(.2,.7,.2,1)" }} />
                  </div>
                  <span style={{ fontFamily: "var(--f-mono)", fontSize: 11, color: edge >= 0 ? "var(--pos)" : "var(--neg)", display: "block", marginBottom: 8 }}>
                    {edge >= 0 ? "+" : ""}{(edge * 100).toFixed(1)}% vs {(breakeven * 100).toFixed(1)}% BE
                  </span>
                  <AddToPortfolio added={added} onClick={() => onToggle(leg)} />
                </div>
              );
            }) : (
              <div style={{ color: "var(--text-faint)", fontFamily: "var(--f-mono)", fontSize: 12, padding: "40px 0", textAlign: "center" }}>No round data</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ══ About Page ══════════════════════════════════════════════════════ */

const ABOUT_MODELS = [
  {
    key: "winner",
    accent: "var(--gold-br)",
    label: "Winner Prediction",
    tag: "DIVERSE ENSEMBLE",
    body: "An 8-member diverse ensemble — 3×LightGBM (seeds 42/43/44), 2×CatBoost, 2×XGBoost, and a ratings-only Logistic member — with SLSQP-optimal OOF blend weights, isotonic calibration, Platt refinement, and a smooth probability soft-cap whose ceiling is ECE-tuned. Rather than hard-clipping every confident pick to an identical maximum, the soft-cap saturates gently — so a stronger favorite still reads slightly higher than a marginal one while staying within the calibrated ceiling. Features include striking output, takedown rates, finishing ability, grappling control, specialist amplifiers, era baselines, and recent form. Debut fighters are seeded from pre-UFC career records: Elo, Glicko-2, and TrueSkill are initialized from pre-UFC win rate instead of flat 1500, eliminating the 50/50 cold-start for first-time UFC fighters.",
  },
  {
    key: "method",
    accent: "var(--m-port)",
    label: "Finish Method",
    tag: "3-CLASS LGBM",
    body: "A single LightGBM multinomial classifier that outputs P(KO/TKO), P(Submission), and P(Decision) directly. Calibrated with temperature scaling and prior shrinkage toward the modern 36-month base rate — preventing any class from collapsing to near-zero. On the served production tier the temperature is fit on temporal out-of-fold predictions (models that never saw those rows), so calibration is never tuned on data the model trained on. Specialist features for grapplers, wrestlers, and knockout artists amplify the signal.",
  },
  {
    key: "duration",
    accent: "var(--m-dur)",
    label: "Fight Duration",
    tag: "MIXTURE CDF",
    body: "A two-stage hurdle: Stage 1 predicts P(decision) via a calibrated LightGBM classifier; Stage 2 fits an 11-quantile regression on log-finish-seconds for non-decision fights. These are combined into a mixture CDF weighted by the method model's output, producing a single continuous survival curve that powers the P(over X minutes) chart and round-by-round finish distribution.",
  },
  {
    key: "props",
    accent: "var(--m-sig)",
    label: "Count Props",
    tag: "RATE × DURATION",
    body: "Nine count markets — sig strikes (full fight, R1, body, leg, and combined), takedowns (full fight and R1), control time, knockdowns, and submission attempts — each use a hurdle model on per-minute rate (not raw count), avoiding the regime-mixing problem where the model confuses 'how active is this fighter' with 'how long does the fight last.' At inference, Monte Carlo integration over the duration CDF turns each rate distribution into a count CDF; because real counts are whole numbers, the over/under at a half-integer line is priced coherently against the integer-count hurdle rather than the continuous draw. A separate finish layer (KO, submission, and by-round) is read directly from the joint method-and-duration simulation.",
  },
];

const ABOUT_METRICS = [
  { value: "66.5%",    label: "WINNER ACCURACY",   accent: "var(--pos)",     note: "held-out test set 2025–2026 (n=783)" },
  { value: "8",        label: "ENSEMBLE MEMBERS",  accent: "var(--gold-br)", note: "3×LGBM + 2×CatBoost + 2×XGB + Logistic" },
  { value: "13",       label: "PROP MARKETS",       accent: "var(--m-dur)",   note: "duration, rounds, sig/R1/body/leg/combined strikes, TDs, R1 TDs, sub att, control, knockdowns, finishes" },
  { value: "Platt",    label: "CALIBRATION",        accent: "var(--m-port)",  note: "isotonic → Platt + soft-cap (winner model)" },
  { value: "KS + BSS", label: "VALIDATION TESTS",  accent: "var(--m-sig)",   note: "per-market distribution checks" },
];

const ABOUT_TAPE = [
  ["Training window", "Frozen 2010–2023", "Rolling — through the latest card"],
  ["Test set", "2025–2026 held-out (n=783)", "None, by construction"],
  ["What it's for", "Every Gate A–D check + every accuracy figure on this page", "The picks you actually see"],
  ["Where it's graded", "Backtest on fights it never trained on", "Forward-only — History logs it pre-fight, grades it after"],
];

export function AboutPage() {
  return (
    <div className="fp-page fp-page-about">
      <div className="fp-page-hd">
        <div className="fp-page-ttl">About</div>
        <span className="fp-page-sub">How FightPath works — models, data, and validation</span>
      </div>
      <div className="fp-page-body fp-about-typeset">

        {/* Lead-in: the pipeline in action, before any of the explaining */}
        <div className="fp-about-leadin">
          <span className="fp-about-leadin-label">How it plays out</span>
          <p>Pereira vs. Adesanya is a likely KO finish → the method model raises P(KO) → the duration mixture shifts earlier → P(over 4.5 sig strikes) falls, because neither fighter is expected to go the distance.</p>
        </div>

        {/* Intro */}
        <div className="fp-about-intro">
          <p>FightPath is a statistical prediction engine for UFC fights. It does not rely on hot takes, consensus, or betting market lines — every number you see is generated by machine learning models trained exclusively on historical fight data.</p>
          <p className="fp-about-pullquote">The goal: surface an independent signal — where the model's probability differs meaningfully from the market's.</p>
        </div>

        {/* Key metrics */}
        <div className="fp-about-section-hd">Performance at a glance</div>
        <div className="fp-about-metrics">
          {ABOUT_METRICS.map((m) => (
            <div key={m.label} className="fp-about-metric">
              <b style={{ color: m.accent }}>{m.value}</b>
              <span className="fp-about-metric-lbl">{m.label}</span>
              <span className="fp-about-metric-note">{m.note}</span>
            </div>
          ))}
        </div>

        {/* Model cards */}
        <div className="fp-about-section-hd">The models</div>
        <div className="fp-about-model-grid">
          {ABOUT_MODELS.map((m) => (
            <div key={m.key} className="fp-about-model-card" style={{ "--model-acc": m.accent }}>
              <div className="fp-about-model-bar" />
              <div className="fp-about-model-hd">
                <span className="fp-about-model-name">{m.label}</span>
                <span className="fp-about-model-tag">{m.tag}</span>
              </div>
              <p className="fp-about-model-body">{m.body}</p>
            </div>
          ))}
        </div>

        {/* Pipeline dependency */}
        <div className="fp-about-section-hd">How the models connect</div>
        <div className="fp-about-pipeline">
          <div className="fp-about-pipeline-intro">
            The winner ensemble (8 members: 3×LightGBM, 2×CatBoost, 2×XGBoost, 1×Logistic), method classifier, 2 duration hurdle stages, and 9 count-prop hurdle models all feed into one another in a strict pipeline. This means every prop you see is internally consistent: a single change in one model ripples through all downstream outputs.
          </div>
          <div className="fp-about-pipeline-flow">
            <div className="fp-about-pipeline-step">
              <div className="fp-about-pipeline-dot" style={{ background: "var(--gold-br)" }} />
              <div>
                <span className="fp-about-pipeline-lbl">Method model</span>
                <span className="fp-about-pipeline-desc">outputs P(KO/TKO), P(SUB), P(DEC) for the matchup</span>
              </div>
            </div>
            <div className="fp-about-pipeline-connector" aria-hidden="true"><span className="fp-about-pipeline-pulse" /></div>
            <div className="fp-about-pipeline-step">
              <div className="fp-about-pipeline-dot" style={{ background: "var(--m-dur)" }} />
              <div>
                <span className="fp-about-pipeline-lbl">Duration model</span>
                <span className="fp-about-pipeline-desc">uses method probabilities to weight finish vs. decision CDFs — a fight predicted to end by KO shifts the mixture toward shorter durations</span>
              </div>
            </div>
            <div className="fp-about-pipeline-connector" aria-hidden="true"><span className="fp-about-pipeline-pulse" style={{ animationDelay: "-1.3s" }} /></div>
            <div className="fp-about-pipeline-step">
              <div className="fp-about-pipeline-dot" style={{ background: "var(--m-sig)" }} />
              <div>
                <span className="fp-about-pipeline-lbl">Count prop models</span>
                <span className="fp-about-pipeline-desc">integrate each fighter's per-minute rate distribution over the duration CDF — a shorter expected fight directly compresses projected strike and takedown counts</span>
              </div>
            </div>
          </div>
        </div>

        {/* Two model tiers — staged as a matchup, the app's own structural device */}
        <div className="fp-about-section-hd">The two tiers</div>
        <div className="fp-about-pipeline-intro" style={{ marginBottom: 14 }}>
          The same code trains two models from one pipeline. What you are graded against and what actually predicts your card are deliberately kept separate — so the accuracy claims stay honest while the live picks use every fight ever recorded.
        </div>
        <div className="fp-about-face">
          <div className="fp-about-face-hd">
            <div className="fp-about-face-side red">
              <span className="fp-about-face-nick">The yardstick</span>
              <span className="fp-about-face-name">Eval Tier</span>
            </div>
            <div className="fp-about-face-center"><span className="fp-about-face-vs">VS</span></div>
            <div className="fp-about-face-side blue">
              <span className="fp-about-face-nick">The served fighter</span>
              <span className="fp-about-face-name">Prod Tier</span>
            </div>
          </div>
          <div className="fp-about-face-tape">
            {ABOUT_TAPE.map(([label, red, blue]) => (
              <div className="fp-about-face-row" key={label}>
                <span className="fp-about-face-val red">{red}</span>
                <span className="fp-about-face-lbl">{label}</span>
                <span className="fp-about-face-val blue">{blue}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Data + Testing two-col */}
        <div className="fp-about-2col">
          <div className="fp-port-how">
            <h3>Data &amp; Features</h3>
            <p>Raw data comes from UFCStats, the sport's canonical record. On the evaluation tier the <b>training window is 2010–2023</b>; the model never sees validation (2024 H1/H2) or test (2025–2026) data during development (the production tier later retrains on all of it). Career feature computation draws on earlier history for fighters active in the training window. Pre-UFC career records (wins/losses across all promotions) seed rating systems for debut fighters.</p>
            <p style={{ marginTop: 8 }}><b>583 features total</b> — ~70 raw per-fighter stats from UFCStats, ~513 engineered. Key groups:</p>
            <ul>
              <li><b>Striking &amp; grappling</b> — career averages, per-round breakdowns (R1 pace vs. late output), R1 activity burst rates</li>
              <li><b>Win/finish rates</b> — KO%, SUB%, DEC%, recency-weighted; specialist amplifiers for elite grapplers and KO artists</li>
              <li><b>Injury-aware labels</b> — freak-injury stoppages (22 curated cases across UFC history, e.g. a knee giving out or an arm dislocated bracing a fall) aren't counted as genuine knockouts or submissions: no finish credit for the winner, no durability penalty for the loser, and damped rating transfer; new injury stoppages are classified automatically each week</li>
              <li><b>Ratings</b> — Elo, Glicko-2 (with uncertainty-scaled diff), TrueSkill; seeded from pre-UFC career records for debutants</li>
              <li><b>Physical</b> — age, height, reach, weight-class context; era baselines so 2010 fight stats compare fairly to 2024</li>
              <li><b>Contextual</b> — common-opponent transitivity, referee stoppage history (242 referees), location</li>
            </ul>
            <p style={{ marginTop: 8 }}>Sportsbook odds and money lines are <b>never used as features</b> — the model is intentionally market-blind, so every probability is derived purely from fight history.</p>
          </div>
          <div className="fp-port-how">
            <h3>Testing &amp; Validation</h3>
            <p>On the evaluation tier all splits are <b>chronological</b> — no shuffling — so no future information leaks into training. Train: 2010–2023 · Val-A: 2024 H1 · Val-B: 2024 H2 · Test: 2025–2026. Prop models face distributional tests in addition to accuracy:</p>
            <ul>
              <li><b>66.5% accuracy</b> — winner model on 783 held-out fights (2025–2026)</li>
              <li><b>Brier Skill Score</b> — probabilistic accuracy vs a naive baseline</li>
              <li><b>KS test</b> — checks that predicted distributions match observed outcomes</li>
              <li><b>Calibration (ECE)</b> — a model saying 70% should be right ~70% of the time</li>
              <li><b>Gate system</b> — every model must pass A/B/C/D quality thresholds before shipping</li>
              <li><b>Forward ledger</b> — production picks are logged at real sportsbook lines <em>before</em> each card and graded after, the only bias-free record of live performance</li>
            </ul>
          </div>
        </div>

        {/* Limitations */}
        <div className="fp-about-section-hd">LIMITATIONS &amp; DISCLAIMERS</div>
        <div className="fp-port-how" style={{ maxWidth: 720 }}>
          <p>FightPath is a statistical tool, not a crystal ball. It cannot account for late camp changes, injuries that aren't public, or the inherent randomness of combat sports. A 65% win probability means the model expects that fighter to win — not that they will. Always treat model output as one input among many, and never bet more than you can afford to lose.</p>
          <p style={{ marginTop: 8 }}>Prop probabilities reflect the model's distribution over possible fight outcomes, not guarantees. Edge figures compare model probability to break-even implied by the selected payout structure — a positive edge is a statistical expectation, not a sure thing.</p>
        </div>

      </div>
    </div>
  );
}

/* ══ Bottom navigation bar (mobile ≤ 640px) ═════════════════════════ */
export function BottomNav({ nav, setNav, picks, cardSource }) {
  // Positions is card-only — hidden for manual matchups (no live-line targets).
  const showPositions = cardSource !== "manual";
  const items = [
    { key: "card",      label: "Fight Card" },
    { key: "proplab",   label: "Prop Lab"   },
    ...(showPositions ? [{ key: "positions", label: "Positions" }] : []),
    { key: "portfolio", label: "Portfolio"  },
    { key: "market",    label: "Vs Market"  },
    { key: "about",     label: "About"      },
    { key: "settings",  label: "Settings"   },
  ];
  return (
    <nav className="fp-bottomnav" aria-label="Main navigation">
      {items.map(({ key, label }) => (
        <button
          key={key}
          className={"fp-bn-item" + (nav === key ? " on" : "")}
          onClick={() => setNav(key)}
          aria-label={label}
          aria-current={nav === key ? "page" : undefined}
        >
          <span className="fp-bn-ic">{NavIcon[key]}</span>
          <span>{label}</span>
          {key === "portfolio" && picks.length > 0 && (
            <span className="fp-bn-badge">{picks.length}</span>
          )}
        </button>
      ))}
    </nav>
  );
}
