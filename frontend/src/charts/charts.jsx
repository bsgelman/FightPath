import { useMemo, useId, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import { survivalCrossing, survivalAt2, wilsonInterval } from "../lib/stats.js";
import { traceTransition, fadeTransition } from "./motion.js";

export { survivalAt2 };

/* ── Title reign — cumulative winner accuracy across events ──────── */
// series: [{date, event, correct, total, live}], oldest first. gateFloor = the
// model's published accuracy target, drawn as a reference line. x-axis is event
// index (cards are irregularly spaced in real time, so a calendar axis would
// compress unevenly). height defaults low — this mounts full-width (not in a
// narrow two-column panel like the other charts here), so the viewBox aspect
// ratio needs to stay wide or it upscales into a wall.
export function TitleReignChart({ series, gateFloor = 0.64, accent = "var(--gold-br)", height = 150 }) {
  const W = 560, H = height, padL = 38, padR = 16, padB = 28, padT = 20;
  const reduced = useReducedMotion();
  const [hover, setHover] = useState(null);
  const n = series?.length || 0;

  const pts = useMemo(() => {
    let cumCorrect = 0, cumTotal = 0;
    return (series || []).map((ev, i) => {
      cumCorrect += ev.correct; cumTotal += ev.total;
      const rate = cumTotal ? cumCorrect / cumTotal : 0;
      const [lo, hi] = wilsonInterval(cumCorrect, cumTotal);
      return { i, date: ev.date, event: ev.event, rate, lo, hi, live: !!ev.live };
    });
  }, [series]);

  if (n < 2) return null;

  const yMin = 0.45, yMax = 0.85;
  const clampY = (p) => Math.min(yMax, Math.max(yMin, p));
  const sx = (i) => padL + (i / (n - 1)) * (W - padL - padR);
  const sy = (p) => padT + (1 - (clampY(p) - yMin) / (yMax - yMin)) * (H - padB - padT);

  const linePath = "M " + pts.map((p) => `${sx(p.i).toFixed(1)} ${sy(p.rate).toFixed(1)}`).join(" L ");
  const bandPath = "M " + pts.map((p) => `${sx(p.i).toFixed(1)} ${sy(p.hi).toFixed(1)}`).join(" L ")
    + " L " + pts.slice().reverse().map((p) => `${sx(p.i).toFixed(1)} ${sy(p.lo).toFixed(1)}`).join(" L ") + " Z";

  const seamIdx = pts.findIndex((p) => p.live);

  const handleMove = (e) => {
    const svgEl = e.currentTarget;
    const rect = svgEl.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const rawI = ((mx - padL) / (W - padL - padR)) * (n - 1);
    if (rawI < -0.5 || rawI > n - 0.5) { setHover(null); return; }
    setHover(Math.round(Math.min(n - 1, Math.max(0, rawI))));
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
         aria-label="Cumulative winner accuracy across events"
         style={{ width: "100%", cursor: "crosshair" }}
         onMouseMove={handleMove} onMouseLeave={() => setHover(null)}>
      {[0.5, 0.6, 0.7, 0.8].map((g) => (
        <g key={g}>
          <line x1={padL} y1={sy(g)} x2={W - padR} y2={sy(g)} className="fp-grid" />
          <text x={padL - 8} y={sy(g) + 3} className="fp-axis" textAnchor="end">{(g * 100).toFixed(0)}%</text>
        </g>
      ))}
      <line x1={padL} y1={sy(gateFloor)} x2={W - padR} y2={sy(gateFloor)}
        stroke="var(--gold-br)" strokeWidth="1.5" strokeDasharray="5 4" opacity="0.65" />
      <rect x={padL} y={padT - 15} width="145" height="13" fill="var(--panel-2)" opacity="0.85" />
      <text x={padL} y={padT - 5} className="fp-axis" textAnchor="start"
        fill="var(--gold-br)" style={{ fontWeight: 700 }}>ACCURACY TARGET {(gateFloor * 100).toFixed(0)}%</text>

      <path d={bandPath} fill={accent} fillOpacity="0.12" />
      {reduced ? (
        <path d={linePath} fill="none" stroke={accent} strokeWidth="2.4" strokeLinejoin="round" />
      ) : (
        <motion.path d={linePath} fill="none" stroke={accent} strokeWidth="2.4" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={traceTransition(reduced)} />
      )}

      {seamIdx > 0 && (
        <g>
          <line x1={sx(seamIdx)} y1={padT} x2={sx(seamIdx)} y2={H - padB}
            stroke="var(--gold-br)" strokeWidth="1" strokeDasharray="2 3" opacity="0.7" />
          <text x={sx(seamIdx)} y={padT - 8} className="fp-axis" fill="var(--gold-br)"
            textAnchor="middle" style={{ fontWeight: 700, fontSize: 8 }}>LIVE</text>
        </g>
      )}

      {hover != null && pts[hover] && (() => {
        const p = pts[hover];
        const tx = Math.min(sx(p.i) + 8, W - 148);
        const ty = Math.max(sy(p.rate) - 36, padT + 2);
        return (
          <g>
            <line x1={sx(p.i)} y1={padT} x2={sx(p.i)} y2={H - padB} stroke="rgba(255,255,255,.2)" strokeWidth="1" strokeDasharray="2 3" />
            <circle cx={sx(p.i)} cy={sy(p.rate)} r="4" fill={accent} stroke="var(--bg)" strokeWidth="1.5" />
            <g transform={`translate(${tx}, ${ty})`}>
              <rect x="0" y="-9" width="144" height="42" rx="3" fill="var(--panel-2)" stroke="rgba(255,255,255,.14)" strokeWidth="1" />
              <text x="6" y="2" className="fp-axis" fill="var(--text)" style={{ fontWeight: 600, fontSize: 9 }}>{(p.event || "").slice(0, 22)}</text>
              <text x="6" y="14" className="fp-axis" fill={accent} style={{ fontWeight: 700, fontSize: 9 }}>{(p.rate * 100).toFixed(1)}% cumulative</text>
              <text x="6" y="26" className="fp-axis" fill="var(--text-faint)" style={{ fontSize: 8 }}>{p.date}{p.live ? " · LIVE" : ""}</text>
            </g>
          </g>
        );
      })()}
    </svg>
  );
}

/* ── P(over) survival curve — counts or duration ─────────────────── */
export function POverCurve({ curve, xMax, lineX, side, accent, height = 280, xLabel, lineLabel, xticks = 6, xTickFmt, discrete = false }) {
  const W = 560, H = height, padL = 44, padR = 16, padB = 34, padT = 16;
  const uid = useId().replace(/:/g, "");
  const tickFmt = xTickFmt || ((x) => x % 1 === 0 ? Math.round(x) : x.toFixed(1));
  const [hover, setHover] = useState(null); // {x, pOver, pUnder}
  const reduced = useReducedMotion();

  const sx = (x) => padL + (x / xMax) * (W - padL - padR);
  const sy = (p) => padT + (1 - p) * (H - padB - padT);
  const showUnder = side === "under";

  // For discrete props: build step-function path sampled at integers
  // P(X > x) = P(X >= floor(x)+1) = survivalAt2(curve, floor(x)+1) for integer-valued X
  const { linePath, areaPath } = useMemo(() => {
    if (!discrete || !curve.length) {
      const plot = curve.map(([x, p]) => [x, showUnder ? 1 - p : p]);
      const lp = "M " + plot.map(([x, p]) => `${sx(x).toFixed(1)} ${sy(p).toFixed(1)}`).join(" L ");
      const ap = `M ${sx(0)} ${H - padB} ` + plot.map(([x, p]) => `L ${sx(x).toFixed(1)} ${sy(p).toFixed(1)}`).join(" ") + ` L ${sx(plot[plot.length - 1]?.[0] ?? 0)} ${H - padB} Z`;
      return { linePath: lp, areaPath: ap };
    }
    // Step function: for x in [n, n+1), survival = S(n+1)
    const steps = [];
    const iMax = Math.ceil(xMax);
    for (let n = 0; n <= iMax; n++) {
      const s = survivalAt2(curve, n + 1);
      const p = showUnder ? 1 - s : s;
      const x0 = sx(Math.min(n, xMax)), x1 = sx(Math.min(n + 1, xMax));
      steps.push([x0, p], [x1, p]);
    }
    const lp = "M " + steps.map(([x, p]) => `${x.toFixed(1)} ${sy(p).toFixed(1)}`).join(" L ");
    const ap = `M ${steps[0][0].toFixed(1)} ${H - padB} ` + steps.map(([x, p]) => `L ${x.toFixed(1)} ${sy(p).toFixed(1)}`).join(" ") + ` L ${steps[steps.length - 1][0].toFixed(1)} ${H - padB} Z`;
    return { linePath: lp, areaPath: ap };
  }, [curve, discrete, showUnder, xMax]);

  const median = survivalCrossing(curve, 0.5);
  // For discrete: snap line probability to ceiling (P(X > 0.5) = P(X >= 1) = S(1))
  const effectiveLineX = discrete && lineX != null ? Math.ceil(lineX) : lineX;
  const pOverAtLine = effectiveLineX != null ? survivalAt2(curve, effectiveLineX) : null;
  const pAtLine = pOverAtLine != null ? (showUnder ? 1 - pOverAtLine : pOverAtLine) : null;

  const handleMouseMove = (e) => {
    const svgEl = e.currentTarget;
    const rect = svgEl.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const xVal = (mx - padL) / (W - padL - padR) * xMax;
    if (xVal < 0 || xVal > xMax) { setHover(null); return; }
    // For discrete: snap to integer boundary
    const snapX = discrete ? Math.floor(xVal) + 1 : xVal;
    const pOv = survivalAt2(curve, snapX);
    setHover({ x: xVal, snapX, pOver: pOv, pUnder: 1 - pOv });
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
         aria-label={`P(${side}) survival curve — ${xLabel}`}
         style={{ width: "100%", cursor: "crosshair" }}
         onMouseMove={handleMouseMove} onMouseLeave={() => setHover(null)}>
      <defs>
        <linearGradient id={`g${uid}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={accent} stopOpacity="0.34" />
          <stop offset="1" stopColor={accent} stopOpacity="0.02" />
        </linearGradient>
        {!reduced && (
          <clipPath id={`clip${uid}`}>
            <motion.rect x={padL} y={padT} height={H - padB - padT}
              initial={{ width: 0 }} animate={{ width: W - padL - padR }}
              transition={traceTransition(reduced)} />
          </clipPath>
        )}
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={padL} y1={sy(g)} x2={W - padR} y2={sy(g)} className="fp-grid" strokeDasharray={g === 0.5 ? "4 4" : ""} />
          <text x={padL - 8} y={sy(g) + 3} className="fp-axis" textAnchor="end">{(g * 100).toFixed(0)}%</text>
        </g>
      ))}
      {Array.from({ length: xticks + 1 }).map((_, i) => {
        const x = (xMax * i) / xticks;
        return <text key={i} x={sx(x)} y={H - 12} className="fp-axis" textAnchor="middle">{tickFmt(x)}</text>;
      })}
      <text x={(padL + W - padR) / 2} y={H - 1} className="fp-axis-ttl" textAnchor="middle">{xLabel}</text>
      <path d={areaPath} fill={`url(#g${uid})`} clipPath={reduced ? undefined : `url(#clip${uid})`} />
      {reduced ? (
        <path d={linePath} fill="none" stroke={accent} strokeWidth="2.4" strokeLinejoin="round" />
      ) : (
        <motion.path d={linePath} fill="none" stroke={accent} strokeWidth="2.4" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={traceTransition(reduced)} />
      )}
      {median != null && (
        <g>
          <line x1={sx(median)} y1={padT} x2={sx(median)} y2={H - padB} stroke="var(--text-faint)" strokeWidth="1" strokeDasharray="3 4" />
          <text x={sx(median) > W - padR - 50 ? sx(median) - 5 : sx(median) + 5} y={padT + 10} className="fp-axis" fill="var(--text-dim)" textAnchor={sx(median) > W - padR - 50 ? "end" : "start"} style={{ fontSize: 8 }}>med {median.toFixed(median < 10 ? 1 : 0)}</text>
        </g>
      )}
      {lineX != null && pOverAtLine != null && (
        <g>
          <line x1={sx(lineX)} y1={padT - 2} x2={sx(lineX)} y2={H - padB} stroke={accent} strokeWidth="2" />
          <circle cx={sx(lineX)} cy={sy(pAtLine)} r="5" fill={accent} stroke="var(--panel-2)" strokeWidth="2" />
          <g transform={`translate(${Math.min(sx(lineX) + 8, W - 132)}, ${Math.max(sy(pAtLine) - 22, padT + 4)})`}>
            <rect x="0" y="-9" width="124" height="40" rx="3" fill="var(--panel-3)" stroke={accent} strokeOpacity="0.5" strokeWidth="1" />
            <text x="7" y="2" className="fp-axis" fill={accent} style={{ fontWeight: 700, fontSize: 9 }}>{lineLabel}</text>
            <text x="7" y="14" className="fp-axis" fill={!showUnder ? accent : "var(--text-dim)"} style={{ fontWeight: !showUnder ? 700 : 400, fontSize: 9 }}>P(over)   {(pOverAtLine * 100).toFixed(1)}%</text>
            <text x="7" y="26" className="fp-axis" fill={showUnder ? accent : "var(--text-dim)"} style={{ fontWeight: showUnder ? 700 : 400, fontSize: 9 }}>P(under)  {((1 - pOverAtLine) * 100).toFixed(1)}%</text>
          </g>
        </g>
      )}
      {hover && (() => {
        const dotP = showUnder ? hover.pUnder : hover.pOver;
        const dotX = discrete ? Math.min(hover.snapX, xMax) : hover.x;
        const hLabel = discrete ? `≥ ${hover.snapX}` : (hover.x < 10 ? hover.x.toFixed(2) : hover.x.toFixed(1));
        const tx = Math.min(sx(hover.x) + 10, W - 132);
        const ty = Math.max(sy(dotP) - 30, padT + 4);
        return (
          <g>
            <line x1={sx(hover.x)} y1={padT} x2={sx(hover.x)} y2={H - padB} stroke="rgba(255,255,255,.22)" strokeWidth="1" strokeDasharray="2 3" />
            <circle cx={sx(dotX)} cy={sy(dotP)} r="4" fill={accent} stroke="var(--bg)" strokeWidth="1.5" />
            <g transform={`translate(${tx}, ${ty})`}>
              <rect x="0" y="-9" width="124" height="40" rx="3" fill="var(--panel-2)" stroke="rgba(255,255,255,.14)" strokeWidth="1" />
              <text x="6" y="2" className="fp-axis" fill="var(--text-dim)" style={{ fontWeight: 500, fontSize: 9 }}>@ {hLabel}</text>
              <text x="6" y="14" className="fp-axis" fill={!showUnder ? accent : "var(--text-dim)"} style={{ fontWeight: !showUnder ? 700 : 400, fontSize: 9 }}>P(over)   {(hover.pOver * 100).toFixed(1)}%</text>
              <text x="6" y="26" className="fp-axis" fill={showUnder ? accent : "var(--text-dim)"} style={{ fontWeight: showUnder ? 700 : 400, fontSize: 9 }}>P(under)  {(hover.pUnder * 100).toFixed(1)}%</text>
            </g>
          </g>
        );
      })()}
    </svg>
  );
}

/* ── Distribution histogram ───────────────────────────────────────── */
export function CountHistogram({ hist, lineX, accent, height = 150, xMax }) {
  const W = 560, H = height, padL = 30, padR = 14, padB = 26, padT = 10;
  const [hovBin, setHovBin] = useState(null);
  const maxFrac = Math.max(...hist.map((b) => b.frac), 0.001);
  const sx = (x) => padL + (x / xMax) * (W - padL - padR);
  const bw = (W - padL - padR) / hist.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
         aria-label="Count distribution histogram"
         style={{ width: "100%", cursor: "crosshair" }}
         onMouseLeave={() => setHovBin(null)}>
      {hist.map((b, i) => {
        const h = (b.frac / maxFrac) * (H - padB - padT);
        const mid = (b.x0 + b.x1) / 2;
        const isOver = mid >= lineX;
        const isHov = hovBin === i;
        return (
          <rect key={i} x={sx(b.x0) + 1} y={H - padB - h} width={Math.max(1, bw - 2)} height={h}
            fill={isOver ? accent : "var(--panel-hi)"} opacity={isHov ? 1 : (isOver ? 0.85 : 1)} rx="1.5"
            style={{ cursor: "pointer" }}
            onMouseEnter={() => setHovBin(i)} />
        );
      })}
      <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} className="fp-grid" />
      {lineX != null && (
        <line x1={sx(lineX)} y1={padT} x2={sx(lineX)} y2={H - padB} stroke={accent} strokeWidth="2" strokeDasharray="4 3" />
      )}
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <text key={g} x={sx(g * xMax)} y={H - 9} className="fp-axis" textAnchor="middle">{Math.round(g * xMax)}</text>
      ))}
      {hovBin != null && hist[hovBin] && (() => {
        const b = hist[hovBin];
        const mx = sx((b.x0 + b.x1) / 2);
        const h = (b.frac / maxFrac) * (H - padB - padT);
        const ty = H - padB - h - 16;
        return (
          <g>
            <rect x={Math.min(mx - 40, W - 100)} y={Math.max(ty, padT)} width="96" height="18" rx="4" fill="var(--panel-2)" stroke="rgba(255,255,255,.12)" strokeWidth="1" />
            <text x={Math.min(mx - 32, W - 92)} y={Math.max(ty + 12, padT + 12)} className="fp-axis" fill="var(--text)" style={{ fontWeight: 600 }}>
              [{b.x0}–{b.x1}) {(b.frac * 100).toFixed(1)}%
            </text>
          </g>
        );
      })()}
    </svg>
  );
}

/* ── Method distribution — horizontal bars ───────────────────────── */
export function MethodBars({ method }) {
  const segs = [
    { key: "ko",  label: "KO / TKO",   val: method.ko,  color: "var(--m-r1)" },
    { key: "sub", label: "SUBMISSION", val: method.sub, color: "var(--m-td)" },
    { key: "dec", label: "DECISION",   val: method.dec, color: "var(--m-port)" },
  ];
  return (
    <div className="fp-method">
      {segs.map((s) => (
        <div className="fp-mrow" key={s.key} style={{ cursor: "default" }}>
          <div className="fp-mrow-hd">
            <span className="fp-mrow-lbl"><i style={{ background: s.color }}></i>{s.label}</span>
            <span className="fp-mrow-val">{(s.val * 100).toFixed(1)}%</span>
          </div>
          <div className="fp-mrow-track">
            <div className="fp-mrow-fill" style={{ width: s.val * 100 + "%", background: s.color }}></div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Win-probability driver bars (diverging tornado) ──────────────── */
// Per-fight feature attribution, colored to match the P(WIN) split above:
// GOLD when a factor backs the model's pick (bar extends right), GRAY when it
// favours the other fighter (extends left). Center line = even-matchup baseline.
const DRV_GOLD = "var(--gold-br)";  // supports the predicted winner
const DRV_GRAY = "var(--other)";    // favours the other fighter (matches WinSplit)

// Translate the occlusion delta (percentage points of win prob) into a plain-
// language strength so the bar means something without knowing the math.
const driverStrength = (pp) =>
  pp >= 5 ? "Strong" : pp >= 2.5 ? "Moderate" : pp >= 1 ? "Slight" : "Faint";

// A few words on what each driver group measures (hover tooltip). Keyed by the
// group key the API returns (see inference/attribution.py _GROUP_DEFS).
const DRV_EXPLAIN = {
  ratings:      "Skill ratings based on competition faced & recent fight momentum",
  grappling:    "Takedowns, control time & submission threat",
  finishing:    "KO/submission finish rate & punching power",
  reach:        "Reach, height & size advantage",
  striking:     "Strike volume, accuracy & pace",
  striking_def: "Strikes absorbed & defensive rate",
  age:          "Age, layoff & accumulated damage",
  experience:   "Career fights & rounds logged",
};

export function WinDriversBar({ drivers, aLast, bLast, aFav = true }) {
  const rows = (drivers || []).filter((d) => d && d.magnitude != null);
  if (rows.length === 0) return null;
  const maxMag = Math.max(0.01, ...rows.map((d) => d.magnitude));
  return (
    <div className="fp-drv">
      {rows.map((d) => {
        const favA = d.favors === "a";
        const who = favA ? aLast : bLast;
        const supportsPick = favA === aFav;   // backs the model's pick?
        const color = supportsPick ? DRV_GOLD : DRV_GRAY;
        const w = Math.max(2, (d.magnitude / maxMag) * 50); // % of half-track
        const pp = (d.magnitude * 100).toFixed(1);
        const strength = driverStrength(d.magnitude * 100);
        const explain = DRV_EXPLAIN[d.key];
        return (
          <div
            className="fp-drv-row"
            key={d.key}
            title={`${d.label}${explain ? " — " + explain : ""}. Pushes the pick toward ${who} by ${pp} pts.`}
          >
            <span className="fp-drv-lbl" title={explain || undefined}>{d.label}</span>
            <div className="fp-drv-track" aria-hidden="true">
              <span className="fp-drv-axis" />
              <div
                className="fp-drv-bar"
                style={{
                  background: color,
                  width: w + "%",
                  ...(supportsPick ? { left: "50%" } : { right: "50%" }),
                  borderRadius: supportsPick ? "0 4px 4px 0" : "4px 0 0 4px",
                }}
              />
            </div>
            <div className="fp-drv-val">
              <span className="fp-drv-who2" style={{ color }}>{who}</span>
              <span className="fp-drv-meta">{strength} · +{pp} pts</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Round-outcome bars ───────────────────────────────────────────── */
export function RoundBars({ dist, rounds, accent, height = 200 }) {
  const [hovIdx, setHovIdx] = useState(null);
  const labels = [];
  for (let r = 1; r <= rounds; r++) labels.push("R" + r);
  labels.push("DEC");
  const W = 560, H = height, padL = 34, padR = 14, padB = 30, padT = 14;
  const max = Math.max(...dist, 0.01);
  const n = dist.length;
  const slot = (W - padL - padR) / n;
  const bw = slot * 0.6;
  const sy = (p) => padT + (1 - p / max) * (H - padB - padT);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
         aria-label="Round outcome distribution"
         style={{ width: "100%" }}
         onMouseLeave={() => setHovIdx(null)}>
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line x1={padL} y1={sy(g * max)} x2={W - padR} y2={sy(g * max)} className="fp-grid" />
          <text x={padL - 7} y={sy(g * max) + 3} className="fp-axis" textAnchor="end">{(g * max * 100).toFixed(0)}%</text>
        </g>
      ))}
      {dist.map((p, i) => {
        const x = padL + slot * i + (slot - bw) / 2;
        const isDec = i === n - 1;
        const isHov = hovIdx === i;
        return (
          <g key={i} onMouseEnter={() => setHovIdx(i)} style={{ cursor: "default" }}>
            <rect x={x} y={sy(p)} width={bw} height={H - padB - sy(p)} rx="3"
              fill={isDec ? "var(--m-port)" : accent} opacity={isHov ? 1 : (hovIdx != null ? 0.6 : 1)} />
            <text x={x + bw / 2} y={sy(p) - 6} className="fp-axis" textAnchor="middle"
                  fill={isHov ? "var(--text)" : "var(--text-dim)"}>{(p * 100).toFixed(0)}%</text>
            <text x={x + bw / 2} y={H - 11} className="fp-axis" textAnchor="middle">{labels[i]}</text>
          </g>
        );
      })}
      {hovIdx != null && dist[hovIdx] != null && (() => {
        const p = dist[hovIdx];
        const x = padL + slot * hovIdx + (slot - bw) / 2;
        const ty = Math.max(padT + 4, sy(p) - 28);
        return (
          <g>
            <rect x={Math.min(x, W - 120)} y={ty} width="112" height="18" rx="4" fill="var(--panel-2)" stroke="rgba(255,255,255,.12)" strokeWidth="1" />
            <text x={Math.min(x + 7, W - 113)} y={ty + 12} className="fp-axis" fill="var(--text)" style={{ fontWeight: 600 }}>
              {labels[hovIdx]}: {(p * 100).toFixed(1)}%
            </text>
          </g>
        );
      })()}
    </svg>
  );
}

/* ── Duration PDF density chart ──────────────────────────────────── */
export function DurationPDFChart({ pdfCurve, xMax, lineX, pOver, accent, height = 155 }) {
  const W = 560, H = height, padL = 10, padR = 16, padB = 34, padT = 14;
  const reduced = useReducedMotion();
  if (!pdfCurve?.length) return null;

  const maxD = Math.max(...pdfCurve.map(([, d]) => d), 0.001);
  const sx = (x) => padL + (x / xMax) * (W - padL - padR);
  const sy = (d) => padT + (1 - d / maxD) * (H - padB - padT);
  const baseline = H - padB;

  // Interpolate density at line for clean split
  let dAtLine = 0;
  if (lineX != null) {
    for (let i = 1; i < pdfCurve.length; i++) {
      if (pdfCurve[i][0] >= lineX) {
        const [x0, d0] = pdfCurve[i - 1], [x1, d1] = pdfCurve[i];
        const t = (lineX - x0) / (x1 - x0 || 1);
        dAtLine = d0 + t * (d1 - d0);
        break;
      }
    }
  }

  const lxPx = lineX != null ? sx(lineX) : null;
  const lyPx = lineX != null ? sy(dAtLine) : null;
  const pts = pdfCurve.map(([x, d]) => [sx(x), sy(d)]);
  const firstX = pts[0][0], lastX = pts[pts.length - 1][0];
  const outline = "M " + pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L ");

  // Build fill areas
  const underPts = lineX != null ? pts.filter((_, i) => pdfCurve[i][0] <= lineX) : [];
  const overPts  = lineX != null ? pts.filter((_, i) => pdfCurve[i][0] >= lineX) : [];
  const underFill = lxPx != null && underPts.length > 1
    ? `M ${padL} ${baseline} L ${underPts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L ")} L ${lxPx.toFixed(1)} ${lyPx.toFixed(1)} L ${lxPx.toFixed(1)} ${baseline} Z`
    : null;
  const overFill = lxPx != null && overPts.length > 1
    ? `M ${lxPx.toFixed(1)} ${baseline} L ${lxPx.toFixed(1)} ${lyPx.toFixed(1)} L ${overPts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L ")} L ${lastX.toFixed(1)} ${baseline} Z`
    : null;
  const fullFill = lineX == null
    ? `M ${firstX.toFixed(1)} ${baseline} L ${pts.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join(" L ")} L ${lastX.toFixed(1)} ${baseline} Z`
    : null;

  const pUnder = pOver != null ? 1 - pOver : null;
  const xticks = 6;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
         aria-label="Fight duration probability density"
         style={{ width: "100%" }}>
      <line x1={padL} y1={baseline} x2={W - padR} y2={baseline} className="fp-grid" />
      {Array.from({ length: xticks + 1 }).map((_, i) => {
        const x = (xMax * i) / xticks;
        return <text key={i} x={sx(x)} y={H - 12} className="fp-axis" textAnchor="middle">{Math.round(x)}</text>;
      })}
      <text x={(padL + W - padR) / 2} y={H - 1} className="fp-axis-ttl" textAnchor="middle">DURATION (MIN) · DENSITY</text>

      <motion.g initial={{ opacity: reduced ? 1 : 0 }} animate={{ opacity: 1 }} transition={fadeTransition(reduced)}>
        {underFill && <path d={underFill} fill="var(--neg)" fillOpacity="0.20" />}
        {overFill  && <path d={overFill}  fill={accent} fillOpacity="0.22" />}
        {fullFill  && <path d={fullFill}  fill={accent} fillOpacity="0.20" />}
      </motion.g>
      {reduced ? (
        <path d={outline} fill="none" stroke={accent} strokeWidth="1.8" strokeLinejoin="round" />
      ) : (
        <motion.path d={outline} fill="none" stroke={accent} strokeWidth="1.8" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={traceTransition(reduced)} />
      )}

      {lxPx != null && (
        <line x1={lxPx} y1={padT} x2={lxPx} y2={baseline} stroke={accent} strokeWidth="1.5" strokeDasharray="4 3" />
      )}

      {/* Percentage labels in fill regions */}
      {pUnder != null && lxPx != null && lxPx > padL + 56 && (
        <text x={(padL + lxPx) / 2} y={padT + 16} className="fp-axis"
              textAnchor="middle" fill="var(--neg)" style={{ fontWeight: 700, fontSize: 10 }}>
          P(under) {(pUnder * 100).toFixed(1)}%
        </text>
      )}
      {pOver != null && lxPx != null && lxPx < W - padR - 56 && (
        <text x={(lxPx + W - padR) / 2} y={padT + 16} className="fp-axis"
              textAnchor="middle" fill={accent} style={{ fontWeight: 700, fontSize: 10 }}>
          P(over) {(pOver * 100).toFixed(1)}%
        </text>
      )}
    </svg>
  );
}

/* ── Edge buckets — expected vs realized hit rate, grouped bars ───── */
// buckets: [{label, n, hitRate, expectedRate}]. expectedRate = mean model_prob
// claimed by picks in that bucket; hitRate = what actually happened. Reading the
// gap between the two pairs tells you whether the model's confidence in that
// edge band is earned or inflated.
export function EdgeBucketChart({ buckets, accent = "var(--m-port)", height = 220 }) {
  const W = 560, H = height, padL = 36, padR = 16, padB = 46, padT = 16;
  const [hov, setHov] = useState(null);
  if (!buckets?.length) return null;
  const n = buckets.length;
  const slot = (W - padL - padR) / n;
  const bw = Math.min(30, slot * 0.32);
  const sy = (p) => padT + (1 - p) * (H - padB - padT);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
         aria-label="Expected vs realized hit rate by edge bucket"
         style={{ width: "100%" }} onMouseLeave={() => setHov(null)}>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={padL} y1={sy(g)} x2={W - padR} y2={sy(g)} className="fp-grid" strokeDasharray={g === 0.5 ? "4 4" : ""} />
          <text x={padL - 8} y={sy(g) + 3} className="fp-axis" textAnchor="end">{(g * 100).toFixed(0)}%</text>
        </g>
      ))}
      {buckets.map((b, i) => {
        const cx = padL + slot * i + slot / 2;
        const isHov = hov === i;
        return (
          <g key={b.label} onMouseEnter={() => setHov(i)} style={{ cursor: "default" }}>
            <rect x={cx - bw - 2} y={sy(b.expectedRate)} width={bw} height={H - padB - sy(b.expectedRate)}
              fill="var(--text-faint)" opacity={isHov ? 0.9 : 0.55} rx="2" />
            <rect x={cx + 2} y={sy(b.hitRate)} width={bw} height={H - padB - sy(b.hitRate)}
              fill={accent} opacity={isHov ? 1 : 0.85} rx="2" />
            <text x={cx} y={H - padB + 16} className="fp-axis" textAnchor="middle">{b.label}</text>
            <text x={cx} y={H - padB + 28} className="fp-axis" textAnchor="middle" fill="var(--text-faint)" style={{ fontSize: 8 }}>n={b.n}</text>
          </g>
        );
      })}
      {hov != null && buckets[hov] && (() => {
        const b = buckets[hov];
        const cx = padL + slot * hov + slot / 2;
        const ty = Math.max(padT + 4, sy(Math.max(b.hitRate, b.expectedRate)) - 40);
        const tx = Math.min(cx - 55, W - 126);
        return (
          <g>
            <rect x={Math.max(tx, padL)} y={ty} width="122" height="34" rx="3" fill="var(--panel-2)" stroke="rgba(255,255,255,.14)" strokeWidth="1" />
            <text x={Math.max(tx, padL) + 7} y={ty + 12} className="fp-axis" fill="var(--text-faint)" style={{ fontSize: 8 }}>EXPECTED (CLAIMED)</text>
            <text x={Math.max(tx, padL) + 7} y={ty + 12} dx="90" className="fp-axis" fill="var(--text)" style={{ fontWeight: 700, fontSize: 9 }}>{(b.expectedRate * 100).toFixed(0)}%</text>
            <text x={Math.max(tx, padL) + 7} y={ty + 25} className="fp-axis" fill={accent} style={{ fontSize: 8 }}>REALIZED (ACTUAL)</text>
            <text x={Math.max(tx, padL) + 7} y={ty + 25} dx="90" className="fp-axis" fill={accent} style={{ fontWeight: 700, fontSize: 9 }}>{(b.hitRate * 100).toFixed(0)}%</text>
          </g>
        );
      })()}
    </svg>
  );
}

/* ── CLV over time — beat-close% per event ────────────────────────── */
export function CLVLineChart({ series, accent = "var(--m-port)", height = 180 }) {
  const W = 560, H = height, padL = 38, padR = 16, padB = 26, padT = 16;
  const reduced = useReducedMotion();
  const [hov, setHov] = useState(null);
  const n = series?.length || 0;
  if (n < 2) return null;

  const sx = (i) => padL + (i / (n - 1)) * (W - padL - padR);
  const sy = (p) => padT + (1 - p) * (H - padB - padT);
  const linePath = "M " + series.map((p, i) => `${sx(i).toFixed(1)} ${sy(p.beatClose).toFixed(1)}`).join(" L ");

  const handleMove = (e) => {
    const svgEl = e.currentTarget;
    const rect = svgEl.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const rawI = ((mx - padL) / (W - padL - padR)) * (n - 1);
    if (rawI < -0.5 || rawI > n - 0.5) { setHov(null); return; }
    setHov(Math.round(Math.min(n - 1, Math.max(0, rawI))));
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
         aria-label="Beat-close rate per event"
         style={{ width: "100%", cursor: "crosshair" }}
         onMouseMove={handleMove} onMouseLeave={() => setHov(null)}>
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line x1={padL} y1={sy(g)} x2={W - padR} y2={sy(g)} className="fp-grid" strokeDasharray={g === 0.5 ? "4 4" : ""} />
          <text x={padL - 8} y={sy(g) + 3} className="fp-axis" textAnchor="end">{(g * 100).toFixed(0)}%</text>
        </g>
      ))}
      {reduced ? (
        <path d={linePath} fill="none" stroke={accent} strokeWidth="2.2" strokeLinejoin="round" />
      ) : (
        <motion.path d={linePath} fill="none" stroke={accent} strokeWidth="2.2" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={traceTransition(reduced)} />
      )}
      {series.map((p, i) => <circle key={i} cx={sx(i)} cy={sy(p.beatClose)} r="3" fill={accent} stroke="var(--bg)" strokeWidth="1" />)}
      {hov != null && series[hov] && (() => {
        const p = series[hov];
        const tx = Math.min(sx(hov) + 8, W - 120);
        const ty = Math.max(sy(p.beatClose) - 30, padT + 2);
        return (
          <g>
            <g transform={`translate(${tx}, ${ty})`}>
              <rect x="0" y="-9" width="116" height="30" rx="3" fill="var(--panel-2)" stroke="rgba(255,255,255,.14)" strokeWidth="1" />
              <text x="6" y="2" className="fp-axis" fill="var(--text)" style={{ fontWeight: 600, fontSize: 9 }}>{p.date}</text>
              <text x="6" y="14" className="fp-axis" fill={accent} style={{ fontWeight: 700, fontSize: 9 }}>{(p.beatClose * 100).toFixed(0)}% beat close (n={p.n})</text>
            </g>
          </g>
        );
      })()}
    </svg>
  );
}

/* ── Exchange CLV — signed cents per event (Kalshi lane) ──────────── */
// series: [{date, n, avgClvCents}], oldest first. Unlike CLVLineChart (a 0-100%
// beat-close rate), this is a signed dollar-cents quantity with no fixed range,
// so the y-domain is derived from the data with the zero line always drawn.
export function ClvCentsChart({ series, accent = "var(--m-exch)", height = 180 }) {
  const W = 560, H = height, padL = 38, padR = 16, padB = 26, padT = 16;
  const reduced = useReducedMotion();
  const [hov, setHov] = useState(null);
  const n = series?.length || 0;
  if (n < 2) return null;

  const vals = series.map((p) => p.avgClvCents);
  const span = Math.max(...vals.map(Math.abs), 1) * 1.15;
  const lo = -span, hi = span;

  const sx = (i) => padL + (i / (n - 1)) * (W - padL - padR);
  const sy = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padB - padT);
  const linePath = "M " + series.map((p, i) => `${sx(i).toFixed(1)} ${sy(p.avgClvCents).toFixed(1)}`).join(" L ");

  const handleMove = (e) => {
    const svgEl = e.currentTarget;
    const rect = svgEl.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (W / rect.width);
    const rawI = ((mx - padL) / (W - padL - padR)) * (n - 1);
    if (rawI < -0.5 || rawI > n - 0.5) { setHov(null); return; }
    setHov(Math.round(Math.min(n - 1, Math.max(0, rawI))));
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
         aria-label="Average closing-line value in cents, per event"
         style={{ width: "100%", cursor: "crosshair" }}
         onMouseMove={handleMove} onMouseLeave={() => setHov(null)}>
      {[lo, 0, hi].map((g) => (
        <g key={g}>
          <line x1={padL} y1={sy(g)} x2={W - padR} y2={sy(g)} className="fp-grid" strokeDasharray={g === 0 ? "4 4" : ""} />
          <text x={padL - 8} y={sy(g) + 3} className="fp-axis" textAnchor="end">{g >= 0 ? "+" : ""}{g.toFixed(0)}c</text>
        </g>
      ))}
      {reduced ? (
        <path d={linePath} fill="none" stroke={accent} strokeWidth="2.2" strokeLinejoin="round" />
      ) : (
        <motion.path d={linePath} fill="none" stroke={accent} strokeWidth="2.2" strokeLinejoin="round"
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={traceTransition(reduced)} />
      )}
      {series.map((p, i) => (
        <circle key={i} cx={sx(i)} cy={sy(p.avgClvCents)} r="3"
          fill={p.avgClvCents >= 0 ? "var(--pos)" : "var(--neg)"} stroke="var(--bg)" strokeWidth="1" />
      ))}
      {hov != null && series[hov] && (() => {
        const p = series[hov];
        const tx = Math.min(sx(hov) + 8, W - 130);
        const ty = Math.max(sy(p.avgClvCents) - 30, padT + 2);
        return (
          <g transform={`translate(${tx}, ${ty})`}>
            <rect x="0" y="-9" width="126" height="30" rx="3" fill="var(--panel-2)" stroke="rgba(255,255,255,.14)" strokeWidth="1" />
            <text x="6" y="2" className="fp-axis" fill="var(--text)" style={{ fontWeight: 600, fontSize: 9 }}>{p.date}</text>
            <text x="6" y="14" className="fp-axis" fill={p.avgClvCents >= 0 ? "var(--pos)" : "var(--neg)"} style={{ fontWeight: 700, fontSize: 9 }}>
              {p.avgClvCents >= 0 ? "+" : ""}{p.avgClvCents.toFixed(1)}c avg (n={p.n})
            </text>
          </g>
        );
      })()}
    </svg>
  );
}

/* ── Model vs Market scatter — Kalshi graded taker picks ──────────── */
// points: [{modelP, marketP, hit}]. Diagonal = model agrees with market; points
// above the diagonal are where the model priced a fighter higher than Kalshi did.
export function ModelVsMarketScatter({ points, height = 280 }) {
  const W = 300, H = height, pad = 30;
  if (!points?.length) return null;
  const s = (v) => pad + v * (W - 2 * pad);
  const sy = (v) => H - pad - v * (H - 2 * pad);

  return (
    <div className="fp-mvm-scatter">
      <svg viewBox={`0 0 ${W} ${H}`} className="fp-chart" role="img"
           aria-label="Model probability vs Kalshi market price, graded picks"
           style={{ width: "100%", maxWidth: 320, margin: "0 auto" }}>
        <line x1={s(0)} y1={sy(0)} x2={s(1)} y2={sy(1)} className="fp-grid" strokeDasharray="4 4" />
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <text x={s(g)} y={H - pad + 14} className="fp-axis" textAnchor="middle">{(g * 100).toFixed(0)}</text>
            <text x={pad - 6} y={sy(g) + 3} className="fp-axis" textAnchor="end">{(g * 100).toFixed(0)}</text>
          </g>
        ))}
        <text x={W / 2} y={H - 4} className="fp-axis" textAnchor="middle" style={{ fontSize: 9 }}>Market ask %</text>
        <text x={10} y={H / 2} className="fp-axis" textAnchor="middle" style={{ fontSize: 9 }} transform={`rotate(-90 10 ${H / 2})`}>Model %</text>
        {points.map((p, i) => (
          <circle key={i} cx={s(p.marketP)} cy={sy(p.modelP)} r="3.5"
            fill={p.hit ? "var(--pos)" : "var(--neg)"} opacity="0.75" />
        ))}
      </svg>
      <div className="fp-drv-legend">
        <span className="fp-drv-leg-item"><i style={{ background: "var(--pos)" }} />Pick hit</span>
        <span className="fp-drv-leg-item"><i style={{ background: "var(--neg)" }} />Pick missed</span>
      </div>
    </div>
  );
}

/* ── Calibration sparkline ────────────────────────────────────────── */
export function Sparkline({ data, width = 150, height = 30, baseline }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data) - 0.02, max = Math.max(...data) + 0.02;
  const sx = (i) => (i / (data.length - 1)) * width;
  const sy = (v) => height - ((v - min) / (max - min)) * height;
  const path = "M " + data.map((v, i) => `${sx(i)} ${sy(v)}`).join(" L ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width, height }}>
      {baseline != null && <line x1="0" y1={sy(baseline)} x2={width} y2={sy(baseline)} className="fp-grid" />}
      <path d={path} fill="none" stroke="var(--gold-br)" strokeWidth="1.5" />
      <circle cx={sx(data.length - 1)} cy={sy(data[data.length - 1])} r="2.5" fill="var(--gold-br)" />
    </svg>
  );
}
