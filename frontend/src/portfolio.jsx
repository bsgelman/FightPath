import { baseMultFor } from "./api/client.js";
import { breakEvenPerLeg } from "./lib/stats.js";

export function PortfolioRail({ picks, onRemove, onClear, payout, mult, multDirty, breakeven }) {
  const legs = picks.length;
  const combinedHit = legs ? picks.reduce((m, p) => m * p.modelP, 1) : 0;

  // Auto-price from the legs ACTUALLY added (picks.length) — not a manually selected
  // leg count, which could desync from the real parlay. Base multiplier is looked up by
  // (platform, legCount) from the operator payout table; each leg's own per-line
  // multiplier (Flat Multi goblin/demon/boost/finish; Power Play deferred → null = 1.0×)
  // is folded on top: effectiveMult = baseMult × Π(lineMult). Reduces exactly to the
  // flat selected multiplier for an all-standard single-platform portfolio.
  //
  // When the user manually edits the multiplier (multDirty=true), their value IS the
  // final parlay multiplier — the exact number the operator shows. We skip both the
  // auto leg-count lookup and per-leg line folding to avoid double-counting. Resets to
  // auto when the user picks a new payout type from the dropdown.
  const autoBase = baseMultFor(payout.platform, legs) ?? mult;
  const prodLineMult = legs ? picks.reduce((m, p) => m * (p.lineMult ?? 1), 1) : 1;
  const effectiveMult = multDirty ? mult : autoBase * prodLineMult;
  const baseBreakeven = legs ? breakEvenPerLeg(effectiveMult, legs) : breakeven;
  const legBE = (p) => multDirty ? baseBreakeven : baseBreakeven / (p.lineMult ?? 1);

  const reqProb = 1 / effectiveMult;
  const ev = legs ? combinedHit * effectiveMult - 1 : 0;
  const enough = legs >= 2;
  let vClass = "flat", vTag = "—";
  if (enough) {
    if (ev >= 0.05)       { vClass = "pos";  vTag = "VALUE"; }
    else if (ev >= -0.03) { vClass = "flat"; vTag = "FAIR"; }
    else                  { vClass = "neg";  vTag = "FADE"; }
  }
  return (
    <aside className="fp-port">
      <div className="fp-port-hd">
        <span className="fp-port-ttl">PORTFOLIO</span>
        <span className="fp-port-count">{legs} {legs === 1 ? "LEG" : "LEGS"}</span>
        {legs > 0 && <button className="fp-port-clear" onClick={onClear}>CLEAR</button>}
      </div>

      {legs === 0 ? (
        <div className="fp-port-empty">
          <div className="fp-port-empty-mark">
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3l7 4-7 4-7-4z"/><path d="M3 11l7 4 7-4"/></svg>
          </div>
          <p>Add at least 2 prop legs with the<br /><b style={{ color: "var(--gold-br)" }}>Add to portfolio</b> button on any<br />prop tab or from Positions.</p>
        </div>
      ) : (
        <div className="fp-port-legs">
          {picks.map((p) => (
            <div className="fp-pleg" key={p.key}>
              <span className="fp-pleg-acc" style={{ background: p.accent }}></span>
              <div className="fp-pleg-main">
                <span className="fp-pleg-lbl">{p.label}</span>
                <span className="fp-pleg-sub">{p.sub}</span>
              </div>
              <div className="fp-pleg-right">
                <span className="fp-pleg-p">{(p.modelP * 100).toFixed(0)}%</span>
                <span className={"fp-pleg-edge " + (p.modelP - legBE(p) >= 0 ? "pos" : "neg")}>
                  {p.modelP - legBE(p) >= 0 ? "+" : ""}{((p.modelP - legBE(p)) * 100).toFixed(0)}%
                </span>
              </div>
              <button className="fp-pleg-x" aria-label="Remove leg" onClick={() => onRemove(p.key)}>×</button>
            </div>
          ))}
        </div>
      )}

      <div className="fp-port-foot">
        {(() => {
          const barMax = Math.max(combinedHit, reqProb, 0.05) * 1.6;
          const hitPct  = (combinedHit / barMax) * 100;
          const needPct = (reqProb     / barMax) * 100;
          const edge    = combinedHit - reqProb;
          const isPos   = edge >= 0;
          return (
            <div className="fp-pb-block">
              <div className="fp-pb-track">
                <div className="fp-pb-progress" style={{ width: hitPct.toFixed(1) + "%", background: isPos ? "var(--pos)" : "var(--neg)" }} />
                <div className="fp-pb-notch" style={{ left: needPct.toFixed(1) + "%" }}>
                  <span className="fp-pb-notch-lbl">NEED</span>
                </div>
              </div>
              <div className="fp-pb-stats">
                <span className={"fp-pbs-model " + (isPos ? "pos" : "neg")}>MODEL {(combinedHit * 100).toFixed(1)}%</span>
                <span className={"fp-pbs-edge " + (isPos ? "pos" : "neg")}>{isPos ? "+" : ""}{(edge * 100).toFixed(1)}% EDGE</span>
                <span className="fp-pbs-need">NEED {(reqProb * 100).toFixed(1)}%</span>
              </div>
            </div>
          );
        })()}
        <div className={"fp-verdict " + vClass}>
          <span className="fp-vd-lbl">{enough ? "MODEL EXPECTED VALUE" : "ADD " + (2 - legs) + " MORE LEG" + (2 - legs === 1 ? "" : "S")}</span>
          <span className="fp-vd-val">{enough ? (ev >= 0 ? "+" : "") + (ev * 100).toFixed(0) + "%" : "—"}</span>
          {enough && <span className="fp-vd-tag">{vTag}</span>}
        </div>
        <div className="fp-payout-grid">
          <div className="fp-pog"><span>LEGS</span><b>{legs}</b></div>
          <div className="fp-pog"><span>MULTIPLIER</span><b className="gold">{effectiveMult.toFixed(multDirty && effectiveMult % 1 ? 2 : effectiveMult % 1 ? 1 : 0)}×</b></div>
          <div className="fp-pog"><span>MODEL EV</span><b className={enough ? (ev >= 0 ? "pos" : "neg") : ""}>{enough ? (ev >= 0 ? "+" : "") + (ev * 100).toFixed(0) + "%" : "—"}</b></div>
        </div>
        <span className="fp-card-note" style={{ textAlign: "center" }}>
          {payout.platform === "ud" ? "Flat Multi" : "Power Play"} · {legs} legs ·{" "}
          {multDirty
            ? `manual override ${effectiveMult.toFixed(effectiveMult % 1 ? 2 : 0)}×`
            : `auto-priced at ${autoBase.toFixed(autoBase % 1 ? 1 : 0)}× base${prodLineMult !== 1 ? ` × ${prodLineMult.toFixed(2)} line` : ""} = ${effectiveMult.toFixed(effectiveMult % 1 ? 1 : 0)}×`}.
        </span>
      </div>
    </aside>
  );
}
