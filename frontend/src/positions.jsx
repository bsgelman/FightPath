import { useMemo, useState, useEffect } from "react";
import { selectPositions, sizePositions, MIN_PSTAR } from "./lib/positions.js";
import { _liqBadge, kindMeta } from "./exchange.jsx";
import { lastName, NumField } from "./components.jsx";

/* Shared grid so figures line up across every position card, mirroring the
   Explore lane's ROW_GRID. */
const POS_GRID = "132px minmax(84px,1fr) 66px 66px 52px 92px 78px 46px";

/** One position card per fight, built from the same chrome the Explore lane uses
 *  (fp-bbcard / -hd / -colhd / fp-bbrow) so the two views read as one product. */
function PositionCard({ fight, pos, rank }) {
  const meta = kindMeta(pos.marketKind);
  const cents = Math.round(pos.ask * 100);

  return (
    <div className="fp-bbcard pos">
      <div className="fp-bbcard-hd">
        <span className="fp-bbcard-names">
          <em className="r">{fight ? lastName(fight.a.name) : `Fight ${pos.fightIdx + 1}`}</em>
          {fight && <><i>vs</i><em className="b">{lastName(fight.b.name)}</em></>}
        </span>
        <span className="fp-bbcard-meta">
          bout {rank} · staking <b className="pos">${pos.stake.toFixed(2)}</b>
        </span>
      </div>

      <div className="fp-bbcard-colhd" style={{ gridTemplateColumns: POS_GRID }}>
        <span>Fighter</span><span>Market</span><span>Model</span>
        <span title="Midway between your model and the market-implied probability — the number the stake is sized from">Midpoint</span>
        <span>Ask</span><span>Buy</span><span>Stake</span><span>Liquidity</span>
      </div>

      {/* data-l supplies the column label on narrow screens, where the shared grid
          collapses and the header row is hidden. */}
      <div className="fp-bbrow fp-pos-row" style={{ gridTemplateColumns: POS_GRID }}>
        <div className="fp-td-fighter">{lastName(pos.fighter)}</div>
        <div><span className="fp-mkt-tag" style={{ color: meta.accent }}>{meta.label}</span></div>
        <div className="fp-pos-model" data-l="Model">{(pos.modelP * 100).toFixed(1)}%</div>
        <div className="fp-pos-priced" data-l="Midpoint">{(pos.pStar * 100).toFixed(1)}%</div>
        <div data-l="Ask">{cents}¢</div>
        <div className="fp-pos-buy" data-l="Buy">{pos.contracts} @ {cents}¢</div>
        <div className="fp-pos-stakecell" data-l="Stake">${pos.stake.toFixed(2)}</div>
        <div data-l="Liquidity">{_liqBadge(pos)}</div>
      </div>
    </div>
  );
}

/** Card risk as one meter: the figure and the bar in the same block, segmented in
 *  bout order so a light card reads short at a glance. Gaps mark fights holding no
 *  position; the rail's full width is the budget, so it can never look overfull. */
function BudgetMeter({ sized, fights, budget }) {
  const total = sized.reduce((s, r) => s + r.stake, 0);
  const byFight = new Map(sized.map((p) => [p.fightIdx, p]));
  const pct = budget > 0 ? (total / budget) * 100 : 0;
  const slack = budget > 0 ? Math.max(0, 1 - total / budget) : 1;

  return (
    <div className="fp-budget-block">
      <div className="fp-budget-figure">
        <b>${total.toFixed(2)}</b>
        <span>of ${budget.toFixed(2)} at risk</span>
      </div>
      <div
        className="fp-budget"
        role="img"
        aria-label={`$${total.toFixed(2)} at risk of a $${budget.toFixed(2)} card budget, across ${sized.length} of ${fights.length} fights`}
      >
        {fights.map((_, i) => {
          const p = byFight.get(i);
          const grow = p && budget > 0 ? p.stake / budget : 0;
          // Gaps are FIXED width, never flex-grown: growable gaps would join the
          // proportional denominator and the fill would understate the real
          // percentage (9 abstains at 0.04 made a 91% card render as ~67%).
          return grow > 0
            ? <div key={i} className="fp-budget-seg" style={{ flexGrow: grow }} />
            : <div key={i} className="fp-budget-gap" style={{ flex: "0 0 3px" }} />;
        })}
        <div className="fp-budget-gap" style={{ flexGrow: slack }} />
      </div>
      <div className="fp-budget-cap">{pct.toFixed(0)}% of budget deployed</div>
    </div>
  );
}

/** Fights holding no position. Stated and countable rather than rendered as a wall
 *  of identical rows — eleven grey rows hid the count instead of showing it, and
 *  buried the one real position among them. Expandable, never hidden. */
function NoPositionNote({ items }) {
  const [open, setOpen] = useState(false);
  if (!items.length) return null;
  const lowData = items.filter((i) => i.lowData).length;
  const belowFloor = items.length - lowData;

  return (
    <div className="fp-nopos">
      <button type="button" className="fp-nopos-btn" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className={"fp-nopos-chev" + (open ? " open" : "")}>›</span>
        No position on {items.length} fight{items.length !== 1 ? "s" : ""}
        <em>
          {lowData > 0 && `${lowData} low data`}
          {lowData > 0 && belowFloor > 0 && " · "}
          {belowFloor > 0 && `${belowFloor} below the ${Math.round(MIN_PSTAR * 100)}% floor`}
        </em>
      </button>
      {open && (
        <ul className="fp-nopos-list">
          {items.map((i) => (
            <li key={i.idx}>
              <span className="fp-nopos-bout">
                {i.fight ? `${lastName(i.fight.a.name)} — ${lastName(i.fight.b.name)}` : `Fight ${i.idx + 1}`}
              </span>
              <span className="fp-nopos-why">
                {i.lowData ? "not enough UFC data on this matchup" : `no contract cleared the ${Math.round(MIN_PSTAR * 100)}% floor`}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const LS_BANKROLL = "fp.bankroll";
const LS_RISK = "fp.cardRiskPct";

const readLS = (k, dflt) => {
  try {
    const v = parseFloat(localStorage.getItem(k));
    return Number.isFinite(v) && v > 0 ? v : dflt;
  } catch {
    return dflt;   // private-mode / storage-disabled: fall back, never throw
  }
};
const writeLS = (k, v) => { try { localStorage.setItem(k, String(v)); } catch { /* ignore */ } };

export function PositionsPanel({ rows, fights }) {
  const [bankroll, setBankroll] = useState(() => readLS(LS_BANKROLL, 64));
  const [cardRiskPct, setCardRiskPct] = useState(() => readLS(LS_RISK, 0.10));

  useEffect(() => { writeLS(LS_BANKROLL, bankroll); }, [bankroll]);
  useEffect(() => { writeLS(LS_RISK, cardRiskPct); }, [cardRiskPct]);

  // Low-data fights are excluded from candidacy (the model is pinned near 0.5 there,
  // and differencing that against an informed price prints a phantom edge).
  const candidates = useMemo(
    () => (rows ?? []).filter((r) => !fights?.[r.fightIdx]?.lowData),
    [rows, fights],
  );

  const sized = useMemo(
    () => sizePositions(selectPositions(candidates), bankroll, cardRiskPct),
    [candidates, bankroll, cardRiskPct],
  );

  const byFight = useMemo(() => new Map(sized.map((p) => [p.fightIdx, p])), [sized]);

  const noPosition = useMemo(
    () => (fights ?? [])
      .map((f, idx) => ({ f, idx }))
      .filter(({ idx }) => !byFight.has(idx))
      .map(({ f, idx }) => ({ idx, fight: f, lowData: !!f?.lowData })),
    [fights, byFight],
  );

  if (!fights?.length) {
    return <div className="fp-pos-empty">No card loaded. Pick an event to see positions.</div>;
  }

  return (
    <div className="fp-positions">
      <div className="fp-exch-hd">
        <div>
          <span className="fp-exch-ttl">Card Budget</span>
          <span className="fp-exch-sub">
            One contract per fight, quarter-Kelly on the shrunk probability, capped by the card budget
          </span>
        </div>
        <div className="fp-pos-ctl">
          <span className="fp-pos-ctl-lbl">Bankroll</span>
          <NumField
            value={bankroll} onChange={setBankroll} ariaLabel="Bankroll in dollars"
            prefix="$" min={1} max={1000000} step={1} decimals={0} width={104} />
          <span className="fp-pos-ctl-lbl">Risk this card</span>
          <NumField
            value={cardRiskPct * 100} onChange={(v) => setCardRiskPct(v / 100)}
            ariaLabel="Percent of bankroll to risk on this card"
            suffix="%" min={1} max={100} step={1} decimals={0} width={92} />
        </div>
      </div>

      <BudgetMeter sized={sized} fights={fights} budget={bankroll * cardRiskPct} />

      {sized.length === 0 ? (
        <div className="fp-exch-note">
          <b>No positions on this card.</b> Nothing cleared the {Math.round(MIN_PSTAR * 100)}% floor
          once the model was reconciled with the market. That is a result, not a failure — check
          Explore to see the full board.
        </div>
      ) : (
        <div className="fp-bbcards">
          {fights.map((f, i) => {
            const pos = byFight.get(i);
            return pos ? <PositionCard key={i} fight={f} pos={pos} rank={i + 1} /> : null;
          })}
        </div>
      )}

      <NoPositionNote items={noPosition} />

      <span className="fp-bb-hint">
        Midpoint is halfway between your model and the market-implied probability — the number the stake is sized from.
        Stakes are quarter-Kelly, scaled down together if they would exceed the card budget.
        Nothing here is established +EV; the board exists to be coherent and consistently sized.
      </span>
    </div>
  );
}
