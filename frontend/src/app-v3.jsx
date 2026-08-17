import { useState, useEffect, useCallback, useRef } from "react";
import {
  getCards, refreshCards, getCard, predictManual,
  getReferees, getWeightClasses, getMarketLines,
  gradePortfolio, getHistory, getLedgerSummary, getMeta, exportCardCsv,
  PAYOUTS,
} from "./api/client.js";
import { breakEvenPerLeg } from "./lib/stats.js";
import { useTweaks, TweaksPanel, TweakSection, TweakRadio } from "./tweaks-panel.jsx";
import {
  SidebarV3, TopNav, BetStrip, Hero, MatchupHeader,
  FightCardPage, PropLabPage, PositionsPage, PortfolioPage,
  ModelVsMarketPage, AboutPage, SettingsPage, BottomNav,
} from "./panels.jsx";

const TWEAK_DEFAULTS_V3 = { density: "regular", base: "midnight" };

/* Card-list retry pacing. Base matches the old fixed 4 s so a local cold start
 * still recovers quickly; the ceiling keeps an unattended tab from hammering a
 * hosted backend (VITE_API_URL repoints this loop off localhost). */
const RETRY_BASE_MS = 4000;
const RETRY_MAX_MS = 30000;

export function AppV3() {
  const [t, setTweak]        = useTweaks(TWEAK_DEFAULTS_V3);
  const [layout, setLayout]  = useState("topbar");
  const [nav, setNav]        = useState("card");

  // Payout
  const [payoutKey, setPayoutKeyRaw] = useState("pp_power_2");
  const [mult, setMult]              = useState(PAYOUTS["pp_power_2"].mult);
  const [multDirty, setMultDirty]    = useState(false);
  // Wrap setMult for user-initiated edits (BetStrip +/- buttons, text input, Settings).
  // Flags multDirty so Portfolio treats the value as a final override rather than a base.
  const setMultUser = useCallback((u) => { setMult(u); setMultDirty(true); }, []);
  function choosePayout(k) { setPayoutKeyRaw(k); setMult(PAYOUTS[k].mult); setMultDirty(false); }

  // Card selection
  const [cards, setCards]               = useState([]);
  const [selectedCardId, setSelectedCardId] = useState(null);
  const [fights, setFights]             = useState([]);
  const [unavailFights, setUnavailFights] = useState([]);
  const [event, setEvent]               = useState(null);
  const [loadingCard, setLoadingCard]   = useState(false);

  // Fight selection
  const [selId, setSelId] = useState(null);

  // Prop-lab
  const [tab, setTab]         = useState("duration");
  const [durLine, setDurLine] = useState(null);
  const [durSide, setDurSide] = useState("over");
  const [cp, setCp] = useState({
    sig:     { f: "a", line: null, side: "over" },
    r1sig:   { f: "a", line: null, side: "over" },
    bodySig: { f: "a", line: null, side: "over" },
    legSig:  { f: "a", line: null, side: "over" },
    combo:   { f: "a", line: null, side: "over" },
    td:      { f: "a", line: null, side: "over" },
    r1td:    { f: "a", line: null, side: "over" },
    subAtt:  { f: "a", line: null, side: "over" },
    ctrl:    { f: "a", line: null, side: "over" },
    kd:      { f: "a", line: null, side: "over" },
  });
  const patchCp = (mk, p) => setCp((prev) => ({ ...prev, [mk]: { ...prev[mk], ...p } }));

  // Portfolio
  const [picks, setPicks] = useState([]);
  function toggleLeg(leg) {
    setPicks((prev) =>
      prev.some((x) => x.key === leg.key)
        ? prev.filter((x) => x.key !== leg.key)
        : [...prev, leg]
    );
  }

  // History + Meta (lazy)
  const [history, setHistory]   = useState(null);
  const [ledger, setLedger]     = useState(null);
  const [meta, setMeta]         = useState(null);

  // Exchange (Kalshi) market lines — lazy-fetched after card load, non-blocking.
  const [marketLines, setMarketLines] = useState(null);
  const [marketLinesLoading, setMarketLinesLoading] = useState(false);

  // Card source: scraped | manual
  const [cardSource, setCardSource] = useState("scraped");

  // Manual form
  const [manualForm, setManualForm] = useState({
    red: "", blue: "", rounds: 3, isTitle: false,
    eventDate: new Date().toISOString().slice(0, 10),
    weightClass: "", referee: "", location: "",
  });
  const [manualLoading, setManualLoading] = useState(false);
  const [referees, setReferees]     = useState([]);
  const [weightClasses, setWeightClasses] = useState([]);

  /* ── Bootstrap ─────────────────────────────────────────────────── */
  const [cardsError, setCardsError] = useState(false);
  const _retryRef = useRef(null);
  const _retryDelayRef = useRef(RETRY_BASE_MS);
  // Mirrors cardsError so the visibilitychange listener (registered once, closing
  // over the initial state) can read the current value.
  const cardsErrorRef = useRef(false);
  useEffect(() => { cardsErrorRef.current = cardsError; }, [cardsError]);

  /* Retries with exponential backoff, and never polls a backgrounded tab.
   * The old version hammered every 4 s forever: harmless against a local API,
   * but VITE_API_URL points this same loop at the hosted Space, and a tab left
   * open overnight on a cold/erroring backend is ~21k requests/day from one IP.
   * Backoff caps the rate; the visibility check means an unwatched tab makes
   * none at all and resumes the moment it is looked at. */
  function loadCards() {
    if (document.hidden) {
      // Re-check lazily; the visibilitychange listener resumes immediately on focus,
      // so there is no reason to wake on the short interval while nobody is looking.
      _retryRef.current = setTimeout(loadCards, RETRY_MAX_MS);
      return Promise.resolve();
    }
    setCardsError(false);
    return getCards().then((list) => {
      _retryDelayRef.current = RETRY_BASE_MS;
      setCards(list);
      if (list.length && !selectedCardId) setSelectedCardId(list[0].id);
    }).catch(() => {
      setCardsError(true);
      const wait = _retryDelayRef.current;
      _retryDelayRef.current = Math.min(wait * 2, RETRY_MAX_MS);
      _retryRef.current = setTimeout(loadCards, wait);
    });
  }

  useEffect(() => {
    getMeta().then(setMeta).catch(() => {});
    loadCards();
    getReferees().then(setReferees).catch(() => {});
    getWeightClasses().then(setWeightClasses).catch(() => {});
    // Come back immediately when the tab is focused rather than waiting out the
    // current backoff — a user who just switched to the tab wants it to load now.
    const onVisible = () => {
      if (!document.hidden && cardsErrorRef.current) {
        clearTimeout(_retryRef.current);
        _retryDelayRef.current = RETRY_BASE_MS;
        loadCards();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      clearTimeout(_retryRef.current);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  /* ── Reset market lines on card / source change ─────────────────── */
  useEffect(() => {
    setMarketLines(null);
  }, [selectedCardId, cardSource]);

  /* ── Fetch Kalshi market lines after the card loads (non-blocking) ── */
  useEffect(() => {
    if (!selectedCardId || cardSource !== "scraped" || fights.length === 0) return;
    setMarketLinesLoading(true);
    getMarketLines(selectedCardId)
      .then(setMarketLines)
      .catch(() => setMarketLines({ rows: [], errors: ["Failed to load Kalshi market lines"] }))
      .finally(() => setMarketLinesLoading(false));
  }, [selectedCardId, cardSource, fights.length]);

  /* ── Manual reload of Kalshi lines (bypasses the 60s TTL, floor-limited) ── */
  const refreshMarketLines = () => {
    if (!selectedCardId || cardSource !== "scraped") return;
    setMarketLinesLoading(true);
    getMarketLines(selectedCardId, { fresh: true })
      .then(setMarketLines)
      .catch(() => {})            // keep previous quotes on a failed refresh
      .finally(() => setMarketLinesLoading(false));
  };

  /* ── Positions is card-only: leave the tab if we switch to manual ── */
  useEffect(() => {
    if (cardSource === "manual" && nav === "positions") setNav("card");
  }, [cardSource, nav]);

  /* ── Load card when id changes ──────────────────────────────────── */
  useEffect(() => {
    if (!selectedCardId || cardSource !== "scraped") return;
    setLoadingCard(true);
    getCard(selectedCardId).then((data) => {
      const taggedFights = (data.fights || []).map((f) => ({ ...f, _cardId: selectedCardId }));
      setFights(taggedFights);
      setUnavailFights(data.unavailableFights || []);
      setEvent(data.event || null);
      if (taggedFights.length) setSelId(taggedFights[0].id);
    }).catch(() => { setUnavailFights([]); }).finally(() => { setLoadingCard(false); });
  }, [selectedCardId, cardSource]);

  /* ── History: lazy-load on first visit ──────────────────────────── */
  useEffect(() => {
    if (nav === "market" && !history) {
      getHistory().then((d) => setHistory(d))
                  .catch(() => setHistory([]));
    }
  }, [nav, history]);

  /* ── Model vs Market ledger: lazy-load on first visit ─────────────── */
  useEffect(() => {
    if (nav === "market" && !ledger) {
      getLedgerSummary().then((d) => setLedger(d))
                         .catch(() => setLedger({ available: false }));
    }
  }, [nav, ledger]);

  /* ── Manual predict ─────────────────────────────────────────────── */
  async function handleManualPredict(e) {
    e?.preventDefault();
    if (!manualForm.red || !manualForm.blue) return;
    setManualLoading(true);
    try {
      const data = await predictManual(manualForm);
      const fight = { ...data, _cardId: null };
      setFights([fight]);
      setUnavailFights([]);
      // Synthetic event so the Fight Card hero shows a name/banner for manual runs.
      // Use the model-resolved canonical fighter names (data.a/b.name), not the raw
      // form input — so "islam makhachev vs topuria" → "Islam Makhachev vs Ilia Topuria".
      setEvent({
        code: "MANUAL MATCHUP",
        name: `${data.a?.name || manualForm.red} vs ${data.b?.name || manualForm.blue}`,
        venue: manualForm.location || "",
        date: manualForm.eventDate || "",
      });
      setSelId(fight.id);
      setNav("card");
    } catch (err) {
      console.error("Manual predict failed:", err);
    } finally {
      setManualLoading(false);
    }
  }

  /* ── Refresh cards ──────────────────────────────────────────────── */
  async function handleRefreshCards() {
    try {
      await refreshCards();
      const list = await getCards();
      setCards(list);
    } catch (err) {
      console.error("Refresh failed:", err);
    }
  }

  /* ── Derived ────────────────────────────────────────────────────── */
  const payout    = PAYOUTS[payoutKey];
  const breakeven = breakEvenPerLeg(mult, payout.legs);
  const sel       = fights.find((f) => f.id === selId) || fights[0] || null;
  const showBetStrip = layout === "topbar" && nav === "portfolio";

  /* ── Render pages ───────────────────────────────────────────────── */
  function renderPage() {
    if (nav === "card")
      return (
        <FightCardPage
          fights={fights} unavailFights={unavailFights} selId={selId} setSelId={setSelId} event={event}
          marketLines={marketLines} marketLinesLoading={marketLinesLoading}
        />
      );

    if (nav === "proplab")
      return (
        <PropLabPage
          fight={sel} fights={fights} selId={selId} setSelId={setSelId}
          tab={tab} setTab={setTab}
          durLine={durLine} setDurLine={setDurLine}
          durSide={durSide} setDurSide={setDurSide}
          cp={cp} patchCp={patchCp}
          breakeven={breakeven} picks={picks} onToggle={toggleLeg}
          onNavigate={cardSource !== "manual" ? setNav : undefined}
        />
      );

    if (nav === "positions" && cardSource !== "manual")
      return (
        <PositionsPage
          fights={fights}
          marketLines={marketLines} marketLinesLoading={marketLinesLoading}
          onRefreshMarketLines={refreshMarketLines}
        />
      );

    if (nav === "portfolio")
      return (
        <PortfolioPage
          picks={picks}
          onRemove={(k) => setPicks((p) => p.filter((x) => x.key !== k))}
          onClear={() => setPicks([])}
          payout={payout} mult={mult} multDirty={multDirty} breakeven={breakeven}
        />
      );

    if (nav === "market")
      return <ModelVsMarketPage ledger={ledger} history={history} />;

    if (nav === "about")
      return <AboutPage />;

    if (nav === "settings")
      return (
        <SettingsPage
          cardSource={cardSource} setCardSource={setCardSource}
          cards={cards} selectedCardId={selectedCardId} onCardChange={setSelectedCardId}
          payoutKey={payoutKey} setPayoutKey={choosePayout}
          mult={mult} setMult={setMultUser}
          breakeven={breakeven} legs={payout.legs}
          onRefresh={handleRefreshCards}
          manualForm={manualForm} setManualForm={setManualForm}
          referees={referees} weightClasses={weightClasses}
          onManualPredict={handleManualPredict} manualLoading={manualLoading}
        />
      );

    return null;
  }

  return (
    <div className="fp-app" data-layout={layout} data-density={t.density} data-base={t.base}>
      {layout === "sidebar" && (
        <div className="fp-side-backdrop" onClick={() => setLayout("topbar")} aria-hidden="true" />
      )}
      <SidebarV3
        nav={nav}
        setNav={(k) => { setNav(k); if (window.innerWidth <= 900) setLayout("topbar"); }}
        picks={picks}
        onCollapse={() => setLayout("topbar")}
        meta={meta}
        cardSource={cardSource}
      />
      <div className="fp-v3-body">
        <TopNav
          nav={nav} setNav={setNav} picks={picks}
          onExpand={() => setLayout((l) => l === "topbar" ? "sidebar" : "topbar")}
          cards={cards} selectedCardId={selectedCardId} onCardChange={setSelectedCardId}
          cardSource={cardSource}
        />
        {showBetStrip && (
          <BetStrip
            payoutKey={payoutKey} setPayoutKey={choosePayout}
            mult={mult} setMult={setMultUser}
          />
        )}
        <main className="fp-main" style={{ position: "relative" }} id="fp-main">
          {cardsError && (
            <div style={{
              display: "flex", alignItems: "center", gap: 12, padding: "10px 16px",
              background: "var(--panel-2)", borderBottom: "1px solid var(--line-2)",
              fontSize: 13, color: "var(--text-dim)",
            }}>
              <span>⚠ Backend is waking up — this may take ~30 seconds.</span>
            </div>
          )}
          {loadingCard && (nav === "card" || nav === "proplab" || nav === "positions") ? (
            <LoadingSkeleton nav={nav} />
          ) : (
            renderPage()
          )}
        </main>
        <BottomNav nav={nav} setNav={setNav} picks={picks} cardSource={cardSource} />
      </div>
      <TweaksPanel>
        <TweakSection label="Appearance" />
        <TweakRadio label="Base" value={t.base} options={["midnight", "carbon"]} onChange={(v) => setTweak("base", v)} />
        <TweakRadio label="Density" value={t.density} options={["compact", "regular", "spacious"]} onChange={(v) => setTweak("density", v)} />
      </TweaksPanel>
    </div>
  );
}

/* ══ Loading skeletons ════════════════════════════════════════════════
   One shared shell (progress bar + status pill), per-page bodies that
   outline the destination layout coarsely — big regions only, so small
   page tweaks don't un-sync them. */
function LoadingShell({ children }) {
  return (
    <div className="fp-card-loading">
      <div className="fp-cl-progress" aria-hidden="true"><div className="fp-cl-progress-bar" /></div>
      {children}
      <div className="fp-cl-status">
        <span className="fp-cl-dot" aria-hidden="true" />
        Analyzing matchups…
      </div>
    </div>
  );
}

/* Fight Card: hero → chip rail → matchup header + method bars */
function FightCardSkeletonBody() {
  return (
    <>
      <div className="fp-cl-hero">
        <div className="fp-skeleton fp-cl-hero-badge" />
        <div className="fp-cl-hero-lines">
          <div className="fp-skeleton fp-cl-hero-t" />
          <div className="fp-skeleton fp-cl-hero-s" />
        </div>
      </div>
      <div className="fp-cl-body">
        <div className="fp-cl-chips">
          {[0,1,2,3,4].map((i) => (
            <div key={i} className="fp-skeleton fp-cl-chip" style={{ opacity: 1 - i * 0.12 }} />
          ))}
        </div>
        <div className="fp-cl-matchup">
          <div className="fp-skeleton fp-cl-vs" />
          <div className="fp-cl-bars">
            <div className="fp-skeleton fp-cl-bar" />
            <div className="fp-skeleton fp-cl-bar" />
            <div className="fp-skeleton fp-cl-bar" style={{ flex: 0.6 }} />
          </div>
        </div>
      </div>
    </>
  );
}

/* Prop Lab: page header → fight selector bar → grouped tab rail → chart + readout panels */
function PropLabSkeletonBody() {
  return (
    <div className="fp-cl-page">
      <div className="fp-cl-pagehd">
        <div className="fp-skeleton fp-cl-hero-t" />
        <div className="fp-skeleton fp-cl-hero-s" />
      </div>
      <div className="fp-cl-selbar">
        <div className="fp-skeleton fp-cl-sel" />
        <div className="fp-skeleton fp-cl-probs" />
      </div>
      <div className="fp-cl-tabs">
        {[110, 90, 120, 100, 96, 118, 84].map((w, i) => (
          <div key={i} className="fp-skeleton fp-cl-tab" style={{ width: w, opacity: 1 - i * 0.08 }} />
        ))}
      </div>
      <div className="fp-cl-proplab">
        <div className="fp-skeleton fp-cl-chart" />
        <div className="fp-cl-readouts">
          <div className="fp-skeleton fp-cl-readout" />
          <div className="fp-skeleton fp-cl-readout" style={{ flex: 1.6 }} />
        </div>
      </div>
    </div>
  );
}

/* Positions: page header + toggles → filter bar → fight cards with row stripes */
function PositionsSkeletonBody() {
  return (
    <div className="fp-cl-page">
      <div className="fp-cl-pagehd row">
        <div>
          <div className="fp-skeleton fp-cl-hero-t" />
          <div className="fp-skeleton fp-cl-hero-s" />
        </div>
        <div className="fp-skeleton fp-cl-toggle" />
      </div>
      <div className="fp-skeleton fp-cl-filter" />
      {[0, 1].map((c) => (
        <div key={c} className="fp-cl-bbcard" style={{ opacity: 1 - c * 0.25 }}>
          <div className="fp-skeleton fp-cl-bbcard-hd" />
          {[0, 1, 2].map((r) => (
            <div key={r} className="fp-skeleton fp-cl-bbrow" />
          ))}
        </div>
      ))}
    </div>
  );
}

function LoadingSkeleton({ nav }) {
  return (
    <LoadingShell>
      {nav === "proplab" ? <PropLabSkeletonBody />
        : nav === "positions" ? <PositionsSkeletonBody />
        : <FightCardSkeletonBody />}
    </LoadingShell>
  );
}

/* ══ Manual matchup form ═════════════════════════════════════════════ */
function ManualForm({ form, setForm, referees, weightClasses, onSubmit, loading }) {
  const patch = (k, v) => setForm((f) => ({ ...f, [k]: v }));
  return (
    <div className="fp-page fp-page-manual">
      <div className="fp-page-hd">
        <div className="fp-page-ttl">Manual Matchup</div>
        <span className="fp-page-sub">Run a prediction for any fighter pairing</span>
      </div>
      <div className="fp-page-body">
        <form className="fp-manual-form" onSubmit={onSubmit}>
          <div className="fp-mf-grid">
            <div className="fp-cgroup">
              <span className="fp-cgroup-lbl">Red corner (Fighter A)</span>
              <input className="fp-input" type="text" value={form.red} placeholder="Full name e.g. Islam Makhachev"
                onChange={(e) => patch("red", e.target.value)} required />
            </div>
            <div className="fp-cgroup">
              <span className="fp-cgroup-lbl">Blue corner (Fighter B)</span>
              <input className="fp-input" type="text" value={form.blue} placeholder="Full name e.g. Charles Oliveira"
                onChange={(e) => patch("blue", e.target.value)} required />
            </div>
            <div className="fp-cgroup">
              <span className="fp-cgroup-lbl">Rounds</span>
              <div className="fp-select" style={{ width: 120 }}>
                <select value={form.rounds} onChange={(e) => patch("rounds", +e.target.value)}>
                  <option value={3}>3 rounds</option>
                  <option value={5}>5 rounds</option>
                </select>
              </div>
            </div>
            <div className="fp-cgroup">
              <span className="fp-cgroup-lbl">Title fight?</span>
              <button type="button"
                className={"fp-toggle-btn" + (form.isTitle ? " on" : "")}
                onClick={() => patch("isTitle", !form.isTitle)}
                style={{ width: 120 }}>{form.isTitle ? "Yes" : "No"}</button>
            </div>
            <div className="fp-cgroup">
              <span className="fp-cgroup-lbl">Event date</span>
              <input className="fp-input" type="date" value={form.eventDate}
                onChange={(e) => patch("eventDate", e.target.value)} />
            </div>
            <div className="fp-cgroup">
              <span className="fp-cgroup-lbl">Weight class (optional)</span>
              <div className="fp-select">
                <select value={form.weightClass} onChange={(e) => patch("weightClass", e.target.value)}>
                  <option value="">Auto-detect</option>
                  {weightClasses.map((wc) => <option key={wc} value={wc}>{wc}</option>)}
                </select>
              </div>
            </div>
            <div className="fp-cgroup">
              <span className="fp-cgroup-lbl">Referee (optional)</span>
              <div className="fp-select">
                <select value={form.referee} onChange={(e) => patch("referee", e.target.value)}>
                  <option value="">Unknown / any</option>
                  {referees.map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>
            </div>
            <div className="fp-cgroup">
              <span className="fp-cgroup-lbl">Location (optional)</span>
              <input className="fp-input" type="text" value={form.location} placeholder="e.g. Las Vegas, NV"
                onChange={(e) => patch("location", e.target.value)} />
            </div>
          </div>
          <button type="submit" className="fp-btn gold" disabled={loading} style={{ marginTop: 24, minWidth: 200 }}>
            {loading ? "Running prediction…" : "Run prediction"}
          </button>
        </form>
      </div>
    </div>
  );
}
