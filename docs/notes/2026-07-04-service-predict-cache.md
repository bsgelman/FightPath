---
type: decision
date: 2026-07-04
model_version: n/a
status: active
related_code_files: [src/ufc/api/service.py, src/ufc/api/app.py]
related_runs_entry: ""
tags: [api, performance, caching]
---

# service.predict() process-lifetime cache

## Summary

`service.predict()` recomputed a full prediction (inference + 50k-sample MC,
~8–15s/fight) on every call; `/api/cards/{id}` and `/api/market-lines/{id}` each call
it per fight, so a 14-fight card cold-loaded in 1.5–3+ minutes, every time. Added a
bounded `OrderedDict` LRU (200 entries) keyed on every result-affecting arg
(`red, blue, rounds, is_title, event_date, weight_class, referee, location,
n_simulate`). Shipped 2026-07-04, commit `bc6ef1f` (GitHub main + HF Space).

## Why a plain cache is exact-correct here (not a staleness tradeoff)

Model artifacts are fixed for a process's lifetime — they only change on a
deploy/restart, which starts a fresh empty cache anyway. The only call-to-call
difference was MC noise in `sim_samples`, which no caller needs fresh. So caching the
returned `FightPrediction` is exact, not approximate.

## Safety check that made it shippable

A cache that hands out a shared reference is only safe if every consumer is read-only.
Traced all four `service.predict()` call sites in `app.py` plus `serialize.py`,
`prop_cdf.py`, `market_edge.py`, and the `FightPrediction` dataclass (including
`@property` memoization patterns): zero downstream mutation. The only attribute writes
(`pred.record_red/blue`) happen inside `service.predict()` before the object enters
the cache. Lock is held only for dict lookup/insert, released during compute — accepts
a rare simultaneous-first-request double-compute rather than serializing unrelated
fights.

## Evidence

Live-verified: repeat `/api/cards` load 10.7s → 0.25s; `/api/market-lines` reused the
cached predictions (2.4s); a different card computed independently — no cross-card
contamination.

## Related

- Discovered while verifying:
  [2026-07-04 Kalshi market expansion](2026-07-04-kalshi-market-expansion.md)
- Same session, UI: stopped dimming THIN-liquidity exchange rows (`23b846f`) —
  thin liquidity is a sizing signal, not a quality signal.
