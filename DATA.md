# Data provenance

**No dataset is distributed with this repository.** The code rebuilds it from
public sources. This document explains what those sources are, why nothing is
redistributed, and how to regenerate.

## Why no data ships here

Three reasons, in order of weight:

1. **Redistribution rights.** Scraped fight statistics are somebody else's
   published work. Being able to read a public page does not grant a right to
   republish a derived corpus of it. Shipping the scraper and the schema is
   defensible; shipping the harvest is not.
2. **Repository weight.** The full artifact set (built parquet files plus
   trained models) runs to several hundred megabytes and would require Git
   LFS. A portfolio repository should clone in seconds.
3. **Model weights are not part of the licence grant.** The Apache-2.0 licence
   covers this source code. Trained models produced from it are a separate
   artifact and are not included. See NOTICE.

## Sources

### Fight statistics — ufcstats.com

Historical bout results, per-round statistics, and fighter attributes.

Retrieved by a **separate, external tool** licensed **GPL-3.0**, which is not
vendored here and must be obtained under its own terms. This project's
`src/ufc/ingest/parse_scraper.py` consumes the CSV files that tool emits — it
does not import, wrap, or derive from GPL-licensed source, so no copyleft
obligation attaches to this codebase.

Upcoming-card scraping (`src/ufc/ingest/scrape_upcoming.py`) and fighter
profile enrichment (`scripts/08_scrape_fighter_stats.py`) are first-party and
hit the same public site directly.

> Anyone running this code is responsible for the target site's terms of
> service and for scraping politely. Rate limits exist for a reason.

### Prediction-market quotes — Kalshi

`src/ufc/ingest/market_lines.py` reads the **public Kalshi REST API**. No
authentication, no key, no account. Kalshi is a CFTC-regulated exchange with a
documented public API, and the quotes are used here as a market-implied
probability baseline to score the model against.

### Not included

Sportsbook and daily-fantasy scrapers are deliberately **absent** from this
repository. They targeted undocumented endpoints and are out of scope for a
public research release.

## Regenerating the dataset

```bash
# 1. Obtain the external historical scraper (GPL-3.0, separate project) and
#    point configs/paths.yaml at its CSV output directory.

# 2. Parse raw CSVs into typed parquet
python scripts/01_ingest.py

# 3. Build model features (leakage-guarded; see tests/test_windows_noleak.py)
python scripts/02_build_features.py

# 4. Train the evaluation tier against the locked temporal split
python scripts/03_train.py

# 5. Run the evaluation gates
python scripts/04_backtest.py           # Gate A - winner
python scripts/05_evaluate_props.py     # Gate B - props
python scripts/05b_evaluate_method.py   # Gate C - method
python scripts/_joint_coherence_check.py # Gate D - joint coherence
```

Paths are configured in `configs/paths.yaml`. The temporal split is locked in
`configs/split.yaml` — train ≤2023, validate 2024, test 2025-26 — so that
evaluation numbers stay honest across runs.

## Tests without data

The test suite runs on a fresh clone. Tests that genuinely require the built
dataset skip themselves cleanly rather than failing; everything else runs
against fixtures in `tests/fixtures/`.

```bash
pytest -q
```
