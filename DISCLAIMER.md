# Disclaimer

## Not affiliated with the UFC

FightPath is an independent research project. It is **not affiliated with,
endorsed by, sponsored by, or connected to** Zuffa, LLC, TKO Group Holdings,
the Ultimate Fighting Championship, or any promotion, athlete, official, or
sanctioning body.

UFC®, Ultimate Fighting Championship®, and Octagon® are registered trademarks
of their respective owners. They appear in this repository only where needed
to describe the sport being modelled, as nominative descriptive use. No claim
of ownership or association is made or implied.

## Research software, provided as-is

This is a personal research and portfolio project built to study forecasting
and probability calibration in a hard, low-signal domain. It is not a
commercial product and carries no warranty of any kind. See the LICENSE for
the full disclaimer of warranties and limitation of liability.

Predictions are statistical estimates from historical data. Combat sports are
high-variance: a well-calibrated model is still wrong on individual fights,
frequently and by design.

## Not betting, financial, or investment advice

Nothing here is a recommendation to place any wager or enter any position.

This repository contains code that compares model probabilities against
market-implied probabilities and measures forecast quality against closing
lines. That machinery exists to **evaluate the forecaster**, not to advise a
bettor. Closing-line value is used here as a scoring rule for probabilistic
forecasts, in the same spirit as Brier score or CRPS.

Nothing in this repository establishes that the model is profitable to bet,
and no such claim is made. Accuracy and calibration on historical fights do
not imply an edge against a market price, and the author does not represent
that any position derived from this software is +EV.

The author accepts no responsibility for any loss arising from use of this
software or its output. If you gamble, gamble responsibly and within your
means; in the US, the National Problem Gambling Helpline is 1-800-522-4700.

## Data

No scraped dataset is redistributed in this repository. Anyone running the
ingest code is responsible for complying with the terms of service of the
sources they access, and for their own jurisdiction's rules. See
[DATA.md](DATA.md).
