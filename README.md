# Fantasy Football Distributional Modeling

Statistical models that produce **distributions** of fantasy football outcomes — not point estimates — on both season-long and weekly horizons, supporting three pillars:

1. **Draft value** — tier gaps, pre-season expected value, and mid-draft positional trade-offs.
2. **Volume prediction** — opportunity is king; predict each player's share of team plays.
3. **Weekly outcomes** — per-week outcome distributions for start/sit and lineup optimization.

## Architecture

Fantasy points for a player-week are simulated bottom-up, with each layer a hierarchical Bayesian model (PyMC):

```
team plays & pass rate  →  opportunity share  →  per-touch efficiency  →  scoring
   (NegBinom/Binomial)     (Dirichlet-Multinomial      (hierarchical         (PPR /
                            over the active roster)     Normal/Poisson)       Half / Std)
```

Sampling all layers over posterior draws yields the full outcome distribution; season projections aggregate simulated weeks. Because opportunity shares renormalize over whoever is *active*, a starter's injury automatically flows volume to backups.

Key modeling choices:

- **Empirical roles over listed depth charts.** Role tiers come from EWMA trailing snap share (route participation where available); listed depth charts + ADP/ECR are only a cold-start fallback for week 1, rookies, and team changes.
- **Efficiency feeds volume.** Trailing per-opportunity efficiency (yds/route-run, yds/touch) enters the share model — coaches route opportunity to efficient players.
- **Partial pooling everywhere.** Small-sample players shrink toward position-level priors.
- **Calibration is the acceptance gate.** Walk-forward backtests score CRPS/log-score against prior-season-PPG and ECR baselines, with PIT/coverage checks that intervals are honest.

## Package

Code lives in an installable package under `src/ffmodel/`:

```
src/ffmodel/
  config.py       scoring rules (verified against this repo's CSVs), paths, season coverage
  data/           hybrid data layer:
    schema.py       canonical player-week schema shared by every source
    ingest.py       nflverse via nfl_data_py, parquet-cached (weekly, snaps, depth charts,
                    injuries, schedules, rosters, id map)
    legacy.py       the CSVs committed to this repo (weekly 1999-2021, yearly 1970-2021,
                    snapcounts 2013-2020, FantasyPros ADP/ECR)
    loaders.py      load_player_weeks(seasons) — one call, one schema, auto source fallback
  simulation/
    scoring.py      stat line → fantasy points (reproduces the CSV point columns exactly)
```

### Quickstart

```bash
pip install -e ".[dev]"        # add ".[models]" for pymc/arviz when fitting
pytest                          # network-free test suite

python -c "
from ffmodel.data import load_player_weeks
df = load_player_weeks([2019, 2020])
print(df.head())
"
```

`load_player_weeks` tries nflverse first (richer: player ids, real targets, 18-week seasons kept current) and falls back to the committed CSVs per season when offline.

### Cross-season volume & breakout report (Phase 3A)

```python
from ffmodel.features import crossseason as cs
from ffmodel.models import volume_season as vs
from ffmodel.projections import season_volume as sv

trans = cs.build_transitions(range(2015, 2021), source="legacy")   # returning players, Y->Y+1
train, test = trans[trans.transition < "2019->2020"], trans[trans.transition == "2019->2020"]

target_model = vs.fit_target_share(train)     # hierarchical Beta (needs the ".[models]" extra)
carry_model  = vs.fit_carry_share(train)
sv.breakout_report(test, target_model, carry_model, threshold=0.05)  # ranked P(volume uptick)
```

The Beta share model is centered on year-over-year persistence (share is sticky) and adjusts for both sides of the opportunity ledger:

- **Vacated opportunity** — volume freed when teammates leave (from roster diffs).
- **Incoming competition** — volume claimed by players *arriving* at the same position: signed/traded veterans (their prior-team share) and drafted rookies (draft capital, `features/draft.py`). This is the other half — freed targets mean little if the team also signed a star and drafted a receiver.

Modeling competition matters: for RBs the competition coefficient is strongly negative and it *unmasks* the vacated-opportunity signal (its coefficient roughly 6× larger once competition is controlled for). Net opportunity (vacated − competition) tracks realized carry-share change far better than vacated alone (Spearman ~0.26 vs ~0.03) — see `scripts/validate_crossseason.py`. It roughly matches a persistence baseline on point error but adds calibrated ~80% intervals and per-player breakout probabilities. v1 covers returning players as the subjects (rookies enter only as competition, not yet as projected players); the veteran-competition proxy and rookie draft data use the offline combine file, upgraded to nflverse draft picks when online.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Package scaffolding, config, scoring, tests | ✅ |
| 1 | Hybrid data layer (nflverse + legacy CSVs, parquet cache) | ✅ |
| 2 | Features: usage shares, empirical role tiers, trailing efficiency, game script, active-set/injury logic | ✅ |
| 3A | **Cross-season volume** (year-over-year share via hierarchical Beta) + breakout report | ✅ |
| 3B | Within-season **volume models** (team plays/pass rate + Dirichlet-Multinomial share) | next |
| 4 | Efficiency models (yds/touch, TD, catch rate) | |
| 5 | Simulation engine: posterior predictive → weekly & season point distributions | |
| 6 | Evaluation: walk-forward backtests, CRPS/log-score, calibration | |
| 7 | Weekly pillar: start/sit lineup optimization | |
| 8 | Draft pillar: tiers, pre-season EV, positional trade-offs | |
| 9 | Alt-data signal layer: BlueSky/news → live role-prior adjustments (not backtestable, so live-only) | |

# Fantasy Football Data Sets

This repo began as a fork of [fantasydatapros/data](https://github.com/fantasydatapros/data); the CSVs below remain available and power the legacy loaders and offline tests.

## Strength of Schedule data
Strength of Schedule data is available in the sos directory. Data is available going back to 1999. To load this data in pandas using the following the following url format:
https://raw.githubusercontent.com/fantasydatapros/data/master/sos/{year}.csv

For example, in pandas do the following:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/sos/1999.csv', index_col=0)
    df.index = df.index.rename('Team')

## Weekly Fantasy Stats
Weekly stats going back to 1999 are available are exposed through the following url format

https://raw.githubusercontent.com/fantasydatapros/data/master/weekly/{year}/week{week}.csv

To grab weekly data for year 2019, week 1 in pandas, you would do:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/weekly/2019/week1.csv')

## Yearly Fantasy stats
Yearly fantasy stats are available going back to 1970.

The url format:
https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/{year}.csv

To grab yearly data for 2019 in pandas, do the following:

    import pandas as pd
    df = pd.read_csv('https://raw.githubusercontent.com/fantasydatapros/data/master/yearly/2019.csv')
