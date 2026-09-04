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
- **Efficiency enters volume only when it validates in the production architecture.** Broad passing, receiving, and carry-share bundles remain gated off. Volume v3 admits one narrow exception: lagged rushing EPA per carry improves the any-carry eligibility hurdle across all three posterior holdouts.
- **Partial pooling everywhere.** Small-sample players shrink toward position-level priors.
- **Calibration is the acceptance gate.** Walk-forward backtests score CRPS/log-score against prior-season-PPG and ECR baselines, with PIT/coverage checks that intervals are honest.

## Package

Code lives in an installable package under `src/ffmodel/`:

```
src/ffmodel/
  config.py       scoring rules (verified against this repo's CSVs), paths, season coverage
  data/           hybrid data layer:
    schema.py       canonical player-week schema shared by every source
    ingest.py       nflverse via nflreadpy, parquet-cached (weekly, PBP, snaps, depth charts,
                    injuries, schedules, rosters, id map)
    legacy.py       the CSVs committed to this repo (weekly 1999-2021, yearly 1970-2021,
                    snapcounts 2013-2020, FantasyPros ADP/ECR)
    loaders.py      load_player_weeks(seasons) — one call, one schema, auto source fallback
  features/       leak-free usage, efficiency, snap/role, context, and active-set features
  models/
    volume_team.py  hierarchical Negative-Binomial plays + Binomial pass rate
    volume_share.py ragged active-roster Dirichlet-Multinomial target/carry allocation
    volume_season.py cross-season hierarchical Beta share projection
    volume_season_average.py coherent team-season rates + QB/RB/WR/TE roster allocation
    efficiency_season_average.py exposure-aware posterior season efficiency
    season_scoring.py volume + efficiency end-to-end season scoring pipeline
  simulation/
    scoring.py      stat line → fantasy points (reproduces the CSV point columns exactly)
    season_scoring.py coherent posterior season stat lines and fantasy points
```

### Quickstart

```bash
pip install -e ".[dev]"        # add ".[models]" for pymc/arviz when fitting
pytest                          # fast, network-free test suite
pytest -m slow                  # sampler-heavy Bayesian integration tests

python -c "
from ffmodel.data import load_player_weeks
df = load_player_weeks([2019, 2020])
print(df.head())
"
```

`load_player_weeks` tries nflverse first (richer: player ids, real targets, 18-week seasons kept current) and falls back to the committed CSVs per season when offline.

### Within-season volume models (Phase 3B)

```python
from ffmodel.features import build_features
from ffmodel.models.volume_team import fit_team_volume
from ffmodel.models.volume_share import fit_target_share, fit_carry_share

features = build_features(range(2018, 2021), source="legacy", with_context=False)
team_model = fit_team_volume(features)
target_model = fit_target_share(features)
carry_model = fit_carry_share(features)
```

The team model uses neutral game script by default; Vegas totals and spreads are
not required. The share models allocate integer targets/carries over each
team-week's active support. Removing a player at projection time renormalizes
the posterior concentrations and transfers the full team total to the players
who remain active.

Use `scripts/validate_volume_models.py` for a final-weeks walk-forward smoke
test. It reports MAE, CRPS, 80% interval coverage, R-hat, ESS, and verifies that
every posterior target/carry draw conserves its simulated team total.

### Season-average volume model

The primary preseason projection path estimates stable average volume over a
full season before any matchup adjustments are applied:

```python
from ffmodel.features.season_average import build_season_average_data, SeasonAverageData
from ffmodel.models import SeasonAverageVolumePipeline

data = build_season_average_data(
    range(2014, 2021),
    source="nflverse",
    roster_mode="point_in_time",
)
train = SeasonAverageData(
    data.team_rows[data.team_rows.season < 2020],
    data.player_rows[data.player_rows.season < 2020],
)
test = SeasonAverageData(
    data.team_rows[data.team_rows.season == 2020],
    data.player_rows[data.player_rows.season == 2020],
)
pipeline = SeasonAverageVolumePipeline().fit(train)
prediction = pipeline.predict_samples(test)
```

Team opportunity plays, pass attempts, sacks, targets, and rushes use
prior-season rates with hierarchical team and new-season innovations. Player
volume is a bottom-up playing-time stack: a hurdle model projects whether and
how long a player is active; an all-position model projects offensive-snap
share conditional on activity; target and carry allocators combine those snap
draws with lagged per-snap propensity. Carries also use a draw-level any-carry
hurdle. QB pass shares combine the continuous within-team QB snap workload with
pass attempts per offensive snap. Non-QBs receive zero pass attempts. Every
integer posterior draw enforces:

```text
opportunity plays = pass attempts + player carries
official plays    = pass attempts + player carries + sacks
pass attempts     = player targets + no-target attempts
```

Player pass attempts, targets, and carries each sum to their simulated team
total. Prior and late-season roles anchor returning players; draft and position
priors cover cold starts. The depth-chart QB1 designation remains a preseason
feature rather than a binary outcome. A synthetic replacement bucket for each
QB/RB/WR/TE room captures volume earned by players absent from the point-in-time
roster, preventing later injury replacements from being credited to known Week
1 players. The final season counts are reported as both per-team-game and
per-active-game averages.

The availability layer also has a leakage-safe injury candidate: historical
injury occurrence and completed roster-return episodes form prior burden and
expected-recovery features, while live projections can use an archived Sleeper
snapshot. It is currently opt-in pending a longer posterior validation; see
[injury availability](docs/injury-availability.md).

Efficiency candidates use a directed, leakage-safe dependency: realized
efficiency through season `Y-1` may enter volume `Y` only after clearing the
production-architecture gate. Only rushing EPA in the carry-eligibility hurdle
currently does. The posterior volume `Y`
distribution and efficiency history through `Y-1` can then become inputs to
efficiency `Y`. Same-season realized efficiency never feeds its own volume
projection, and all model-generated training features must be out-of-fold.

Historical point-in-time support uses nflverse regular-season week 1 rosters
and offensive depth charts. This excludes players already cut and preserves
reserve/PUP players as availability outcomes. Live preseason projections can
provide an archived roster snapshot through the same feature contract. Legacy
CSV-only runs remain available with `roster_mode="inferred"`, but are labeled
`inferred_postseason` because those files cannot reconstruct a leakage-safe
preseason roster. nflverse also supplies observed sacks; legacy files use a
conservative league prior rather than a false observed zero.

`scripts/validate_season_average.py` holds out a complete season and compares
the Bayesian distributions with persistence, regularized linear, and optional
XGBoost roster-softmax challengers. Install `.[ml]` only when running XGBoost.

### Posterior efficiency and total-season scoring

Ten exposure-aware efficiency marginals cover passing, receiving, rushing,
touchdown, interception, completion/catch, and fumble-lost outcomes. The
validated mean policy retains the accepted point forecast for nine responses;
only receiving yards per target earned a richer posterior-regression mean.
See [docs/efficiency-v2-validation.md](docs/efficiency-v2-validation.md) for
the 2022-2024 walk-forward result and calibration diagnostics.

```python
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

pipeline = SeasonAverageScoringPipeline().fit(train)
scoring = pipeline.predict_samples(test)
standard = scoring.fantasy_points["standard"]
half_ppr = scoring.fantasy_points["half_ppr"]
ppr = scoring.fantasy_points["ppr"]
```

The simulator enforces count constraints such as completions plus
interceptions not exceeding pass attempts, receptions not exceeding targets,
and touchdowns not exceeding the corresponding realized opportunities. The
first combined total-scoring candidate is implemented but not promoted: its
small MAE improvement did not produce stable CRPS gains and 95% coverage
remained below the promotion floor. See
[docs/season-scoring-v1-validation.md](docs/season-scoring-v1-validation.md).

The next architecture challenger is documented in
[docs/latent-regime-ablation.md](docs/latent-regime-ablation.md): one
leakage-safe player-season state is sampled jointly into role/volume and
efficiency, with role-only and efficiency-only ablations first.

### Kickers and team defenses

The weekly layer covers all six startable slots. K and DST get their own panel
rather than a position dummy on the skill panel, because opportunity share --
targets, carries, snaps -- means nothing for either; they share the row schema,
the walk-forward and the estimator protocol, so all six positions concatenate
into one table of comparable points.

```python
from ffmodel.weekly.specialists import (
    add_defense_features, add_kicker_features, attach_market,
    build_defense_panel, build_kicker_panel, kicker_ladder,
)
from ffmodel.weekly.weather import attach_weather

kickers = add_kicker_features(attach_weather(attach_market(build_kicker_panel(range(2016, 2026)))))
```

Scoring is ESPN-style and configurable in `config.KickerRules` /
`config.DefenseRules` -- distance-tiered field goals read straight from
nflverse's per-bucket columns, and the points-allowed step function. Both
reproduce the real 2024 leaderboards.

Against a persistence baseline, walk-forward over 2023/2024/2025: kickers -2.7%
CRPS, defenses -4.6%. The defense result is the interesting one -- **a defense's
own box-score history adds nothing** over its own recent fantasy points (-0.10%,
a tie), while the opponent plus the closing line is worth -4.6%, forty-six times
as much, and takes within-position ordering from 0.085 to 0.315. A DST projection
is mostly a projection of the opponent, because the points-allowed half of the
response is a step function of the other side's final score.

Rest of season for both promotes the direct regression over the latent-player
simulator; see the document for the diagnosis of the simulator's bias.

One standing caveat for anything consuming kicker numbers: **kicker scoring is
not stationary.** The 2024 dynamic kickoff and the 2025 touchback spot moved
average field position forward, and attempts per game rose 1.93 to 2.03 with
accuracy flat -- more chances, not better kicking. Projections are biased low by
roughly 0.9-1.1 points a game late in those seasons. A trailing league baseline
was built to correct it and *fails*, because the size of a rule change cannot be
learned from data that predates it; the document explains why and what would
work instead.

### Weather

`roof`, `temp` and `wind` are joined by `ffmodel.weekly.weather` and measured
rather than assumed. Pooled over the weekly panel they are a tie -- but that is a
**pooling artefact, not an absence of signal**. Weather depresses scoring exactly
as folklore says (46.5 points a game in the calmest wind bucket against 41.8 at
15-20 mph, with the closing line pricing only a quarter of that), it works
through the pass/run mix, and the shipped model's residual carries it: relative
to calm rows it over-projects quarterbacks by +1.84 points in a 15 mph wind,
receivers +0.93, tight ends +0.76, backs +0.44.

It reads as a null because only 5.6% of rows are exposed. Scored *by exposure*
the same rung is worth **-0.60% CRPS above 15 mph** while costing +0.13% on calm
rows. A gated hinge (`roof`, a 15 mph threshold, and the excess above it, with
temperature dropped as a genuine null) keeps the gain, removes the cost and wins
3 folds of 3. Still under the pooled floor and so not promoted, but it is the
right form of the feature -- and for a per-row start/sit decision the exposed
rows are identifiable in advance. On kickers it is material in its own right
(roof -0.32% CRPS on 3/3 folds, the readings -0.40% on 2/3).

These are 0.1-0.4% effects on three folds and they move between runs at about
that size, which is the resolution limit of the design rather than a precision it
has. Every number is also a ceiling, measured on conditions recorded at the game;
a live version reads the forecast archive in `ffmodel.data.weather`. See
[specialists & weather](docs/specialists-and-weather-2026-09.md).

### Data acquisition

The provider-aware data CLI caches parquet plus provenance manifests and keeps
mutable inputs as immutable `as_of` snapshots:

```bash
ffmodel-data doctor
ffmodel-data bootstrap --seasons 2022 2023 2024 2025
ffmodel-data nflverse --seasons 2022 2023 2024 2025 --datasets pbp
ffmodel-data sleeper
```

CollegeFootballData reads its credential from the Git-ignored project `.env`,
caches every response as Parquet, and enforces local/per-run quota guards. The
Odds API is intentionally deferred. Open-Meteo and Sleeper require no key.
Setup, scheduling, licensing, and point-in-time backtest instructions are in
[docs/data-sources.md](docs/data-sources.md).

Wikipedia HC/OC assignments and coach lineage use a separate, resumable pull:

```bash
pip install -e ".[scrape]"
ffmodel-coaches                       # every committed team-season, 1970-2021
ffmodel-coaches --seasons 2018:2025  # add nflverse-backed recent seasons
```

The command archives exact MediaWiki revisions in the ignored cache and writes
source-attributed assignment, career-history, selected scheme-source, lineage,
and review tables under `data/coaching/wikipedia/`. Wikipedia job titles are not
proof of play-calling responsibility; confirmed effective-date overrides remain
in `data/manual/coach_team_period.csv`. See the coaching section of the data
source guide before using the lineage as a model prior.

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
| 3A | **Season-average volume** (coherent team rates + availability, snaps, per-snap roles, replacement demand) | volume v3 promoted; availability-coupled QB gate, postseason role features and efficiency exposure floor 5 promoted 2026-08 ([followups](docs/pipeline-followups-2026-08.md)); all-data production fit built via `scripts/fit_production.py` (2015-2024, max R-hat 1.005, 0 divergences) |
| 3B | Within-season **volume models** (team plays/pass rate + Dirichlet-Multinomial share) | core models complete |
| 4 | Efficiency models (lagged efficiency -> volume; OOF volume + history -> future efficiency) | efficiency v2 posterior marginals validated; receiving YPT mean promoted |
| 5 | Simulation engine: posterior predictive → weekly & season point distributions | coherent total-season candidate implemented; the v1 coverage failure was traced to a superseded volume layer, not the scoring architecture ([followups](docs/pipeline-followups-2026-08.md)) |
| 6 | Evaluation: walk-forward backtests, CRPS/log-score, calibration | volume v3 and efficiency v2 complete; total-scoring calibration active. **`docs/volume-v3-validation.md` and `docs/season-scoring-v1-validation.md` predate the 2026-08 review and no longer describe this code** — see [the review](docs/pipeline-review-2026-08.md) and [its follow-ups](docs/pipeline-followups-2026-08.md) |
| 7 | Weekly pillar: start/sit lineup optimization | next-week and rest-of-season responses validated; **kickers and team defenses added 2026-09**, so all six startable slots now project on one walk-forward ([specialists & weather](docs/specialists-and-weather-2026-09.md)) |
| 8 | Draft pillar: tiers, pre-season EV, positional trade-offs | K/DST rest-of-season projections available for the full draftable pool |
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
