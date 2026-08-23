# Expected fantasy points: available, unused, and it does not help (2026-08-20)

nflverse ships `ff_opportunity`, which computes expected fantasy points from
play context — down, distance, field position, air yards — and reports actual,
expected and their difference for every player-week. It is already wired into
this package's data layer (`ingest.load_ff_opportunity`, `pull.py`,
`providers/nflverse.py`) and is referenced by **no feature and no model**.

It loads cleanly for 2015-2024 and gives `total_fantasy_points_exp` plus every
component: `pass_touchdown_exp`, `rec_yards_gained_exp`, `rush_touchdown_exp`
and their `_diff` counterparts, which are the luck terms.

## The folk claim does not hold here

Prior-season expected points per game is widely held to predict next season
better than actual points per game, because it strips unsustainable touchdown
luck. On 2047 consecutive player-season pairs with 8+ games in both years, it is
the other way round at every position:

| | prior actual/G | prior xFP/G |
|---|---:|---:|
| all | **0.768** | 0.756 |
| QB | **0.516** | 0.467 |
| RB | **0.690** | 0.657 |
| WR | **0.717** | 0.706 |
| TE | **0.722** | 0.688 |

## And it adds nothing incrementally

Correlation is the wrong test — two correlated predictors can both contribute,
and `actual - expected` is precisely the luck term whose coefficient should be
below one if regression is forecastable. Walk-forward over 2019-2024, predicting
next season's points per game, position controls throughout:

| features | MAE | vs actual only |
|---|---:|---:|
| actual/G only | **2.6556** | — |
| xFP/G only | 2.7774 | +4.59% |
| actual/G + xFP/G | 2.6587 | +0.12% |
| actual/G + luck | 2.6587 | +0.12% |
| actual/G + xFP/G + games | 2.6514 | −0.16% |

Nothing clears the 0.25% materiality floor. The third and fourth rows agreeing
to four decimals is the intended sanity check: `{actual, xFP}` and
`{actual, luck}` span the same space, so a difference there would have meant a
bug rather than a finding.

## Why this is consistent rather than surprising

The pipeline already regresses touchdown luck, through `shrunk_pass_td_rate`,
`shrunk_rec_td_rate` and `shrunk_rush_td_rate`. Expected points is a different
route to the same correction, and the correction is already applied. This is the
same shape as the teammate-quality and ADP-interaction results: a re-transform of
information the frame already carries adds nothing.

## Scope

Tested on raw season aggregates outside the pipeline, which is the cheap version.
It does not rule out a within-season use — the strongest published claims for
expected points concern mid-season regression, not season-over-season — and it
does not test the per-component expectations (red-zone-weighted touchdown
expectation in particular) as separate features. What it does rule out is the
straightforward version: swap in xFP, or add it alongside, and get a better
season projection.

Worth noting one genuine gap this surfaced: the frame carries EPA and air yards
but **no red-zone or goal-line usage features at all**, and play-by-play is
pulled but never used for feature construction.
