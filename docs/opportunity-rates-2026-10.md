# Usage per opportunity

*October 2026*

The history the weekly models read carries shares of the team's work (target share, rush
share, snap share). A target share is two decisions multiplied: how often he is on the
field for a pass, and how often he is targeted when he is there. A share cannot tell a
receiver whose snaps fell from one the offense stopped looking for.
`ffmodel.weekly.opportunity_rates` separates them, and
`scripts/validate_opportunity_rates.py` tests whether the points model uses it.
**It does not help.** The module stays in the repository, opt-in and off, as a measured
negative.

## What was built

Routes run are not published. The denominator is the proxy the participation file
supports: **dropback plays with the player on the field** (`pass_snaps`) and designed runs
with him on the field (`run_snaps`), 100% of dropbacks in every season 2016-2025. It
counts a back or tight end who stayed in to block, so it understates targets per route for
them and is close for receivers. Each rate is a ratio of two exponentially weighted sums
(one-game decay, plus a six-game view for most), so a three-snap game does not count like
a sixty-snap game; a game with no opportunities leaves the rate alone.

| group | features |
|---|---|
| **simple** | targets and rushes per offensive snap (the panel's own snap counts) |
| **participation** | share of the team's dropbacks and of its designed runs he is on the field for |
| **earn** | targets per dropback snap (targets per route, roughly); rushes per run snap |
| **yield** | receiving yards, fantasy points and touches per snap |

## Result

Walk-forward over 2023-2025 on the configuration `scripts/project_live.py` runs, adding one
group at a time to the points model (13,859 player-weeks). MAE of the mean, in points:

| population | n | base | simple | participation | earn | yield | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| everyone | 13,859 | 4.696 | 4.699 | 4.701 | 4.702 | 4.704 | 4.701 |
| weeks 1-4 | 3,139 | 4.672 | **4.661** | 4.670 | 4.669 | 4.668 | 4.669 |
| weeks 5+ | 10,720 | 4.703 | 4.710 | 4.710 | 4.711 | 4.715 | 4.710 |
| WR | 5,402 | 4.869 | 4.872 | 4.877 | 4.880 | 4.885 | 4.882 |
| RB | 3,492 | 4.735 | 4.731 | 4.735 | 4.733 | 4.731 | 4.739 |
| TE | 2,318 | 3.989 | 3.983 | 3.989 | 3.989 | 3.989 | 3.996 |

Gain in MAE over the base, points per player-week, paired bootstrap over player-seasons
(positive is better, `-` marks an interval entirely below zero):

| population | simple | participation | earn | yield | all |
|---|---:|---:|---:|---:|---:|
| everyone | -0.006 | **-0.016 -** | **-0.011 -** | -0.008 | **-0.015 -** |
| WR | -0.007 | **-0.025 -** | -0.011 | -0.014 | -0.020 |
| QB | **-0.033 -** | -0.019 | -0.023 | **-0.030 -** | -0.003 |
| RB, TE | none significant | | | | |
| weeks 1-4 | +0.012 | +0.001 | +0.003 | +0.005 | -0.001 |

- **No group improves the points model.** Participation and earn rates make it slightly
  worse overall, and the participation group is clearly worse for receivers.
- **Nothing is significant in the positive direction**, including early in the season.

## Do the rates at least describe usage better?

Slightly, and not enough to carry into points. Predicting a player's *next-game* targets
or carries from linear fits trained before 2023 and tested on 2023-2025, with shares alone
against shares plus the rates:

| outcome | n | R-squared, shares | R-squared, + rates | MAE change |
|---|---:|---:|---:|---:|
| WR targets | 7,425 | 0.489 | 0.497 | -0.2% |
| TE targets | 3,670 | 0.445 | 0.465 | +1.0% |
| RB targets | 4,586 | 0.338 | 0.352 | +0.4% |
| RB carries | 4,586 | 0.553 | 0.562 | +1.2% |

The rates carry about as much information about next week's usage as the shares already
do (their correlation with this game's targets is 0.46 and 0.45, against 0.48 for target
share), and a small extra amount for tight ends and backs. That is a 1% improvement on the
usage and nothing on points.

Why they do not help points is not measured. A likely reason, untested, is that the model
already carries the shares, snap shares, last-game values and expected-points inputs, so
the rates mostly restate them, while the points response is dominated by touchdowns and
yardage variance that no usage rate predicts.

## Not tried

- **True routes run.** Participation gives dropbacks on the field, not routes. A real route
  count (a charting provider's) would be a cleaner denominator for backs and tight ends.
- **Rates inside the rest-of-season model**, and **yards or touchdowns per target**.
  The weekly null suggests the same at that horizon, and efficiency rates are noisier
  than the volume rates tested here.
