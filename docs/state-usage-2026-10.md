# Usage by game state

*October 2026*

The history the weekly models read is built from box scores, and a box score counts a
target thrown in the fourth quarter of a 31-point game the same as one thrown in a tied
first half. `ffmodel.weekly.state_usage` rebuilds player usage from play-by-play, one game
state at a time, and `scripts/validate_state_usage.py` tests whether the models use it.
**It does not help.** The module stays in the repository, opt-in and off, as a measured
negative.

## What was built

Every share is the player's plays in a state over his team's plays in that state, lagged
like every other history column (one-game decay; the neutral shares also carry a six-game
view).

| group | features |
|---|---|
| **neutral** | target share and rush share when win probability was 20-80% with more than two minutes left in the half |
| **clean** | target share and rush share with decided games removed (win probability under 5% or over 95%) |
| **script** | share of his targets that came trailing by 9+; share of his carries that came leading by 9+ |
| **team neutral** | the team's pass rate in neutral states, an 8-game decay across seasons |
| **last game** | how much of his team's last game was garbage time |

"Snaps per team play" was not built: snap share already is that ratio.

## Result

Walk-forward over 2023-2025 on the configuration `scripts/project_live.py` runs, adding
one group at a time to the points model (13,859 player-weeks). MAE of the mean, in points:

| population | n | base | neutral | clean | script | team neutral | last game | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| everyone | 13,859 | 4.696 | 4.692 | 4.707 | 4.706 | 4.703 | 4.703 | 4.699 |
| weeks 1-4 | 3,139 | 4.672 | **4.647** | 4.667 | 4.674 | 4.670 | 4.669 | **4.651** |
| weeks 5+ | 10,720 | 4.703 | 4.705 | 4.718 | 4.715 | 4.712 | 4.712 | 4.713 |
| WR | 5,402 | 4.869 | 4.869 | 4.868 | 4.886 | 4.869 | 4.874 | 4.873 |

Gain in MAE over the base, points per player-week, paired bootstrap over player-seasons
(positive is better, `*` marks an interval that excludes zero):

| population | neutral | clean | script | team neutral | last game | all |
|---|---:|---:|---:|---:|---:|---:|
| everyone | +0.002 | -0.010 | -0.010 * | -0.012 * | -0.008 | -0.006 |
| drafted | -0.005 | -0.014 * | -0.012 | -0.013 * | -0.010 | -0.008 |
| WR | -0.003 | -0.010 | -0.021 * | -0.011 | -0.011 | -0.009 |
| RB, TE, QB | none significant | | | | | |
| **weeks 1-4** | **+0.023 \*** | +0.006 | +0.002 | +0.003 | +0.005 | **+0.021 \*** |

- **No group helps overall**, and four of them make the average a little worse (the
  "script" and "team neutral" intervals exclude zero on the wrong side).
- **The one real gain is early in the season.** Neutral-state shares improve weeks 1-4 by
  0.023 points per player-week (0.5%, interval +0.004 to +0.040). That is when the
  in-season history is thin and a cleaner read of last year's role is worth the most. By
  week 5 the same features are neutral to slightly negative.
- **Nothing is shipped.** A 0.5% early-season gain, from the one of many comparisons that
  happened to clear its interval, is not enough to change the model; the "all" arm gives
  the same gain with more features and a loss later.

Why they do not help is not measured. The likely reasons, untested: snap share, the
last-game values, spread and implied totals, and pass rate over expected already carry
most of what state-adjusted usage would, and a lagged share from one game in a
sub-state is a small, noisy sample (a few neutral targets per game).

## Not tried

- **State-adjusted shares inside the rest-of-season model.** The weekly result is a null
  at that horizon too, so it was not run.
- **An early-season gate** (the neutral features only while the season is young). It
  would fit the one positive result and is a multiple-comparison risk worth a second
  season of data before building.
- **State-adjusted efficiency** (yards per target or per carry by state). Only shares and
  rates of work were built here.
