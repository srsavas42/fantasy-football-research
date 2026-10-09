# Games a player started and did not finish

*October 2026*

The panel records `played` as one flag: did he have a stat line. A receiver hurt at
halftime and a receiver who played every snap are both `played = 1`, so the absence
state the models read saw no difference, and the next week was projected as if nothing
had happened. The data says something had.

Both projection models now read whether the previous game was cut short.
`ffmodel.weekly.partial_game` builds the flag; `scripts/validate_partial_game.py` and
`scripts/validate_partial_weekly.py` reproduce every number below.

## The label

A game is **cut short** when the player played, and his snap share was under 60% of his
average over the previous four club games, from a baseline of at least 50%
(`THRESHOLD`, `MIN_BASELINE`, `WINDOW`). It describes the game that was played, so it is
only ever read lagged: `partial_prev` is whether the previous game was cut short and
`partial_recent` counts the last three.

On 2016-2025 it flags 2.3-3.6% of player-games by position (6.8% among RB/WR/TE with a
real role). It marks the player-weeks that precede trouble:

| | flagged | not flagged |
|---|---:|---:|
| on the next week's injury report | 31.5% | 8.3% |
| misses the next game | 41.7% | 6.7% |
| next-game snap share, if he plays | 0.46 | 0.72 |

Neither the injury report nor the snap counts say *why* snaps fell, so the label is
noisy by construction. Two things that might have cleaned it did not:

- **Game margin does not separate real exits from garbage time.** Among flagged players
  the next-game miss rate is 48% in games decided by 0-6, 45% by 7-13, 53% by 14-16, 43%
  by 17-20 and 36% by 21 or more, against 13-17% for unflagged players in the same games.
  A blowout lowers the signal a little and does not remove it, so no margin filter is used.
- **The baseline needs history.** It needs two prior games in the same season, so weeks
  1-2 are never flagged. A player hurt in week 1 carries no flag into week 3, which is a
  known gap.

## Rest-of-season total

Walk-forward over 2023-2025, only the expected-games fit changing. MAE in points, with
bias (actual minus projected) in brackets.

| population | n | no flag | flag | flag + count | **own group** |
|---|---:|---:|---:|---:|---:|
| everyone | 13,859 | 27.97 (0.0) | 27.95 | 27.95 | **27.94** (0.0) |
| drafted | 9,351 | 32.24 (+0.6) | 32.22 | 32.21 | **32.19** |
| cut short last game | 266 | 26.86 (-8.2) | 26.26 (-2.4) | 26.27 (-2.4) | **25.77** (-0.2) |
| ... drafted | 195 | 30.43 (-10.6) | 29.67 (-4.0) | 29.67 (-4.0) | **28.87** (-0.5) |
| played full last game | 10,542 | 30.11 (-0.3) | 30.10 | 30.11 | 30.11 |

The share of remaining games a cut-short player goes on to play was **0.652**; the model
without the flag said 0.740, with it 0.657 (flag) or 0.637 (own group). Players who
played a full game last week are unchanged.

The shipped arm is the **own group**: players cut short last game are fitted as a group of
their own, like the reserve list and designated Out already were. It has the lowest MAE
and a bias near zero; the plain flag matches the games share more exactly (0.657 against
0.652 actual). The gap between them is inside the noise.

**It is not a significant improvement in MAE.** On the 266 cut-short players the paired
bootstrap gives +0.85 points (95% CI -0.87 to +2.59) and +0.04 across all drafted players
(-0.02 to +0.09). What it fixes is the bias: a 9-point over-projection of games played
for exactly the players it is about.

## Weekly start/sit

The hurdle model's play-probability gets `partial_prev` and `partial_recent`
(`Hurdle(use_partial=True)`). They are **not** given to the points-given-played model:
added there they moved the mean for everyone else (MAE 4.696 to 4.709) without helping.

| population | n | MAE, before | after | Brier, before | after | played / predicted before / after |
|---|---:|---:|---:|---:|---:|---|
| everyone | 13,859 | 4.696 | 4.695 | 0.0846 | 0.0842 | 0.757 / 0.756 / 0.756 |
| after a game cut short | 266 | 4.097 (-1.69) | **3.843** (-1.02) | 0.1734 | **0.1564** | 0.534 / 0.655 / 0.577 |
| ... drafted | 195 | 4.381 | **4.085** | 0.1697 | **0.1533** | 0.518 / 0.639 / 0.565 |
| played full last game | 10,542 | 5.315 | 5.318 | 0.0552 | 0.0552 | 0.910 / 0.881 / 0.883 |

After a cut-short game the play probability falls from 65.5% to 57.7% against an actual
53.4%: closer, and still too high.

## What this does not do

- **It cannot see an exit that did not cut the snap share enough.** A receiver who goes
  down in the fourth quarter has played most of the snaps.
- **Weeks 1-2 are never flagged** (no baseline), and the flag does not carry across
  seasons.
- **It does not replace the Friday report.** The designation for the projected week is
  still the strongest signal; this fills the Wednesday gap before it exists. How much it
  adds once that report is known was not measured separately.
- **Stars flagged this week lose a lot.** After week 4, Ja'Marr Chase (20% of snaps),
  Saquon Barkley (7%), D'Andre Swift (35%), Lamar Jackson (52%, a 6-point win), Ladd
  McConkey (34%) and nine others carry the flag. Their rest-of-season totals fall by
  23-32%; the group is calibrated on average, but 266 events is a small base for any one
  player, and Jackson's 52% may be rest or an injury. Read the flag as a prompt to check
  the news.
- **Three holdout seasons**, with no significance across folds beyond the bootstraps above.

## Not built

- **Full-game equivalents.** Counting expected snap-weighted games, with the rate per full
  game on the field, would split the total into availability, participation and
  efficiency. The label here is the cheap version of that and it was enough to fix the
  bias; the full split is not needed to ship this.
- **A clean role baseline** that drops flagged games from the snap and target-share
  averages, so a half-game does not read as a role collapse. Untested.
- **Play-by-play exits.** The cache has pbp; the time a player left the field would
  separate injury from rotation. Not checked for completeness.
