# Rest-of-season projections that multiply out

*October 2026*

Every row of the rest-of-season file is now one calculation:

```
rest_of_season_points  =  expected_games_played  x  points_per_active_game
```

`points_per_scheduled_game` is the same total over the games on the schedule, and
`p10/p50/p90` are quantiles of the same total. Each column is a different way of
breaking down one number, so none can disagree with another. The model is
`ffmodel.weekly.ros_reconciled`; `scripts/validate_reconciled_ros.py` reproduces every
number below, and `reports/reconciled_ros.json` holds the last run.

Out of sample on 2023-2025 it is **more accurate than the total it replaces** (MAE 28.0
against 29.7, bias 0.0 against +0.9) and far better on the players the decision is about
— drafted players who just missed a game, 28.6 against 35.2 — with one real weakness,
the first three weeks of a season, where it is 1-2% worse.

## Why the old rows did not multiply

The first version took its three numbers from three places. The total was the direct
regression **blended toward the draft-board curve**, which knows nothing about a
player's absences. Expected games came from an availability model, discounted for those
absences. The rate was a third fit. They disagreed, and they disagreed most for a star
who has just missed time: Puka Nacua read 215.9 points over 6.7 expected games, and the
rate that reconciled them was 32 a game. The rate was then modelled directly, which
fixed the rate (MAE 3.39 against 4.29 for the ratio) and left the total and the games
mutually inconsistent, because a total cannot be rebuilt from two numbers that were never
fitted to multiply.

The obvious repair was a correction factor on rate x games. It fails honestly. Fitted on
players who had just missed a game it came out at **1.27, 1.36 and 1.35** in the three
folds, and folding that into the games made "expected games" 0.554 of the games left for
drafted returners against an actual 0.448. The identity held and the games column no
longer meant games. A 30% fudge factor is a sign that something structural is missing.

## The rate is weighted by games played

A total is a sum over games, so the rate that multiplies out exactly is the
**games-weighted** average, `E[games x rate] / E[games]`, not the plain average of points
per game played. They differ whenever games played and scoring are correlated, and they
are: a player who plays more games is usually one whose role held up.

`WeightedRate` fits the rate by weighted least squares, each row weighted by the games he
actually played, on the same design as the direct regression. That removed the overall
bias with no correction at all (total bias +2.8 to +0.7 with the same games model). As a
*rate* it gives up nothing — 3.42 MAE against 3.39, and 2.90 against 2.91 when each row is
weighted by its games — but it runs about 0.35 higher, and about 1.0 higher for top-50
returners. That is the covariance showing up, and it is the price of the identity.

## "Returners" was two opposite errors

`missed_last` (did he not play the previous game) lumps an injured-reserve placement, a
designated Out, a Questionable who sat, a healthy scratch and a player on the field with
no stat line. `played` is one 0/1 flag — did he have a stat line — so the target cannot
tell them apart, and nothing gave the availability model *why* last game was a zero.

Among fantasy-relevant players 2022-2025, the zero rows are:

| why the row is a zero | share | had offensive snaps |
|---|---:|---:|
| reserve list | 39.1% | 0% |
| designated Out | 14.0% | 0% |
| Questionable/Doubtful, did not play | 10.3% | 6.4% |
| inactive, no designation | 9.0% | 0% |
| active, no designation, no stat line | 27.5% | 38.9% |

The last row is a labelling flaw worth knowing about: 39% of those players took offensive
snaps. They were on the field and recorded no stat, and the panel marks them as not having
played and overwrites their snaps with zero — about 11% of all zero rows.

Bias by why he missed his last game (actual minus projected, positive means the
projection was too low), shipped total against the reconciled model:

| last game | n | shipped | reconciled |
|---|---:|---:|---:|
| reserve list | 1,255 | -10.4 | +3.3 |
| designated Out | 500 | +6.3 | +8.0 |

The shipped total over-projected reserve-list players by 57% (80% for top-50 ADP) and
under-projected short-term injuries, and the two averaged to a bias of -4 on drafted
returners. The aggregate hid both. Short-term injuries are still under-projected after
the change, by a little more than before (+8.0 against +6.3) while MAE on them improves
(25.2 against 29.1): the draft-board blend that was propping those players up is gone.

**Expected games is therefore fitted separately for each absence state**, each group
crossed with how many games he has missed in a row, with a global fit as the fallback for
groups under 150 training rows.

## What did not help

| idea | result |
|---|---|
| **Games missed in a row**, as buckets, separate fits, or a continuous term | The gradient is real (0.555 of remaining games played after one missed game, 0.166 after nine or more) but `weeks_since_played` was already a feature. Fitting by streak alone was slightly worse (top-50 returners 37.2 against 36.0). |
| **Injury type** as body part | Adds about one point of MAE for players who missed a game with a designation, on top of separate fits. Return rates do differ by type (0.57 knee and groin/hip to 0.74 leg muscle) but the model's misses are a similar size for every type: a level shift, not a type effect. |
| **Injury mechanism** (muscle, joint, head, bone, tendon, spine, illness, non-medical) | No different from body part (top-50 returners 36.2 against 36.6). |
| **Nature of injury** (tear, pull, soreness) | Not available. Across 87,686 injury-text entries (267 distinct strings) there is no tear, strain, sprain, pull, soreness, contusion, fracture or surgery; the feed names a body part and nothing else. Sleeper carries richer notes but only as a present-day snapshot, with no history to learn an effect from. A weekly archive of those snapshots would let a model learn it next season. |
| **Fitting the rate separately** for each absence state | No improvement (top-50 returners 36.4 against 35.7). |

Injury type is in the shipped model because the variant that carries it measured best, not
because it has been shown to matter. The one visible effect came from **coverage**: the
game-status injury field is filled on 48% of report rows and the practice field on 99.5%, so
the type falls back to the practice text, which doubled the player-weeks carrying a type
(8% to 17%) and trimmed the bias for top-50 returners from +16.8 (mechanism from the
game-status field alone) to +14.9. It is also filed earlier in the week, which is what a
Wednesday run has. That is inside the noise three seasons can resolve.

## Return-to-play rates changed around 2021

The share of remaining games played by players who had missed three to five games in a row:

| | 2016-2020 | 2021-2025 |
|---|---:|---:|
| missed 3-5 in a row | 0.21-0.30 | 0.33-0.37 |
| drafted, missed 1-5 | 0.36-0.47 | 0.49-0.53 |
| played last game | 0.82-0.87 | 0.80-0.83 |

A fit over all years averages the two eras and under-projects returners in every recent
season. An era indicator, a linear trend, or training on only the last four or five seasons
all help: the drafted-returner bias falls from +10.4 to between +6.3 and +9.2. (An earlier
draft said games then became nearly calibrated, 0.443 against 0.448. That does not hold for
drafted returners, who still play 0.531 of their remaining games against 0.454 projected;
see the next section.) **Why it changed is not known.** Every holdout season is in the newer era, which flatters the fix; for a live
season it matters because half of the training years are in the older one. The shipped
model uses the era indicator.

## Results

MAE in points, with bias (actual minus projected) in brackets, 2023-2025 holdouts. `shipped`
is the blended total the file carried before; `direct` is the regression without the blend.

| population | n | shipped | direct | **reconciled** |
|---|---:|---:|---:|---:|
| everyone | 13,859 | 29.7 (+0.9) | 30.1 (+1.4) | **28.0** (+0.0) |
| drafted | 9,351 | 34.4 (+2.2) | 35.0 (+2.9) | **32.2** (+0.6) |
| missed last game | 3,051 | 24.3 (-3.8) | 24.0 (-0.6) | **20.7** (+1.8) |
| drafted, missed last | 1,337 | 35.2 (-4.3) | 34.6 (+3.2) | **28.6** (+7.9) |
| top-50, missed last | 331 | 45.5 (-9.2) | 46.1 (+1.9) | **35.7** (+13.2) |
| reserve list last game | 1,255 | 24.8 (-10.4) | 24.2 (-6.6) | **18.5** (+3.3) |
| top-50, Out/Q/D last game | 292 | 41.7 (+5.8) | 43.1 (+13.8) | **37.6** (+10.2) |

### By week

The draft board knows things a model with no in-season history does not, which is why the
shipped total blended toward it early. The reconciled model has no blend, and it shows:

| | shipped | reconciled |
|---|---:|---:|
| week 1 | 52.3 (-5.6) | 53.5 (-8.6) |
| week 2 | 47.8 (-1.5) | 48.8 (-1.6) |
| week 3 | 43.8 (-0.2) | 44.3 (-0.3) |
| **week 4** | 41.3 (-1.3) | **40.9** (-1.2) |
| week 5 | 42.0 (+5.9) | 39.5 (+0.8) |
| week 6 | 38.3 (+4.0) | 35.7 (+0.3) |
| weeks 10+ | 18.8 (+0.1) | 16.3 (+0.7) |

Weeks 1-3 are 0.5-1.2 points worse (1-2%) and the top-50 slice of weeks 1-4 is 62.7 against
61.5. From week 4 it is at least as good, and clearly better from week 5. The cost of
dropping the blend is visible in who moves: high-ADP players with little history rank lower
(Nico Collins 69th to 159th at week 4), and whether that is right or a loss of board
information is exactly the early-season question above. Restoring the board's information
at the *rate* rather than the total would keep the identity and is untested.

## Intervals

Empirical rather than composed. The residuals of the total are measured on the two most
recent complete seasons, each predicted by a model fitted strictly before it, and binned by
projection size and by whether he just missed a game and was drafted. p10/p50/p90 are the
projection plus those residual quantiles. The share of outcomes below each quantile, which
should be 10%, 50% and 90%; "in 80" is the share inside p10-p90 (nominal 80%):

| | shipped: <p10 | <p50 | <=p90 | in 80 | CRPS | **reconciled:** <p10 | <p50 | <=p90 | in 80 | CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| everyone | 9.9 | 45.5 | 89.2 | 79.4 | 21.07 | 11.3 | 51.0 | 89.2 | 77.9 | **20.02** |
| drafted, missed last | 26.5 | 58.3 | 87.0 | 60.5 | 23.92 | 12.6 | 49.9 | 86.1 | 73.5 | **20.23** |
| top-50, missed last | 32.9 | 62.5 | 87.0 | 54.1 | 31.72 | 14.2 | 49.5 | 82.2 | 68.0 | **25.61** |
| reserve list last game | 30.1 | 70.1 | 90.9 | 60.8 | 15.47 | 14.6 | 59.7 | 88.9 | 74.3 | **12.45** |

Totals have an atom at zero (12% are exactly zero: a season that ends), so the lower tail is
counted strictly below the quantile; "at or below" would count those zeros as under it and
overstate the shipped lower tail. The shipped intervals were badly wrong for returning
players: a third of top-50 returners landed below the shipped p10, and only 54% landed inside
a nominal 80% band. The new ones are much better and **still too narrow for returners**
(68-75% inside), and slightly narrower overall (77.9% against 79.4%).

## Where the returner gap comes from: games played, and they come back sooner

Drafted players who missed their last game, 2023-2025, split into the games he plays and
the points he scores in them. Among the 1,121 with at least three games left (bias +10.1):

| | projected | actual |
|---|---:|---:|
| games played | 4.16 | 4.94 |
| share of games left | 0.455 | 0.540 |
| points per game played (games-weighted) | 10.35 | 10.72 |

Of the +10.1, **+8.2 is games and +1.8 is rate.** They are not out for longer than the model
thinks; they are back sooner, and score about what the model says when they are. The
shortfall is in the designated-absence cases and not the reserve list:

| last game missed as | n | share projected | share actual |
|---|---:|---:|---:|
| reserve list | 518 | 0.290 | 0.292 |
| designated Out | 325 | 0.564 | 0.647 |
| Questionable/Doubtful | 194 | 0.660 | 0.745 |
| inactive, no designation | 501 | 0.590 | 0.668 |

It is worst for short absences (the gap in share is 0.03 after one missed game, 0.07-0.08
after three or more) and it moves by season: 0.605 actual against 0.473 projected in 2024,
0.482 against 0.449 in 2025.

Two cautions. First, "early" is against the model's average, not the team's timetable; with
no severity or return-date information nothing here can say whether the player came back
earlier than the club expected. Second, the same cut has the opposite sign late in the
season: with fewer than three games left, returners play 0.214 of them against 0.412
projected (bias -3.2), presumably benchings and shutdowns that the model does not see. The
+7.9 for all drafted returners is the net of the two. The outcomes are also bimodal, which is
part of why the returner intervals run narrow: 29% never play again and 42% are back the
next game.

### Return-timing terms in the games model

Tested after finding the gap is games, on the same 2023-2025 holdouts, changing only the
expected-games fit (the rate is unchanged):

| variant | drafted returners MAE (bias) | top-50 returners | everyone | share of games, drafted returners (actual 0.531) |
|---|---:|---:|---:|---:|
| current | 28.6 (+7.9) | 35.7 (+13.2) | 28.0 (0.0) | 0.454 |
| **returner x horizon terms** (log games left, games left <= 6 and <= 3, each for players who just missed a game) | **28.0** (+5.4) | **34.3** (+9.5) | 28.0 (-0.3) | 0.484 |
| the same plus the player's own history of past absences (spell count, mean length, share of short spells) | 28.0 (+4.9) | 34.7 (+8.7) | 27.9 (-0.2) | 0.490 |
| the horizon terms for everyone, not just returners | 28.0 (+5.6) | 34.2 (+9.6) | 28.0 (+1.1) | 0.482 |
| the club's prior-season return rate | 28.6 (+7.8) | 35.7 (+13.0) | 28.0 (0.0) | 0.456 |

- The horizon terms help: a paired bootstrap over player-seasons gives +1.1 points on drafted
  returners (95% CI +0.6 to +1.6) and +2.3 on the top 50 (+0.9 to +3.7), with no change
  for everyone else. The reason is that with a long horizon a returner gets most of his
  games and with a short one the season's end (benching, shutdowns) takes them.
- It closes about half of the gap, not all of it: +7.9 becomes +5.4. By season the MAE gain
  is 2024 (30.4 to 28.9) and 2025 (26.4 to 26.0), with 2023 slightly worse (29.3 to 29.5).
- **Player history adds nothing** beyond the horizon terms (+1.0 against +1.1, and worse for
  the top 50). Players with short past spells are not reliably the ones who return sooner.
- **The club's return rate adds nothing**.
- Applying the horizon terms to *everyone* is wrong: it moves the overall bias to +1.1 and
  weeks 1-3 to +5.6, because the same share does not fall with horizon for players who
  just played. They belong to the returner groups only.
- Not shipped. The model in `ros_reconciled.py` is unchanged.

## What is not solved

- **Returners are still under-projected**: +7.9 for drafted returners and +13.2 for the
  top 50, after every change above. Most of it is **games, not rate**: see the next
  section. The missing information is probably severity and timetable news, which the
  feeds do not carry.
- **Weeks 1-3 are slightly worse**, as above.
- **A game cut short** was invisible to the games model (`played` is one flag); it is now
  read as a group of its own, which fixes a 9-point over-projection of games for the
  players it covers. See `docs/partial-game-2026-10.md`.
- **The intervals** for returners are too narrow.
- **A live run on Wednesday** has no game-status designations yet (they are filed Friday), so
  a player who was Out last week and may be back reads as if nothing were known. Re-running
  Friday adds real information.
- **Three holdout seasons**, with no significance tests across folds. The differences in
  the early-week table in particular are inside what three seasons can resolve.

## Reserve-list players and overrides

The model knows a player was on reserve but not that an injured-reserve stint has a minimum
length, so the certain absence is applied afterwards in `scripts/project_live.py`: expected
games and the quantiles are scaled by the share of remaining games left once it is removed.
The rate is untouched, so the identity holds on the scaled row. `projections/overrides/` can
only extend an absence.
