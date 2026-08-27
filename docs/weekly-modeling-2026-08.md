# The weekly layer (2026-08-27)

Two responses, one panel, built independently of the season-average pipeline.
Nothing here reads a season projection and nothing here is constrained by one.

**Model 1 — next week.** Points in week `w`, given every week before it. The
start/sit decision.

**Model 2 — rest of season.** Points from week `w` to the end of the regular
season, given every week before it. At `w = 1` this is the draft question; from
about week 5 it is the waiver question.

Code in `src/ffmodel/weekly/`. Validated by `scripts/validate_weekly.py`,
checked against the draft board by `scripts/compare_weekly_to_adp.py`, blended by
`scripts/blend_weekly_with_market.py`, diagnosed by
`scripts/diagnose_weekly_calibration.py`, run by `scripts/project_week.py`.

## The bar: a naive draft board

Every claim here is measured against ADP, because ADP is what a manager already
has for free. The weekly restatement is the season layer's rank curve prorated:
a per-position log fit of season points on draft rank, divided by games for one
week, multiplied by games remaining for the rest of a season, with spread from
the curve's own residuals at nearby ranks.

It is stale by construction and increasingly so — published in August, it knows
everything in week 1 and nothing new by week 12 — so **the comparison is always
reported by week**, never pooled alone. And it is scored on the **drafted pool**,
because beating a board on players it declined to rank is close to free.

### Where the models beat it, and where they do not

Against the ADP curve on the drafted pool, walk-forward over 2023/2024/2025:

| response | pooled | weeks 1–4 | weeks 5–10 | weeks 11–18 |
|---|---|---|---|---|
| **next week** | wins 7/7 | wins 6/7 | wins 7/7 | wins 7/7 |
| **rest of season (blended)** | wins 7/7 | wins 4/7, ties 1 | wins 7/7 | wins 7/7 |

Next week, pooled drafted pool: MAE −10.8%, CRPS −19.0%, within-position
Spearman +75%, top-24 hit rate +26.6%, and CRPS wins on 3/3 folds individually.
Late in the season the margin widens to CRPS −24.9%.

**The three metrics that do not clear the bar, all in weeks 1–4:**

| response | metric | model | ADP | margin | folds |
|---|---|---:|---:|---:|---|
| next week | top-24 hit rate | 0.278 | 0.329 | −15.3% | lost 3/3 |
| rest of season | within-position ρ | 0.535 | 0.543 | −1.5% | lost 2/2 |
| rest of season | top-24 hit rate | 0.459 | 0.483 | −5.0% | lost 1/2, tied 1 |

The pattern is specific and worth stating precisely: **early in the season the
model orders the whole position better than the board and picks the very top
worse.** Within-position Spearman is a win in weeks 1–4 on 3/3 folds for next
week; the top-24 hit rate is a loss on 3/3. A draft board is a consensus ranking
built by people concentrating on the players who go early, and that is exactly
where it stays sharper than a model reading box scores. Everywhere else, and on
every loss metric everywhere, the models win.

This is the season layer's finding reappearing at a new cadence, and it was
attacked the two ways that package already has evidence for.

**ADP as a feature, not just a rival.** The board carries the one thing usage
history cannot: an offseason. A trade, a rookie, a vacated backfield reaches no
box score until it is too late to help. Entering `adp_log_rank` and
`adp_drafted` interacted with an early-season indicator — so the fit leans on the
board while history is thin and discounts it after — cut the rest-of-season
early-week deficit from −3.05% MAE to −0.51%, and from −4.78% CRPS to −0.69%.

**A blend, weighted per horizon.** The variance-optimal weight on the model
against the curve is the slope of `observed - curve = a + b(model - curve)`.
Estimated separately per horizon, from earlier holdouts only, it produces the
schedule this section argues for without being told to:

| horizon | 2024 weight | 2025 weight |
|---|---:|---:|
| weeks 1–4 | 0.65 | 0.51 |
| weeks 5–10 | 1.00 | 0.93 |
| weeks 11–18 | 1.00 | 1.00 |

The model earns its weight as the season gives it something the board never saw.
Pooled over the two scorable folds, the blend beats the curve on **every metric
overall and in mid and late**, and in weeks 1–4 beats it on MAE (+2.0%), RMSE
(+1.0%), CRPS (+2.1%) and 80% coverage while ties on 95% and trailing on the two
ordering metrics above. It also beats the unblended model on all three folds.
2023 is unscored throughout: no earlier holdout exists to take its weight from.

Mixture, not averaging — averaging paired draws produces a distribution narrower
than either input and wrecks calibration, which the season layer measured at
0.689 against a nominal 0.80.

## Where the error actually is (2026-08-27, follow-up)

`scripts/diagnose_weekly_errors.py` attributes the error to segments and reports
the signed bias, because over- and under-projection want opposite fixes. It
overturns the framing above and finds a different, larger problem.

### The next-week model does not get worse early. The board gets better.

| window | model MAE | model CRPS | ADP MAE | ADP CRPS |
|---|---:|---:|---:|---:|
| weeks 1–4 | 4.967 | 3.376 | 5.590 | 4.061 |
| weeks 5–10 | 4.974 | 3.329 | 5.976 | 4.407 |
| weeks 11–18 | 5.055 | 3.412 | 6.317 | 4.708 |

The model is flat across the season to within a rounding error. **Every bit of
the narrowing early-season gap is the draft board being good in September and
decaying from there** — its CRPS worsens by 16% from week 1 to week 18 while the
model's moves 1%. "The model struggles early" is the wrong description of the
next-week response and the wrong thing to go and fix.

It remains true for rest of season, where weeks 1–4 really are harder in absolute
terms — but most of that is horizon length (a 17-game sum against a 3-game one),
not a defect that appears in September.

### The real failure is role change, and it is not seasonal

Bias is projected minus observed, so positive is over-projection. Both windows,
relevant population plus anyone the board drafted:

| segment | share of error | observed | projected | bias (early) | bias (later) |
|---|---:|---:|---:|---:|---:|
| **role grew (share +10pt)** | 14% | 15.0 | 8.0 | **−6.99** | **−6.87** |
| **role shrank (share −10pt)** | 22% | 2.4 | 6.4 | **+4.00** | **+4.00** |
| did not play | 20–22% | 0.0 | 3.9 | +3.91 | +3.70 |
| returning after 1 week out | 7–9% | 3.8 | 5.7 | +1.88 | +2.80 |
| handcuff: lead back out | 3.5% | 6.7 | 4.8 | −1.95 | −2.19 |

The two role-change rows are the same number in both windows. This is not an
early-season problem that fades; it is a permanent property of a model whose
usage features are all exponentially weighted averages of what a player has
already done. A promotion enters the features only as it is produced.

The board is worse on both (−7.3/−7.7 on role growth, +4.3/+5.2 on role
decline), so this is not something ADP solves — it is what any
backward-looking method does.

### How long it takes to catch up: about two weeks

Following the model's bias forward from the week a player's share of his team's
carries or targets first steps up by ten points and reaches twenty percent:

| weeks since the role changed | n | bias | MAE | observed |
|---|---:|---:|---:|---:|
| 0 (the week it happens) | 591 | **−7.41** | 8.11 | 14.57 |
| 1 | 566 | −1.98 | 5.80 | 10.37 |
| 2 | 543 | −0.49 | 5.48 | 9.17 |
| 3 | 533 | −0.30 | 5.81 | 8.97 |
| 5 | 476 | −0.06 | 5.61 | 8.76 |

**The entire cost is the first week.** Seven and a half points low the week a
role opens, two the week after, unbiased by the second. The four-game half-life
is doing exactly what it was specified to do, and the damage is concentrated
where no amount of retuning the decay will help — in the week before there is
any new data at all.

### Handcuffs specifically

A backup running back in a week his team's established lead rusher (lagged carry
share ≥ 0.35) is inactive: **the model projects him 1.9–2.2 points low**, on 86
early rows and 503 later ones, about 3.5% of total error. Smaller than the
general role-growth miss because many activations do not produce a full takeover
— a committee absorbs the carries. The board is slightly worse (−2.2/−2.3).

The gap between the handcuff miss (−2.1) and the role-growth miss (−7.0) is the
useful part: knowing the starter is out is worth much less than knowing the
backup will actually get the volume, and only the second is a large error.

### The two hypotheses that did not hold

**Rookies and thin history are not the problem.** On players with no career
weeks at all the model is essentially unbiased (+0.10 early) and beats the board
outright (CRPS 2.15 against 3.19) — the board over-projects them by +2.94. Thin
history (1–7 games) is the same story: bias −0.49, CRPS 2.98 against the board's
3.94. Imputing missing history to the training median with an explicit
"no history" indicator, plus the ADP columns, handles these rows better than the
consensus that was built to price them.

Note that the headline `relevant` population cannot see this at all — it requires
four prior appearances and so excludes every rookie by construction. The
diagnostic has a `--population drafted-or-relevant` mode for exactly this reason.

**Low-ADP over-projection is real but small, and only for the undrafted.**
Players the board declined to rank are over-projected by **+0.71 early** against
−0.17 later, which is the "last year's part-timer projected into a role he has
already lost" effect and it is worth about seven tenths of a point on players
averaging 2.6. Within the drafted board there is no rank gradient in the bias at
all: ADP 1–36 −0.26, 37–84 −0.60, 85–150 −0.89, 151+ +0.04.

### What this points at

Ranked by share of error, and none of it is a decay-rate question:

1. **Role change, in the first week only** (36% of error across both directions).
   Wants a leading indicator of role, not a better average of the trailing one:
   snap share and depth charts are both cached and unused, and a depth chart is
   published before the game.
2. **Absence** (20–22%). The model projects 3.9 points for players who score
   zero, and over-projects returners by 1.9–2.8. The availability half reads
   appearance history and nothing else; the weekly injury report is cached and
   unused.
3. **Everything else.** No segment above accounts for more than 3.5%.

## The panel, and why it starts in 2016

A stat feed contains rows for players who recorded something. Modelling on those
rows answers "how many points did he score, given he scored", which is not the
question a lineup poses — the week a starter is inactive is the outcome, and it
is the outcome that costs the most. So the panel is keyed on the **roster**: one
row per rostered skill player per team-week, carrying the stat line if there is
one and an honest zero if there is not.

Bye weeks are dropped rather than zeroed. A team-week with no game produces no
lines for anybody, which is indistinguishable from a roster full of inactives
unless the schedule is consulted.

Weekly rosters exist upstream from 2011 and **are not usable before 2016**:

| seasons | rows/season | play rate | RES play rate |
|---|---:|---:|---:|
| 2011–2015 | ~6,050 | 0.69 | **0.65–0.73** |
| 2016–2025 | ~9,600 | 0.54–0.59 | ~0.00 |

A player on injured reserve who records a stat line 70% of the time is not on
injured reserve. The panel is therefore **2016–2025, 98,225 rows**, 56.7% of them
weeks a player actually appeared. Week-`w` roster status is recorded for
diagnostics and is never a feature: whether a player was declared inactive is the
thing being predicted.

Every number below is reported on the **relevant** population — recency-weighted
average over weeks played of at least 4 points, and at least 4 prior appearances,
both read from lagged columns only. That is 49.1% of the panel. The season layer
learned the cost of not doing this: scoring on a fringe-heavy mixture flatters a
model that is good at forecasting zero.

## Leakage

On a seventeen-game series an expanding mean that forgets to shift includes the
week being predicted. The resulting model validates beautifully, cannot be used,
and nothing in the metrics says so. The lag is therefore structural: every
history column is produced by one function that applies its statistic and then
shifts it by one row within the group.

`tests/test_weekly_features.py` pins it the only way that demonstrates it —
perturb an outcome by 1000 points, assert no feature at or before that week
moves, **and assert at least one feature after it does**. A feature layer that
ignored history entirely would pass the first half alone. A second test pins the
features invariant to row order, which a grouped expanding statistic does not
give for free. A third does the same for the defence-allowed columns.

The market lines are the one exception and are deliberately unlagged: a closing
spread is published before kickoff and is legitimately known at decision time.

## Model 1: next week

Relevant population, pooled over three holdouts, n = 15,320, 800 draws.

| rung | MAE | RMSE | CRPS | cov80 | ρ (within pos) |
|---|---:|---:|---:|---:|---:|
| ADP curve | 6.046 | 7.703 | 4.468 | 0.458 | 0.405 |
| position climatology | 6.564 | 8.851 | 4.855 | 0.811 | 0.003 |
| career mean | 5.801 | 7.552 | 3.944 | 0.779 | 0.409 |
| recency-weighted mean | 5.173 | 6.905 | 3.493 | 0.792 | 0.570 |
| hurdle | 5.091 | 6.872 | 3.409 | 0.888 | 0.582 |
| + team & points-allowed matchup | 5.072 | 6.863 | 3.403 | 0.888 | 0.583 |
| + game script & phase defence | 5.049 | 6.834 | 3.391 | 0.887 | 0.587 |
| + per-position fitting | 5.031 | 6.829 | 3.385 | 0.885 | 0.587 |
| **+ ADP (ships)** | **5.010** | **6.815** | **3.377** | 0.884 | 0.588 |

Each step against the one above it, on CRPS:

| step | ΔCRPS |
|---|---:|
| climatology → career mean | −18.75% |
| career mean → **recency** | **−11.45%** |
| recency → **hurdle** | **−2.41%** |
| hurdle → team + points-allowed matchup | −0.16% |
| → **game script & phase defence** | **−0.36%** |
| → **per-position fitting** | −0.18% |
| → ADP | −0.24% |

**Recency is still the result.** A four-game half-life on a player's own history
is worth 11.5% of CRPS, more than every structural idea after it combined. The
decay was fixed a priori at four games rather than tuned, so no holdout was spent
on it.

### Game script and opponent: real, and small

The first attempt at matchup — points allowed by the opponent to this position,
recency-weighted — was worth 0.16% and exactly zero on one fold. That null did
not survive asking the question properly, but what replaced it is modest.

Three changes, together worth **−1.60% MAE and −0.93% CRPS** over the plain
hurdle:

**Phase-split defence, volume and efficiency apart.** "Good against the run" is
two claims. A defence can hold rushing yards down because it is hard to run on or
because nobody runs on it, and those point in opposite directions for a back's
workload. So carries and targets conceded are separate columns from yards per
carry and EPA per play conceded, and run is kept apart from pass — a defence is
routinely good at one and poor at the other, which a points-allowed aggregate
averages away.

**Game script from the closing line.** The spread says who is expected to lead,
and the implied totals — `total/2 ± spread/2` — say how much this offence is
expected to score and how much it will have to answer. A team favoured by ten
runs out the fourth quarter; a team down two scores throws. The sign convention
is load-bearing and tested: `spread_line` is quoted from the home team's
perspective and is re-signed per team, so positive always means *this* team is
favoured. Backwards, every game-script coefficient inverts and nothing about the
fit looks wrong.

**Per-position fitting, which is what lets the above be seen at all.** Several
script terms point in opposite directions by position: a favourite's running back
gets the fourth quarter and a favourite's receivers do not. A pooled slope
averages them toward zero and reports a real effect as a null. Fitting each
position its own design avoids the collinear-interaction pathology that sank the
season layer's ADP interactions — four separate designs, not one carrying a level
plus three deviations under a shared prior. A test asserts the RB and WR spread
slopes come out with opposite signs and the pooled slope between them.

So the mechanisms are real and they are measurable only in the right encoding.
They are also worth about a seventh of what recency is worth, and that proportion
is the honest headline. Own-defence quality (the "modern Bengals" effect — a team
that cannot get off the field throws to keep up) is in the script block and is not
separately identified from the implied opponent total, which measures the same
thing prospectively.

### The hurdle's coverage is the atom, not a defect

The hurdle reports 80% coverage of 0.888 against nominal 0.80. Central-interval
coverage is not a proper scoring rule and behaves badly on a distribution with a
point mass: if a player has a 12% chance of not playing, the 10th percentile *is*
zero and every positive outcome clears it from below.

`scripts/diagnose_weekly_calibration.py` separates the halves. Availability gets
a reliability table; magnitude is scored on the weeks he played using the
conditional predictive **without the atom** — scoring the unconditional
predictive against outcomes selected on having played would report a
correctly-sized zero mass as a downward bias, which is the mistake the diagnostic
was rewritten to avoid.

| holdout | availability gap (worst bucket) | magnitude cov80 | cov95 | PIT |
|---|---:|---:|---:|---|
| 2023 | −0.050 | 0.813 | 0.951 | flat |
| 2024 | −0.033 | 0.813 | 0.951 | flat |
| 2025 | +0.024 | 0.803 | 0.949 | flat |

Both halves are calibrated. The pooled 0.888 is the atom. The one standing
blemish is the largest availability bucket, which under-predicts the play rate by
about 2 points in all three folds — consistent in sign, small, recorded.

## Model 2: rest of season

Same folds and population. The games-remaining offset is the player's club's,
taken from the schedule at week `w`, so it cannot encode that he was about to be
cut. A player who leaves the league scores zero for the weeks he is gone and
those zeros are summed.

| estimator | MAE | CRPS | cov80 | cov95 | ρ | top-24 |
|---|---:|---:|---:|---:|---:|---:|
| ADP curve | 34.89 | 24.61 | 0.663 | 0.873 | 0.674 | 0.380 |
| direct total | 29.59 | 20.90 | 0.802 | 0.946 | 0.766 | 0.392 |
| direct total + phase | 29.60 | 20.89 | 0.802 | 0.945 | 0.766 | 0.396 |
| **+ ADP (ships, before blending)** | **29.17** | **20.56** | 0.789 | 0.945 | **0.770** | **0.433** |
| independent weeks | 29.51 | 22.48 | 0.564 | 0.733 | 0.764 | 0.365 |
| hierarchical | 29.51 | 22.18 | 0.589 | 0.763 | 0.764 | 0.365 |

### The argument for the hierarchy was right, and lost anyway

Simulating the remaining games and adding them up understates the total's
variance if the weeks are drawn independently: most of what is unknown about a
player's rest of season is unknown in all his weeks at once and does not average
out. So the hierarchical arm draws the player first — a latent availability rate
from a Beta whose concentration comes from how much realised play counts
over-disperse relative to Binomial, and a latent per-game level whose spread is
estimated by variance components from the covariance between two different weeks
of the same player — and then plays the games against that fixed player.

**That reasoning is confirmed**: the hierarchy beats independent weeks by 1.31%
CRPS on 3/3 folds and moves coverage the right way at both levels. **And it is
nowhere near enough**: 0.589 against nominal 0.80 is still severely
over-confident, and a plain ridge on the total beats it by 5.80% CRPS on 3/3
folds while covering 0.802.

The direct fit is trained on realised season totals, so it learns their spread
from data. The simulator assembles that spread from parts, and every part it gets
slightly wrong compounds across seventeen games — including the one it has no term
for: that a role can simply end. Its bias is +2.97 against the direct fit's −1.36.

This is the third time this package has run that comparison and got the same
answer, after the rank curve and the composition test. **Fit the thing you are
going to be scored on.**

Game script is deliberately absent from this response even though it helps the
weekly one. A spread is published for one game; this response spans up to
seventeen, and this week's line says nothing about week twelve's.

## What ships

`scripts/project_week.py`, fitted on seasons strictly before the one requested.

- **Next week** — the hurdle with team, matchup, phase defence, game script and
  ADP, fitted per position. Output carries `p_plays` alongside quantiles: a p10
  of zero means "he might not play", a different call from a low projection for
  someone certain to suit up.
- **Rest of season** — the direct total with phase and ADP, blended with the rank
  curve at a per-horizon weight. The weight is estimated inside the training
  window by holding out its most recent season, since a live projection has no
  later season to borrow from; on 2025 that gives 0.41 early, 0.80 mid, 1.00
  late, consistent with the walk-forward weights. `--no-blend` disables it.

Shipping the better-motivated model over the better-calibrated one would be
choosing the story over the evidence, which is why the simulator does not ship.

## What is not established

- **Three folds, two of them scorable for the blend.** 2023 has no earlier
  holdout to take a weight from.
- **The early-season ordering gap is real and unfixed.** Both remedies narrowed
  it and neither closed it. A model that reads the board still picks the top 24
  worse than the board does in weeks 1–4.
- **Snap counts are not in the panel.** They exist upstream from 2014 but join on
  name and PFR id rather than the gsis id everything else uses. Snap share is the
  most direct measure of role available and its absence is the largest known gap.
- **No injury designations.** The weekly injury report is cached (2009–2025) and
  unused. It is the obvious next input for the availability half, whose largest
  bucket carries the standing 2-point bias.
- **Game script is measured through the closing line only.** No pace, no
  personnel-grouping tendency, no coverage-specific matchup. The line prices what
  the market expects, which is not the same as what the offence will do.
- **The panel is PPR only.** The scoring rules are parameterised and the panel
  builder takes a format, but nothing has been validated in standard or half-PPR.
