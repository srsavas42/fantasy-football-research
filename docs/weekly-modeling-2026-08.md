# The weekly layer (2026-08-27)

Two responses, one panel, built independently of the season-average pipeline.
Nothing here reads a season projection and nothing here is constrained by one.

**Model 1 — next week.** Points in week `w`, given every week before it. The
start/sit decision.

**Model 2 — rest of season.** Points from week `w` to the end of the regular
season, given every week before it. At `w = 1` this is the draft question asked
without a draft board; from about week 5 it is the waiver question.

Code in `src/ffmodel/weekly/`, validated by `scripts/validate_weekly.py`,
diagnosed by `scripts/diagnose_weekly_calibration.py`, run by
`scripts/project_week.py`.

The headline is two clean negatives and one clean positive, and the negatives are
the more useful half:

- **Recency is almost the whole story.** Exponentially weighting a player's own
  history beats averaging it by 11.4% CRPS. Everything after that is small.
- **Matchup contributes nothing.** The most widely believed weekly input in the
  sport moves CRPS by 0.10% and is exactly zero on one of three folds.
- **The forward simulation lost to a four-term regression.** The hierarchical
  rest-of-season model is better motivated, directionally correct, and worse:
  5.8% worse CRPS on 3/3 folds, and badly over-confident where the direct fit is
  calibrated.

## The panel, and why it starts in 2016

A stat feed contains rows for players who recorded something. Modelling on those
rows answers "how many points did he score, given he scored", which is not the
question a lineup poses — the week a starter is inactive is the outcome, and it
is the outcome that costs the most. So the panel is keyed on the **roster**: one
row per rostered skill player per team-week, carrying the stat line if there is
one and an honest zero if there is not.

Bye weeks are dropped rather than zeroed. A team-week with no game produces no
lines for anybody, which is indistinguishable from a roster full of inactives
unless the schedule is consulted. Leaving byes in would hand the model a large
block of trivially predictable zeros and corrupt both the play rate and the
rest-of-season sum.

Weekly rosters exist upstream from 2011, and **are not usable before 2016**:

| seasons | rows/season | play rate | RES play rate |
|---|---:|---:|---:|
| 2011–2015 | ~6,050 | 0.69 | **0.65–0.73** |
| 2016–2025 | ~9,600 | 0.54–0.59 | ~0.00 |

A player on injured reserve who records a stat line 70% of the time is not on
injured reserve. Whatever those rows are, they are not "employed and did not
play", so the zeros would not be honest and the response would change meaning
halfway through the window. The panel is therefore **2016–2025, 98,225 rows**,
56.7% of them weeks a player actually appeared.

The ACT/INA split does move inside that window — inactives are only reported
separately from 2019, and before that are carried inside ACT — but that is a
relabelling of rows the panel already contains, not a change to which rows it
contains, and no feature reads the label. Week-`w` status is recorded for
diagnostics and is never a feature: whether a player was declared inactive is
the thing being predicted.

### Two populations, both reported

The season layer learned this expensively: fitting and scoring on every rostered
player, most of whom are fringe, cost 3.6 CRPS points on the players who actually
get drafted and accounted for most of an apparent deficit against the draft
board. A pooled weekly metric has the same defect and worse — roughly 43% of the
panel is a player who did not play, most of them third-stringers, and a forecast
of zero for all of them scores well.

So every number below is reported on the **relevant** population: rows where the
player's recency-weighted average *over weeks he played* is at least 4 points and
he has at least 4 prior appearances. That filter reads only lagged columns, so it
defines a population without looking at the outcome, and it keeps an injured
starter in the population he belongs to. It is 49.1% of the panel, about 5,100
rows a season.

## Leakage

On a seventeen-game series an expanding mean that forgets to shift includes the
week being predicted, which is a large share of the average. The resulting model
validates beautifully, cannot be used, and nothing in the metrics says so.

The lag is therefore structural rather than per-feature: every history column is
produced by one function that applies its statistic and then shifts it by one row
within the group, and there is no path through the feature layer that reaches the
current week. `tests/test_weekly_features.py` pins it the only way that actually
demonstrates it — perturb an outcome by 1000 points, assert that no feature at or
before that week moves, **and assert that at least one feature after it does**. A
feature layer that ignored history entirely would pass the first half alone.

A second test pins something subtler. A grouped expanding statistic comes back
ordered by group, not by frame; realigning it positionally happens to work while
the input is sorted by that group and misaligns silently when it is not, handing
every player somebody else's history with no error and entirely plausible
numbers. The features are asserted invariant to row order.

## Model 1: next week

Walk-forward, holdouts 2023/2024/2025, each fitted on seasons strictly before it.
Relevant population, pooled, n = 15,320. 800 draws.

| rung | MAE | RMSE | CRPS | cov80 | within-position ρ |
|---|---:|---:|---:|---:|---:|
| 1. position climatology | 6.564 | 8.851 | 4.855 | 0.811 | 0.003 |
| 2. career mean | 5.801 | 7.552 | 3.944 | 0.779 | 0.409 |
| 3. recency-weighted mean | 5.173 | 6.905 | 3.493 | 0.792 | 0.570 |
| 4. hurdle | 5.091 | 6.872 | 3.409 | 0.888 | 0.582 |
| 5. + team context | 5.087 | 6.866 | 3.406 | 0.889 | 0.582 |
| 6. + matchup | 5.072 | 6.863 | 3.403 | 0.888 | 0.583 |

Each rung against the one above it, on CRPS:

| step | ΔCRPS | folds improved |
|---|---:|---|
| climatology → career mean | −18.75% | 3/3 |
| career mean → **recency** | **−11.45%** | 3/3 |
| recency → **hurdle** | **−2.41%** | 3/3 |
| hurdle → team context | −0.06% | — |
| team → matchup | −0.10% | 2/3, one exactly 0.000 |

**Recency is the result.** A four-game half-life on a player's own history is
worth 11.5% of CRPS over averaging that history flat, and it is worth more than
every structural idea that follows it combined. This is the same instinct as the
season layer's `HISTORY_ALPHA = 0.50`, at a cadence where there is enough series
to make it pay. The decay was fixed a priori rather than tuned, so none of the
holdout was spent on it.

**The hurdle is real but modest.** Splitting availability from magnitude is worth
2.4% CRPS, 3/3 folds. It earns its place — and see the calibration section, which
is where it earns it properly rather than on the pooled number.

**Team context and matchup are null.** Adding the offence a player is attached to
moves CRPS by 0.06%. Adding a recency-weighted average of the points his opponent
has allowed to his position moves it by a further 0.10%, and by exactly zero on
2024. Both are well inside noise.

The matchup null deserves stating plainly because of what it contradicts. Nearly
every start/sit column in the sport is built on defence-versus-position, and here
it contributes nothing to a model that already knows the player's own usage. The
explanation is the season layer's, in a new setting: *whatever the matchup knows
about a player's week, his usage history already knows*. A target share of 27%
is a fact about a role, and it is the role that carries the week.

Two things this does **not** establish. It is one encoding of matchup — a
recency-weighted points-allowed average by defence and position. A different one
(pace, coverage-specific splits, defence-adjusted efficiency rather than raw
points) is a different hypothesis and is untested. And the null is measured with
the full feature set present, which is the right room to measure it in; the same
term in a feature-poor design would very likely look useful, which is exactly how
the season layer's ADP interaction produced a +4.11% probe result and a −1.57%
model result.

### The hurdle's coverage is the atom, not a defect

Rung 4 improves CRPS and simultaneously reports 80% coverage of 0.888 against a
nominal 0.80. Read naively that is a model that got better and worse at once, and
the temptation is to widen or narrow something until the number looks right.

Central-interval coverage is not a proper scoring rule and behaves badly on a
distribution with a point mass. If a player has a 12% chance of not playing, 12%
of the predictive sits at exactly zero, so the 10th percentile **is** zero and
every positive outcome clears it from below. The interval is not too wide; the
summary is wrong for the shape.

`scripts/diagnose_weekly_calibration.py` separates the halves rather than
asserting this. Availability gets a reliability table; magnitude is scored on the
weeks he played using the conditional predictive **without the atom** — scoring
the unconditional predictive against outcomes selected on having played would
report a correctly-sized zero mass as a downward bias, which is the mistake this
diagnostic was rewritten to avoid.

Availability, predicted against observed play rate:

| bucket | 2023 gap | 2024 gap | 2025 gap |
|---|---:|---:|---:|
| 0.00–0.30 | −0.050 | −0.033 | +0.009 |
| 0.30–0.50 | −0.004 | −0.011 | +0.000 |
| 0.50–0.70 | −0.014 | +0.012 | +0.024 |
| 0.70–0.85 | +0.002 | −0.011 | +0.019 |
| 0.85–0.95 | −0.018 | −0.025 | −0.026 |

Magnitude, on the weeks he played:

| holdout | cov80 | cov95 | bias | PIT shape |
|---|---:|---:|---:|---|
| 2023 | 0.813 | 0.951 | +0.27 | flat |
| 2024 | 0.813 | 0.951 | −0.22 | flat |
| 2025 | 0.803 | 0.949 | +0.34 | flat |

Both halves are calibrated. The zero mass is the right size and the magnitude
interval sits on nominal at both levels with a flat PIT, so the pooled 0.888 is
the atom widening a central interval and nothing needs adjusting. The one
standing blemish is the largest availability bucket, which under-predicts the
play rate by about 2 points in all three folds — consistent in sign, small, and
recorded rather than rounded away.

## Model 2: rest of season

Same folds, same population. The target is the sum from week `w` to the end of
the season; the number of games remaining is the player's club's, taken from the
schedule at week `w`, so the offset cannot quietly encode the fact that he was
about to be cut. A player who leaves the league scores zero for the weeks he is
gone and those zeros are summed.

| estimator | MAE | CRPS | cov80 | cov95 | PIT dev |
|---|---:|---:|---:|---:|---:|
| **direct total** | 29.59 | **20.90** | **0.802** | **0.946** | **0.141** |
| independent weeks | 29.51 | 22.48 | 0.564 | 0.733 | 0.496 |
| hierarchical | 29.51 | 22.18 | 0.589 | 0.763 | 0.449 |

### The argument for the hierarchy was right, and lost anyway

The obvious construction simulates the remaining games from the weekly model and
adds them up. Drawing those weeks independently makes the total's variance `G`
times a single week's, which is the variance of a player whose true ability is
*known* and whose weeks differ only by luck. That is not this problem: most of
what is unknown about a player's rest of season is unknown in all his weeks at
once, and does not average out.

So the hierarchical arm draws the player first — a latent availability rate from
a Beta whose concentration comes from how much realised play counts over-disperse
relative to Binomial, and a latent per-game level whose spread is estimated by
variance components from the covariance between two different weeks of the same
player — and then plays the games against that fixed player.

**That reasoning is confirmed.** The hierarchy beats independent weeks on CRPS by
1.31% on 3/3 folds, and moves coverage the right way at both levels (0.564 →
0.589, 0.733 → 0.763). Drawing the player once does put the correlation back.

**And it is nowhere near enough.** 0.589 against a nominal 0.80 is still severely
over-confident, and a plain ridge on the total — the same lagged features, a
games-remaining offset and one interaction, with residuals resampled locally —
beats it by **5.80% CRPS on 3/3 folds** while covering 0.802 and 0.946 against
nominal 0.80 and 0.95.

The reason is not subtle in hindsight. The direct fit is trained on realised
season totals, so it learns the spread of season totals from data. The simulator
assembles that spread from parts, and every part it gets slightly wrong compounds
across seventeen games — including the one it has no term for at all: that a
player's role can simply end. The simulator's bias is +2.97 against the direct
fit's −1.36, which is what a model that assumes today's role persists for the
rest of the year looks like.

This is the third time this package has run that comparison and got the same
answer. A rank curve beat the season pipeline; composing volume × efficiency cost
nothing over projecting points directly; and now a forward simulation loses to a
regression on the quantity of interest. The pattern is worth naming: **fit the
thing you are going to be scored on.**

### It is much harder at the draft than on the waiver wire

Splitting by when the question is asked, on the relevant population:

| horizon | n | direct CRPS | direct cov80 | hierarchical CRPS | hierarchical cov80 |
|---|---:|---:|---:|---:|---:|
| weeks 1–4 (draft) | 3,379 | 35.73 | 0.712 | 39.87 | 0.455 |
| weeks 5–10 (waiver) | 4,954 | 23.99 | 0.798 | 25.71 | 0.535 |
| weeks 11–18 | 6,987 | 11.53 | 0.847 | 11.13 | 0.692 |

The draft horizon is where both models are worst and where the direct fit's
calibration also breaks down — 0.712 against nominal 0.80 is real
over-confidence, and it is the one horizon where this model should be quoted with
that caveat attached. Mid-season, which is the waiver question, it is calibrated
almost exactly.

Late in the season the ordering flips on CRPS: the hierarchical arm is 3.4%
*better* (11.13 against 11.53) while still under-covering. With two or three
games left the compounding that sinks the simulator has little room to operate.
That is a narrow enough window, and a mixed enough result, that it is recorded
rather than acted on.

Ordering, which is what a draft or waiver decision actually consumes, is
respectable: within-position Spearman 0.766 and a top-24 hit rate of 0.392 for
the direct fit pooled across folds.

## What ships

`scripts/project_week.py` runs both, fitted on seasons strictly before the one
requested, so asking it for a past week reproduces what it would have said at the
time.

- **Next week** — the hurdle, without team or matchup terms, since neither paid.
  Output carries `p_plays` alongside the quantiles: a p10 of zero means "he might
  not play", which is a different call from a low projection for a player certain
  to suit up, and the mean alone cannot distinguish them.
- **Rest of season** — the direct total, not the simulation. Shipping the
  better-motivated model over the better-calibrated one would be choosing the
  story over the evidence.

## What is not established

- **Three folds, and the decay constant was never tuned on them.** That is the
  honest position, but the population filter (4 points, 4 games) was chosen a
  priori and not swept; a different threshold defines a different population and
  the numbers would move.
- **Matchup is null in one encoding only.** See above. This is the result most
  likely to be overturned by a better feature, and the one worth attacking next.
- **Snap counts are not in the panel.** They exist upstream from 2014 but join on
  name and PFR id rather than the gsis id everything else uses. Snap share is the
  most direct measure of role available and its absence is the largest known gap
  in the feature layer.
- **No injury designations.** The weekly injury report is cached (2009–2025) and
  unused. It is the obvious next input for the availability half, which is
  currently inferred entirely from appearance history, and its largest bucket is
  the one with the standing 2-point bias.
- **The panel is PPR only.** The scoring rules are parameterised and the panel
  builder takes a format, but nothing has been validated in standard or half-PPR.
