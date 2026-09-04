# Kickers, defenses, and the weather (2026-09-04)

Three questions, answered in the order they could be answered cheaply.

1. **Does the weather help a weekly projection?** No, for skill positions —
   decisively, and the test was built to be generous. Yes, marginally, for
   kickers, and the gain is in the temperature and the wind rather than the roof,
   which means it needs a forecast behind it before it can ship.
2. **Are opposing-defense metrics in the weekly model?** They already were. Eight
   columns of them, plus the own-defense block inside game script. Nothing was
   missing; the section below records what is there so the question does not get
   asked a third time.
3. **Can kickers and team defenses be projected on the same footing as the skill
   positions?** Yes. Both panels are built, both beat their baselines on the same
   walk-forward, and the finding on each is not the one folklore predicts.

Code in `src/ffmodel/weekly/weather.py` and `src/ffmodel/weekly/specialists.py`.
Validated by `scripts/validate_weekly_weather.py` and
`scripts/validate_specialists.py`. Every number below is walk-forward over
2023/2024/2025, fitted on seasons strictly before each holdout, scored on the
`relevant` population unless stated.

## The opponent question, settled

The weekly model has read the opposing defense since the layer was built. There
are three separate blocks of it, and the reason there are three is that the first
one was nearly a null on its own.

| block | n | columns |
|---|---:|---|
| opponent, coarse | 1 | points allowed to this position, recency-weighted |
| opponent, phase-split | 7 | carries, yards, YPC and EPA allowed rushing; targets, yards and EPA allowed receiving |
| game script | 7 | spread, total, implied team and opponent totals, **own** defence EPA and yards allowed |

The coarse column — points allowed to the position — was worth 0.16% and exactly
zero on one fold. What rescued it was splitting the claim in two. "Good against
the run" can mean a defense is hard to run on or that nobody bothers running on
it, and those point in opposite directions for a back's workload, so volume
conceded and efficiency conceded are separate columns and run is kept apart from
pass. Together with game script and per-position fitting that is worth −1.60% MAE
and −0.93% CRPS over the plain hurdle — real, and about a seventh of what recency
alone is worth.

Own-defense quality is in the script block rather than a block of its own, and it
is not separately identified from the implied opponent total: a team that cannot
get off the field throws to keep up, and the closing line already prices that
prospectively.

## Weather: a generous test, and a null

`roof`, `temp` and `wind` sit on every nflverse schedule row and nothing in this
package had ever read them. They are not the same kind of column, and the whole
design of the test turns on the difference.

`roof` is known when the schedule is published, months before kickoff. It is
fully populated — every regular-season game 2016–2025 carries one of `outdoors`,
`dome`, `closed` or `open` — and a live projection may read it as freely as it
reads the opponent.

`temp` and `wind` are recorded **at** the game. They are what the conditions
turned out to be, not what anyone knew on Sunday morning. Fitting on them and
scoring on a holdout therefore measures a **ceiling**: what perfect foreknowledge
of the weather would have been worth. That is deliberately generous, and it is
the right test to run first — because if perfect knowledge does not pay, a
forecast of it certainly does not, and the question closes for the price of one
ablation.

### The encoding

Indoors the conditions are not missing, they are **controlled**. A closed roof is
70 degrees and still by construction, so filling it that way is a statement of
fact rather than an imputation, and the `roof_indoor` indicator alongside lets the
fit separate a dome from a calm 70-degree afternoon in September — the same
numbers, not the same game. Outdoor games with no reading (9.1% overall, and
48.7% of 2022, where the feed simply stopped recording) stay NaN for the design's
median fill, with `wx_missing` marking them so those rows get their own level.

Wind and cold are not believed to act linearly — the claim is that wind matters
once it can move a ball in flight and cold once it is at freezing — so
`wx_wind_high` (15 mph) and `wx_freezing` (32 °F) enter alongside the continuous
columns. Both thresholds are fixed a priori rather than tuned, so no holdout is
spent selecting them.

The join covers 100% of the 98,225-row panel: 29.1% indoor, 6.5% missing a
reading.

### The result on skill positions

Control is the shipped next-week model. Each rung is the control plus its own
columns and nothing else.

| rung | MAE | CRPS | Δ CRPS |
|---|---:|---:|---:|
| `hurdle+everything+ngs/position` | 4.7011 | 3.1822 | — |
| `+roof` | 4.7019 | 3.1814 | **−0.027%** |
| `+weather` | 4.7056 | 3.1823 | **+0.002%** |

Both are ties against the package's 0.25% materiality floor, by more than an
order of magnitude. And the sign does not hold across folds:

| holdout | control | +roof | +weather |
|---|---:|---:|---:|
| 2023 | 3.1667 | 3.1655 | 3.1677 |
| 2024 | 3.2943 | 3.2954 | 3.2987 |
| 2025 | 3.0877 | 3.0851 | 3.0825 |

Nor across the season: `+weather` is +0.118% in weeks 1–4, +0.054% in 5–10 and
−0.090% in 11–18. If weather mattered to skill-position scoring, the late-season
window — the cold and windy one — is where it would show, and −0.09% is not it.

**This is the strong form of the null.** The rung had the actual conditions, not a
forecast of them, and still found nothing. There is no version of a weather join
for QB/RB/WR/TE that does better than this one, so the question is closed rather
than deferred.

The mechanism is presumably real and simply too small to reach a fantasy line
through a projection that already reads the closing total — a game expected to be
a slog is priced as a low total before anyone consults a forecast.

## Kickers

`PANEL_POSITIONS` was `("QB", "RB", "WR", "TE")`. Kickers get their own panel
because the skill panel is built out of opportunity share — targets, carries,
snaps — and none of those words mean anything for a kicker. What they share is
the interface: same row schema, same walk-forward, same estimator protocol, so a
start/sit or draft agent can concatenate all six positions into one table of
comparable points.

Scoring is ESPN-style and configurable (`config.KickerRules`): 3/4/5 by distance
tier, 1 for an extra point, −1 for a miss. nflverse publishes makes and misses
already bucketed by distance, so the tiers are read directly rather than inferred
from a total and an average. The implementation reproduces 2024's actual kicker
leaderboard — Boswell 188, Aubrey 185, Dicker 173.

Panel: 1,825 kicker-weeks over 2016–2025 after the contract-status filter the
skill panel uses, one kicker per team-week in 87% of cases, 90% played.

| rung | MAE | CRPS | within-position ρ | vs recency |
|---|---:|---:|---:|---:|
| climatology | 4.2317 | 2.9509 | −0.011 | +6.02% |
| recency-mean | 3.9929 | 2.7834 | 0.225 | — |
| `kicker-history` | 3.9189 | 2.7360 | 0.247 | −1.71% |
| `kicker+market` | 3.8878 | 2.7242 | 0.272 | −2.13% |
| `kicker+market+roof` | 3.8876 | 2.7213 | 0.278 | −2.23% |
| `kicker+market+weather` | 3.8712 | 2.7118 | 0.292 | **−2.57%** |

Two things worth stating.

**The ladder is shallow, and that is the finding.** Everything after the recency
mean is worth 2.6% of CRPS put together, against the 5.7% the recency mean itself
buys over climatology. A kicker's week is an opportunity count times a conversion
rate; the count belongs to his offence and the rate is close to unpredictable at a
one-week horizon. The model leans on volume and the implied team total and treats
the leg as a small correction, and the numbers say that is the right weighting.

**Weather is material here, and it is the readings rather than the roof.**
Against the `kicker+market` control, `+roof` is −0.107% — a tie — and `+weather`
is −0.454%, which clears the 0.25% floor. Splitting the two was the point of
running both: the gain is in `wx_temp`/`wx_wind`, not in the fully-known roof
column, so **the shippable version needs a forecast**, not a schedule lookup.

It is also not consistent: `+weather` wins 2 folds of 3 (2024 −0.45%, 2025
−0.95%, 2023 +0.11%). So the honest status is *promising and not promoted*. The
next step is a real one rather than a retune — join `ffmodel.data.weather`'s
Open-Meteo previous-run archive at a stated lead time and re-run this exact
ladder. That measures the forecast rather than the outcome, and the gap between
the two numbers is the cost of not being able to see the future.

## Team defenses

One row per team-week; the "player" is the club. Scoring is ESPN-style and
configurable (`config.DefenseRules`): sacks 1, takeaways 2, touchdowns 6,
safeties and blocks 2, plus the points-allowed step function. It reproduces
2024's actual DST leaderboard — Denver 167, Minnesota 155.

| rung | MAE | CRPS | within-position ρ | vs recency |
|---|---:|---:|---:|---:|
| climatology | 4.3137 | 3.0452 | 0.007 | +0.27% |
| recency-mean | 4.3114 | 3.0370 | 0.102 | — |
| `defense-history` | 4.3213 | 3.0389 | 0.086 | **+0.06%** |
| `defense+opponent` | 4.2644 | 2.9796 | 0.206 | −1.89% |
| `defense+opponent+market` | 4.1407 | 2.9042 | 0.307 | **−4.37%** |
| `+weather` | 4.1434 | 2.9034 | 0.307 | −4.40% |

**The headline is the third row.** A defense's own box-score history — its recent
sacks, interceptions, fumble recoveries, points and yards allowed — is worth
*nothing at all* over its own recent fantasy points. It is fractionally worse, and
it lowers within-position ordering from 0.102 to 0.086. Six columns describing
what this defense has been doing add no information about what it will do next
week.

What does work is the other team. Adding the opponent's recent scoring is −1.89%;
adding the closing line on top is −4.37% and nearly triples the ordering
correlation, 0.102 → 0.307. A DST projection is mostly a projection of the
opponent, and the reason is structural rather than empirical: the points-allowed
half of the response is a step function of the other side's final score, and it
dominates the variance of the whole. The event half is a modest, noisy count.

Weather is a null here (−0.03%, 1 fold of 3), which is what the kicker result
predicts — nothing is being kicked.

## Rest of season, for both

The response and the machinery already existed (`weekly/restofseason.py`, Model
2). Running it over the two new panels gives the draft and waiver question for
kickers and defenses without a new construction, and the result is a clean
negative worth recording.

Kickers:

| rung | MAE | CRPS | cov80 | cov95 | PIT dev | bias |
|---|---:|---:|---:|---:|---:|---:|
| **`direct-total`** | 18.97 | **13.48** | **0.780** | **0.946** | **0.200** | **−0.18** |
| `season-hierarchical` | **18.42** | 13.97 | 0.607 | 0.792 | 0.395 | +3.37 |
| `season-independent` | 18.42 | 14.09 | 0.591 | 0.777 | 0.428 | +3.37 |

Defenses:

| rung | MAE | CRPS | cov80 | cov95 | PIT dev | bias |
|---|---:|---:|---:|---:|---:|---:|
| **`direct-total`** | **15.29** | **10.87** | **0.767** | **0.922** | **0.219** | +2.47 |
| `season-hierarchical` | 17.17 | 12.36 | 0.663 | 0.854 | 0.311 | +4.28 |
| `season-independent` | 17.21 | 12.41 | 0.658 | 0.853 | 0.313 | +4.27 |

**The direct regression wins, and the simulator over-projects.** The latent-player
hierarchy does what it is supposed to do relative to independent weeks — coverage
0.591 → 0.607 for kickers, 0.658 → 0.663 for defenses, PIT deviation down in both
— but the gap it has to close is far larger than the gap it closes, and the
control is better on every distributional metric. This is the same ordering the
skill panel found at the same stage of construction, where `direct total +
everything` also beat the aggregated simulator and only the *recursive* version
with a fitted drift term got close. The recursion is not implemented for these
two panels, and this table is the argument for whether it is worth it.

Two causes of the positive bias, and they are different for the two panels.

**Kickers: panel exit.** The offset the model is given is the *club's* games
remaining — deliberately, since using the player's own remaining rows would leak
the fact that he was about to be cut. But the target sums only the rows he
actually has. Measured directly: mean offset 8.64 games against 8.17 rows
actually remaining, a gap of 0.48 games on 8.2% of rows. At ~7.9 points a game
that is 3.8 points, which is the +3.37 bias almost exactly. The direct regression
does not suffer from it because it fits the deflated total directly and learns the
discount; the simulator multiplies a per-game level by an offset that assumes he
is present throughout. The fix is not a different offset — it is an availability
model whose denominator is team-games rather than player-rows.

**Defenses: mean reversion in a selected population.** The gap there is exactly
zero — a club never leaves the panel — so the +2.47/+4.28 is something else. The
`relevant` population is selected on recent scoring (≥4 points recency-weighted),
and the row above shows DST recent scoring barely persists. Selecting on a
quantity that does not persist and then projecting it forward over ten games
over-projects by construction. Era drift is not the explanation and was checked:
defense scoring moves 6.37 → 6.27 between training and holdout seasons, worth
about 0.9 points over a rest-of-season horizon, and kicker scoring moves the
*wrong way* for this story (7.50 → 7.98).

One implementation note that mattered. The first version floored every simulated
game at zero. That is wrong for these two positions — a kicker who misses a field
goal and an extra point scores −2, a defense conceding 35 with no takeaways
scores −4 — and since the residual pool is empirical, `level + noise` already
reproduces that tail. Clipping it deleted real mass from the low side of every
game and pushed a ten-game sum several points high. Removing the floor cut the
defense bias 5.60 → 4.28 and improved coverage 0.647 → 0.663.

## Status

| piece | state |
|---|---|
| weather on skill positions | **closed as a null**, at the ceiling, not deferred |
| `roof` anywhere | tie; free to keep, earns nothing |
| weather on kickers | material (−0.45% CRPS) but 2/3 folds and needs a forecast join — **not promoted** |
| kicker next-week | built and beats persistence by 2.6% CRPS |
| defense next-week | built and beats persistence by 4.4% CRPS |
| K/DST rest-of-season | `direct-total` promoted; simulator arms lose and are diagnosed |
| opposing-defense metrics | were already in; documented above |

## What is worth doing next

1. **Join the Open-Meteo forecast archive** and re-run the kicker ladder. This is
   the only open weather question and it is now a narrow one.
2. **An availability denominator of team-games** for the specialist
   rest-of-season simulator, which is the identified cause of the kicker bias.
3. **Strength of remaining schedule for defenses.** The one-week opponent column
   is the largest single lever on the weekly response, and its rest-of-season
   analogue — the quality of the offences a defense still has to face — is not
   built. The repo has an `sos/` directory that predates this work.
4. **Recursive simulation for the specialist panels**, if and only if the
   rest-of-season response turns out to matter more than `direct-total` already
   delivers.
