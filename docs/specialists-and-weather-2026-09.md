# Kickers, defenses, and the weather (2026-09-04)

Three questions, answered in the order they could be answered cheaply.

1. **Does the weather help a weekly projection?** Pooled, no. **Conditionally,
   yes, and the pooled number is misleading**: on the 5.6% of rows where the wind
   is above 15 mph it is worth −0.60% CRPS, while costing a little everywhere
   else. Temperature is a genuine null. A gated wind hinge keeps the gain and
   drops the cost. On kickers it is material in its own right (roof −0.32% on
   3/3 folds).
   All of it needs a forecast behind it before it can ship, because the test used
   conditions recorded at the game.
2. **Are opposing-defense metrics in the weekly model?** They already were. Eight
   columns of them, plus the own-defense block inside game script. Nothing was
   missing; the section below records what is there so the question does not get
   asked a third time.
3. **Can kickers and team defenses be projected on the same footing as the skill
   positions?** Yes. Both panels are built, both beat their baselines on the same
   walk-forward, and the finding on each is not the one folklore predicts.
4. **A fourth question arrived from review and changed two answers.** Kicker
   scoring is not stationary — the 2024–25 kickoff rules moved it +0.58 points a
   game — and the weather null turned out to be a pooling artefact. Both are
   below.

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

## Weather: a generous test, and a conditional effect

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

The rung had the actual conditions, not a forecast of them, so no version of this
join does better on a pooled metric.

**But "pooled" is carrying the whole claim, and the section below shows it should
not be.** An earlier draft of this document explained the null by saying the
closing total already prices the weather. That explanation is wrong, and the
correction is worth more than the headline was.

### The null is real and the explanation was wrong (2026-09-04, follow-up)

Weather depresses scoring, exactly as the folklore says. Outdoor regular-season
games 2016–2025, by wind at kickoff:

| wind | n | actual total | closing line | actual − line |
|---|---:|---:|---:|---:|
| 0–5 mph | 391 | 46.48 | 45.24 | **+1.24** |
| 5–10 | 746 | 45.33 | 44.87 | +0.46 |
| 10–15 | 377 | 42.91 | 44.09 | −1.18 |
| 15–20 | 152 | 41.77 | 44.10 | **−2.33** |

Scoring falls 4.7 points from the calmest bucket to a 15–20 mph one. And the
market does **not** fully price it: the closing total moves only 1.1 points
across that range while the actual moves 4.7, leaving a 3.6-point swing in the
residual. So the "the line already knows" explanation is false on its own terms.

The mechanism is the one anyone would guess, and it is in the data:

| wind | team pass att | team rush att | pass share | team fantasy pts |
|---|---:|---:|---:|---:|
| 0–5 | 34.39 | 26.12 | 0.568 | 86.22 |
| 15+ | 32.72 | 27.15 | 0.547 | 78.76 |

Fewer passes, more runs, less scoring. And the shipped model's **residual**
carries it — over-projection relative to the calm baseline, by position:

| position | bias, wind < 10 | bias, wind 15+ | gap |
|---|---:|---:|---:|
| QB | −1.60 | +0.24 | **+1.84** |
| WR | −0.70 | +0.24 | +0.93 |
| TE | −1.17 | −0.41 | +0.76 |
| RB | −1.00 | −0.57 | +0.44 |

The ordering is the passing mechanism: quarterbacks lose most, backs are
insulated because the volume shifts toward them.

So the information is there. **What killed it pooled was exposure.** Only 5.6%
of scored rows sit above 15 mph. Scoring the same two arms by exposure instead
of by week:

| population | n | control CRPS | +weather | Δ |
|---|---:|---:|---:|---:|
| all relevant (pooled) | 13,859 | 3.1851 | 3.1837 | −0.04% |
| outdoor, wind < 10 | 5,827 | 3.1905 | 3.1948 | **+0.13%** |
| outdoor, wind 10–15 | 1,880 | 3.0932 | 3.0895 | −0.12% |
| **outdoor, wind 15+** | 787 | 2.9772 | 2.9593 | **−0.60%** |
| wind 15+, RBs only | 196 | 3.2697 | 3.2356 | **−1.04%** |
| outdoor, freezing | 553 | 3.2555 | 3.2679 | **+0.38%** |

The full block clears the materiality floor by a factor of two where the wind
actually blows, and *costs* something everywhere else — including on freezing
rows, where cold is not a signal at all. Pooling a −0.60% over 6% of rows with a
+0.13% over 42% of them produces the −0.04% that reads as a null.

### The gated version

That diagnosis names its own fix: drop temperature, drop the linear wind slope,
and enter the roof, the threshold, and a **hinge** in the excess above it
(`wx_wind_excess = max(wind − 15, 0)`), so the columns are zero wherever the
effect is.

From the shipped ladder (`reports/weekly_weather.json`, relevant pooled):

| rung | CRPS | Δ | folds improved |
|---|---:|---:|---:|
| control | 3.1822 | — | — |
| +weather (full) | 3.1823 | +0.002% | 1 of 3 |
| +roof only | 3.1814 | −0.027% | 2 of 3 |
| **+wind (gated)** | 3.1820 | −0.007% | 2 of 3 |

And by exposure, from `diagnose` runs that pool rows across folds rather than
scoring each fold separately (so these are not directly comparable to the table
above, and are reported as the conditional effect they are):

| population | n | control | +weather (full) | +wind (gated) |
|---|---:|---:|---:|---:|
| outdoor, wind < 10 | 5,827 | 3.1905 | +0.13% | **−0.02%** |
| outdoor, wind 15+ | 787 | 2.9772 | **−0.60%** | −0.53% |

The gate does what it was designed to do: it keeps most of the high-wind gain
and stops charging for it on the calm majority. Pooled it remains a tie under
the 0.25% rule and **is not promoted**.

**A caution that applies to this whole section.** These effects are 0.1–0.4% on
three folds, and they move between runs at roughly that size — rebuilding the
panel shifted the kicker roof rung from −0.11% to −0.32%. That is the resolution
limit of this design, not a precision it actually has. The claims worth standing
behind are the ones that survive it: the conditional high-wind effect is several
times larger than the pooled one and has a mechanism and a residual signature
behind it; temperature is consistently a non-effect; and no weather rung clears
the promotion bar pooled.

**This matters more for the start/sit use than the pooled number suggests.** A
start/sit decision is made one row at a time, and the rows where this feature
pays are identifiable *before* kickoff from the schedule and a forecast. A
+1.8-point over-projection on a quarterback in a 15 mph wind is a lineup
decision even when it is invisible in a pooled CRPS.

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
| climatology | 4.2242 | 2.9474 | 0.038 | +5.60% |
| recency-mean | 4.0050 | 2.7912 | 0.211 | — |
| `kicker-history` | 3.9242 | 2.7393 | 0.234 | −1.86% |
| `kicker+market` | 3.8947 | 2.7266 | 0.266 | −2.31% |
| `kicker+market+league` | 3.8795 | 2.7308 | 0.274 | −2.16% |
| `kicker+market+roof` | 3.8911 | 2.7178 | 0.285 | −2.63% |
| `kicker+market+weather` | 3.8762 | 2.7157 | 0.289 | **−2.71%** |

Two things worth stating.

**The ladder is shallow, and that is the finding.** Everything after the recency
mean is worth 2.6% of CRPS put together, against the 5.7% the recency mean itself
buys over climatology. A kicker's week is an opportunity count times a conversion
rate; the count belongs to his offence and the rate is close to unpredictable at a
one-week horizon. The model leans on volume and the implied team total and treats
the leg as a small correction, and the numbers say that is the right weighting.

**Weather is material here, and both halves of it clear the floor.** Against the
`kicker+market` control, `+roof` is −0.320% on **3 of 3 folds** and `+weather` is
−0.400% on 2 of 3. Splitting the two was the point of running both, and the
answer is less clean than the skill-position case: on kickers the *roof* is
carrying real signal on its own — which is the shippable half, since a roof is
known when the schedule is published — while the readings add a little more,
less consistently, and would need a forecast.

Neither is promoted. The roof rung is the better candidate of the two (material,
consistent, and free of the forecast problem), and the honest next step is to
re-run the ladder against `ffmodel.data.weather`'s Open-Meteo previous-run
archive at a stated lead time. That measures the forecast rather than the
outcome, and the gap between the two is the cost of not seeing the future.

Note that this rung moved between runs — an earlier build put roof at −0.107%
and weather at −0.454%. See the caution above: 0.1–0.4% on three folds is the
resolution limit of this design.

### The kickoff rules moved kicker scoring, and it cannot be learned

Kicker scoring is **not stationary**, and the break is where the rulebook says it
should be. The 2024 dynamic kickoff and the 2025 touchback spot both moved
average starting field position forward, which means drives reach field-goal
range more often.

| era | points/game | FG att | PAT att | FG% | 50+ att |
|---|---:|---:|---:|---:|---:|
| 2016–2023 | 7.53 | 1.929 | 2.320 | 0.845 | 0.32 |
| 2024–2025 | **8.12** | **2.028** | 2.303 | 0.848 | **0.49** |

**More chances, not better kicking.** Attempts rise 5.1% while accuracy (0.845 →
0.848) and extra points (2.32 → 2.30) do not move — so this is not more
touchdowns and not a better generation of legs, it is more drives stalling in
range. Long attempts nearly double, which is partly the same field-position
story and partly a separate analytics trend that starts around 2022.

The model's error has the matching signature. Bias by season and window,
negative meaning under-projection:

| holdout | weeks 1–4 | weeks 5–10 | weeks 11–18 |
|---|---:|---:|---:|
| 2023 | −1.18 | −0.76 | **−0.12** |
| 2024 | −1.56 | −0.68 | **−0.91** |
| 2025 | **−1.76** | −0.38 | **−1.08** |

The last third of the season is where the recency features should have fully
absorbed the new level, and instead it is where the two post-rule seasons
separate from 2023 most cleanly: −0.12 becomes −0.91 and −1.08. The career-mean
feature stays anchored in the old era and drags the level down all year.

**The obvious fix was built, measured, and does not work.**
`add_league_baseline` attaches the lagged league-wide mean so the fit can see the
era it is in. As a rung it *costs* CRPS (2.7266 → 2.7308) and nearly doubles the
bias it was meant to remove (−0.235 → −0.420). Two reasons, both structural:

- **It lags the discontinuity it exists to track.** Against the realised level
  the baseline is off by +0.36 and +0.31 in 2024 and 2025, against ±0.13 in
  every season before them. A trailing average cannot lead a step change.
- **Its coefficient is extrapolated rather than estimated.** Over 2016–2022 the
  column has a standard deviation of 0.103. The shift it must then price is
  0.271 — 2.6× the variation any coefficient on it was ever fitted from.

The general form of this is worth stating plainly: **the size of a rule change
cannot be learned from data that predates it**, and no lagged feature evades
that. Nor does waiting a season out — 2025 is fitted with a full new-era season
in training and its weeks 1–4 are *worse* than 2024's (−1.76 against −1.56),
because 2025 moved the touchback spot again and is its own new era.

What would actually work is an external prior on the size of the effect, which is
what `data/manual/` exists for in this package (as `coach_team_period.csv`
already does for a different unlearnable fact). The rung is kept as the evidence
for that conclusion, not as a candidate.

Two practical consequences for anything built on these projections. Kicker
projections in 2024–2025 are biased **low** by roughly 0.9–1.1 points a game late
in the season and 1.6–1.8 early, so a draft or waiver agent reading them will
systematically under-rate the position. And any future rule change will do this
again, silently, in whichever direction it pushes field position.

## Team defenses

One row per team-week; the "player" is the club. Scoring is ESPN-style and
configurable (`config.DefenseRules`): sacks 1, takeaways 2, touchdowns 6,
safeties and blocks 2, plus the points-allowed step function. It reproduces
2024's actual DST leaderboard — Denver 167, Minnesota 155.

| rung | MAE | CRPS | within-position ρ | vs recency |
|---|---:|---:|---:|---:|
| climatology | 4.3193 | 3.0460 | 0.001 | +0.22% |
| recency-mean | 4.3172 | 3.0394 | 0.085 | — |
| `defense-history` | 4.3238 | 3.0365 | 0.089 | **−0.10%** |
| `defense+opponent` | 4.2622 | 2.9826 | 0.208 | −1.87% |
| `defense+opponent+market` | 4.1353 | 2.8984 | 0.315 | **−4.64%** |
| `+weather` | 4.1489 | 2.9078 | 0.304 | −4.33% |

**The headline is the third row.** A defense's own box-score history — its recent
sacks, interceptions, fumble recoveries, points and yards allowed — is worth
−0.10% over its own recent fantasy points, which is a tie by a factor of
twenty-five, and moves within-position ordering 0.085 → 0.089. Six columns
describing what this defense has been doing add essentially no information about
what it will do next week. (An earlier build had this rung fractionally *worse*
than the baseline; the sign of a 0.1% effect is not resolvable here. The claim
that survives is the size, not the direction.)

What does work is the other team. Adding the opponent's recent scoring is −1.87%;
adding the closing line on top is −4.64% — **forty-six times** the own-history
rung — and nearly quadruples the ordering correlation, 0.085 → 0.315. A DST projection is mostly a projection of the
opponent, and the reason is structural rather than empirical: the points-allowed
half of the response is a step function of the other side's final score, and it
dominates the variance of the whole. The event half is a modest, noisy count.

Weather makes it *worse* here (+0.32%), which is what the kicker result
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
| weather on skill positions, pooled | tie (−0.04% full block, −0.09% gated) |
| weather on skill positions, **wind 15+** | **−0.60% CRPS on 5.6% of rows** — conditionally material, not promoted |
| gated wind hinge | better than the full block: 2/3 folds against 1/3, and no cost on calm rows |
| temperature / freezing | **null and slightly negative** (+0.38% on freezing rows); dropped from the gated form |
| kicker scoring is non-stationary | confirmed: +0.58 pts/game after the 2024–25 kickoff rules, from attempts not accuracy |
| era normalisation for it | **built and failed**; the shift cannot be learned from pre-change data |
| weather on kickers | material: roof −0.32% on 3/3 folds, readings −0.40% on 2/3 — **not promoted**, roof is the better candidate |
| kicker next-week | built and beats persistence by 2.6% CRPS |
| defense next-week | built and beats persistence by 4.4% CRPS |
| K/DST rest-of-season | `direct-total` promoted; simulator arms lose and are diagnosed |
| opposing-defense metrics | were already in; documented above |

## What is worth doing next

1. **Join the Open-Meteo forecast archive** and re-run both the kicker ladder and
   the gated wind hinge. Every weather number here is a ceiling measured on
   conditions recorded at the game; the forecast version is what ships, and the
   gap between the two is the cost of not seeing the future.
2. **A manual kicker-era adjustment** in `data/manual/`, since the section above
   shows the shift is not learnable. This is the single largest known bias in
   any projection this package currently produces.
3. **An availability denominator of team-games** for the specialist
   rest-of-season simulator, which is the identified cause of the kicker bias.
4. **Strength of remaining schedule for defenses.** The one-week opponent column
   is the largest single lever on the weekly response, and its rest-of-season
   analogue — the quality of the offences a defense still has to face — is not
   built. The repo has an `sos/` directory that predates this work.
5. **Recursive simulation for the specialist panels**, if and only if the
   rest-of-season response turns out to matter more than `direct-total` already
   delivers.
