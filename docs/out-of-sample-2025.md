# The 2025 season, out of sample

Fit on 2015–2024, scored on 2025. This is the strongest test in the package,
because 2025 is the one season nothing here has seen.

Every promotion decision — the efficiency exposure floor at 5, the innovation
cap at 0.25, the innovation-scale calibration, the protected-stream override —
was selected and scored on 2022/2023/2024. Nested inner folds protect a single
procedure against selecting on its own test set, but they cannot protect against
the degrees of freedom accumulated across a session of separate decisions. 2025
can, and only for as long as nothing tunes on it. `HOLDOUTS` therefore stays
`(2022, 2023, 2024)`.

## Total fantasy points

| scoring | metric | 2022–24 mean | 2025 | |
|---|---|---:|---:|---|
| standard | MAE | 32.808 | **31.211** | −4.9% |
| standard | CRPS | 23.921 | **22.738** | −4.9% |
| standard | RMSE | 46.992 | 46.219 | −1.6% |
| half-PPR | MAE | 37.883 | **35.939** | −5.1% |
| half-PPR | CRPS | 27.723 | **26.249** | −5.3% |
| PPR | MAE | 43.139 | **40.746** | −5.5% |
| PPR | CRPS | 31.638 | **29.862** | −5.6% |
| PPR | RMSE | 59.126 | 58.584 | −0.9% |

Point accuracy and CRPS are better out of sample than the in-window average on
all three scoring systems, and PPR MAE of 40.75 beats every individual in-window
fold including the best of them (41.56). Nothing here looks like a model that
was tuned into its validation window.

## The volume layer

| stream | MAE 2022–24 → 2025 | CRPS 2022–24 → 2025 |
|---|---|---|
| pass_qb | 5.257 → **4.294** | 4.030 → **3.302** |
| qb_workload | 0.153 → **0.128** | 0.119 → **0.098** |
| target | 0.868 → **0.823** | 0.637 → **0.595** |
| snap | 0.148 → **0.139** | 0.103 → **0.096** |
| availability | 0.216 → 0.214 | 0.146 → 0.143 |
| carry | 0.787 → 0.844 | 0.567 → 0.594 |

Carry is the only stream that degrades, on both metrics. Everything else
improves.

## The one real defect: 95% intervals are too narrow

Judged in counts against a binomial reference, the 80% intervals are fine and
the 95% intervals are not — and this reproduces out of sample:

| | cov80 misses / expected | z | cov95 misses / expected | z |
|---|---:|---:|---:|---:|
| 2025 (n=508), PPR | 109 / 101.6 | +0.82 ✓ | **46 / 25.4** | **+4.19** |
| 2025, half-PPR | 110 / 101.6 | +0.93 ✓ | 44 / 25.4 | +3.79 |
| 2025, standard | 107 / 101.6 | +0.60 ✓ | 40 / 25.4 | +2.97 |
| in-window (n=1531), PPR | 342 / 306.2 | +2.29 | 118 / 76.6 | +4.86 |

PPR coverage at the 95% level is 0.909 against a 0.95 nominal — about four
coverage points too narrow — and the same gap appears in every scoring system,
in-window and out.

This one is worth trusting where several earlier coverage findings were not.
Three reasons:

1. **The direction rules out the artifact that invalidated the others.** An atom
   at zero *inflates* coverage, because any interval containing zero covers a
   zero outcome. This is under-coverage. A zero atom would be hiding some of it,
   not creating it.
2. **It reproduces on unseen data**, at z=+4.19 on 2025 alone.
3. **It is on the output**, not an intermediate stream — total fantasy points is
   what the package publishes.

Note the 80% level is *better* calibrated out of sample (z=+0.82) than in-window
(z=+2.29). The problem is specific to the tails, which is what a too-thin
predictive distribution looks like: the body is right and the extremes are not
reached often enough.

## Where that leaves things

The volume and efficiency engine generalizes. Point accuracy, CRPS and 80%
coverage all hold up or improve on a season nothing here has seen, which is the
main thing this test was for.

The tail calibration does not. Somewhere between the volume draws, the
efficiency draws and the scoring simulation, the extremes are too rare — a
season where a player vastly outperforms or collapses is under-represented. That
is a distributional question rather than a parameter, and it is now the
best-evidenced open item in the package.

---

# Attributing the tail deficit (2026-08-04)

Three layers could own the missing tail mass. Each was tested by turning it up
alone, from one fit on 2015–2024 predicted onto 2025, scored in counts against a
binomial reference over n=508 (expected misses 101.6 at the 80% level, 25.4 at
the 95%).

## Not the efficiency layer

Scaling the efficiency dispersion:

| dispersion | cov80 | cov95 | MAE | CRPS |
|---:|---:|---:|---:|---:|
| 0.0 | 0.778 | 0.906 | 40.744 | 29.872 |
| **1.0** | **0.785** | **0.909** | **40.746** | **29.862** |
| 2.0 | 0.795 | 0.915 | 40.785 | 29.945 |
| 3.0 | 0.811 | 0.925 | 40.875 | 30.146 |

Deleting the efficiency noise entirely costs three tenths of a coverage point,
and tripling it recovers 1.6 of the 4.1 points missing at the 95% level while
MAE rises. A layer whose full removal barely moves the statistic is not the
statistic's source.

## Not volume–efficiency dependence

The accepted scoring path draws efficiency from independent marginals.
`draw_conditioned_efficiency` evaluates the fitted efficiency means at each
simulated volume draw, so a player drawing a heavy workload also draws the
efficiency that goes with it — positive dependence, which fattens both tails of
the product.

| path | cov80 | z | cov95 | z |
|---|---:|---:|---:|---:|
| independent (accepted) | 0.785 | +0.82 | 0.909 | +4.19 |
| draw-conditioned | 0.781 | +1.04 | 0.911 | +3.99 |

Two tenths of a point. Whatever the case for coupling the layers, closing this
gap is not it.

## It is the volume layer — but it is shape, not width

Stretching the volume count draws k-fold about each row's own posterior mean,
with the per-game rates the efficiency exposures read stretched to match:

| k | cov80 | z80 | cov95 | z95 | MAE |
|---:|---:|---:|---:|---:|---:|
| **1.00** | 0.785 | **+0.82** ✓ | 0.909 | **+4.19** | 40.746 |
| 1.25 | 0.848 | −2.73 | 0.929 | +2.16 | 40.891 |
| 1.50 | 0.870 | −3.95 | 0.939 | +1.14 | 41.304 |
| 2.00 | 0.900 | **−5.61** | 0.949 | **+0.12** ✓ | 42.999 |
| 3.00 | 0.921 | −6.83 | 0.967 | −1.71 | 50.491 |

At k=2.0 the 95% level lands on nominal almost exactly — 26 misses against 25.4
expected — and the 80% level is destroyed, 51 misses against 101.6. The volume
layer is where the missing mass lives, and **no single width fixes it**: the
body is already right at k=1 and the tails are not, so any scaling that repairs
the tails over-widens the body by more than it gains.

That is excess kurtosis, not deficient variance. The volume predictive is
correctly spread through its middle and too light in its extremes, which in this
domain is the observation that a fixed fraction of players every season either
break out or collapse by more than any smooth season-average process will
generate.

## What this rules in

The remaining candidates are all inside the volume generator's shape:

- the count likelihoods' tail behaviour (NegativeBinomial dispersion is fitted
  to the bulk and its tail is determined by the same parameter);
- the absence of a heavy-tailed player-level random effect — a Student-t
  role term would leave the body alone and thicken the extremes, which is
  exactly the shape the sweep asks for;
- regime change not represented as such: a player whose role genuinely changes
  between seasons is a mixture, and a unimodal predictive cannot produce one.

The third is the most interesting because `SeasonRegimeModel` already exists in
this package and is currently used for covariates rather than as a predictive
mixture. Note also that this diagnostic scales every row identically, so it
cannot distinguish "all rows slightly too thin" from "a minority of rows badly
too thin" — and the mixture reading predicts the latter. Splitting the deficit by
row is the next measurement, not another global knob.

Raw numbers in `scripts/validation_runs/tail_attribution_2025.json`.
