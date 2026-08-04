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
