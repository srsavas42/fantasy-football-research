# Can the hierarchical rest-of-season model's intervals be repaired after the fact?

*September 2026*

**No. The width is fixable and the shape is not, and the shape is the problem.**
A single scalar per fold puts 80% coverage exactly on nominal — 0.576 → 0.800 —
and the structured model still loses to the direct regression on CRPS, on MAE,
on ordering, and at the 95% level. It is not promoted.

## Why it was worth asking

The hierarchical estimator is a viable *mean*. With the full feature surface it
reaches **MAE 30.28** against the shipped direct regression's **30.32** — a dead
heat — and it is rejected on calibration alone:

| arm | MAE | CRPS | cov@80 | ρ |
|---|---:|---:|---:|---:|
| direct-total+everything | **30.06** | **21.27** | **0.789** | **0.780** |
| aggregated-weekly (full features) | 30.28 | 23.23 | **0.575** | 0.752 |
| hierarchical (defaults) | 30.45 | 23.07 | 0.579 | 0.774 |
| independent-weeks | 30.45 | 23.22 | 0.564 | 0.774 |

An earlier note quoted the *defaults* arm against the full-feature control, which
was not a fair comparison — `use_phase`, `use_adp` and `use_role` were all off.
Given the same surface the mean is a tie, so adaptation to in-season data was
never the issue: `HierarchicalSeason` is fitted on player-weeks and reads the
same lagged usage and the same pre-game injury and depth feeds the control does.

Nor is the interval failure an information problem. The full arm covers **0.575**
against the stripped arm's 0.579 — more features leave it very slightly *worse* —
which is the signature of a variance that is composed rather than observed. A
Beta concentration read off realized play counts, a persistent level SD read off
the covariance between distinct weeks of the same player, and a week-noise scale
backed out so the residual pool is not double-counted: each is a point estimate
plugged in as though known, and the product is too narrow.

That leaves a clean question. If the *shape* is right and only the width is
wrong, one number fixes it and the structured model becomes usable — with the
interpretability the control cannot offer, since availability and magnitude stay
separate layers.

## The test

`scripts/calibrate_hierarchical_ros.py` stretches each row's draws about their
own median — the median because the totals are right-skewed and a scale about the
mean would shift the centre as it widens — and clips at zero, since buying
coverage with negative season totals is not buying coverage.

**The factor is estimated out of sample.** For holdout `Y` the model is fitted on
seasons before `Y-1`, predicts `Y-1`, and the width that would have made `Y-1`
cover nominally is solved there; the model is then refitted on everything before
`Y` and that stored factor applied. `Y` never informs its own calibration. This
is the nesting `fit_blend_weights` already uses.

The solved factors are strikingly stable, which is itself a finding — the
deficit is a consistent structural ~1.7x, not fold noise:

| holdout | inner season | raw coverage | factor |
|---|---|---:|---:|
| 2023 | 2022 | 0.603 | 1.70 |
| 2024 | 2023 | 0.594 | 1.65 |
| 2025 | 2024 | 0.562 | 1.80 |

## It fixes the band it was tuned on, and nothing else

| arm | MAE | CRPS | cov@80 | cov@95 | ρ |
|---|---:|---:|---:|---:|---:|
| direct | **30.06** | **21.27** | 0.789 | **0.945** | **0.780** |
| hierarchical | 30.28 | 23.23 | 0.576 | 0.752 | 0.752 |
| hierarchical + calibrated | 30.56 | 22.06 | **0.801** | 0.911 | 0.752 |

Coverage at 80% lands on nominal to three decimals. CRPS recovers about 60% of
the gap and remains **3.7% worse**. MAE gets *worse*, 30.28 → 30.56, because
widening a right-skewed distribution that is clipped at zero moves its mean.
Ordering does not move at all — a width cannot reorder anything.

## Why one factor cannot be enough

The width required to hit each nominal level, on 2025:

| nominal | direct | hierarchical raw | after x1.80 | factor needed |
|---:|---:|---:|---:|---:|
| 0.50 | 0.488 | 0.353 | 0.556 | **1.55** |
| 0.80 | 0.799 | 0.571 | 0.820 | **1.70** |
| 0.90 | 0.904 | 0.682 | 0.888 | **1.90** |
| 0.95 | 0.950 | 0.756 | 0.919 | **2.55** |

If the distribution were the right shape and merely too narrow, that last column
would be one constant. It rises monotonically and half again from the median to
the 95th, so the tails are disproportionately thin: the simulator is not a
squeezed version of the truth, it is the wrong distribution. Tuning at 80%
necessarily under-covers at 95% and over-covers at 50%, which is exactly what the
`after x1.80` column shows.

## What this rules in and out

Post-hoc calibration is **rejected** — cheap, honestly measured, and insufficient.

What it does buy is a sharper diagnosis than "the intervals are too narrow". The
defect is in the tails, which points at the layer uncertainties being plugged in
as point estimates: `concentration` and `level_sd` are each estimated from one
slice and then treated as known, so none of *their* uncertainty reaches the
total. Propagating them — a posterior over the concentration rather than a
number, a t-tailed rather than Gaussian persistent level — is the change the
shape argues for, and it is a real piece of work rather than a patch.

Nothing shipped changed. The direct regression remains the rest-of-season model.
