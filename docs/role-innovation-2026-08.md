# The role innovation: a mean bias fixed, a scale bug found

Two defects in the same three lines of code. One was corrected and rejected by
the gate. Diagnosing why it was rejected turned up the other, which is larger,
pre-existing, and still open.

## The mean bias (corrected, rejected, flag off)

The role models perturb a log-odds vector with Gaussian noise and take a
softmax. The softmax is not linear, so renormalization takes probability mass
from whoever leads the room and hands it to everyone else. Nothing in the model
claims that transfer is real, and its size grows with how concentrated the room
is:

| room | allocation | draw-average | loss |
|---|---:|---:|---:|
| quarterback (0.90 / 0.08 / 0.02) | 0.9000 | 0.8733 | **−0.0267** |
| two-man (0.65 / 0.35) | 0.6500 | 0.6306 | −0.0194 |
| seven-deep target room | 0.2600 | 0.2526 | −0.0074 |

At the quarterback room's innovation scale of 0.60 that is about nine tenths of
an attempt per game going to passers nobody projected to take them.

`mean_preserving_shares` solves for a per-player offset, constant across draws,
that restores the draw-average to the noiseless allocation — proportional
fitting in log space, four passes, machine precision at these room sizes. As a
pure location shift it preserves every pairwise log-odds contrast exactly.

**The gate rejects it**, against the promoted configuration on 2022/2023/2024:

| stream | pooled | verdict |
|---|---:|---|
| pass_qb CRPS | **+4.30%** | regressed (protected, 0.5% allowance) |
| qb_workload CRPS | **+5.42%** | regressed (protected) |
| pass_qb cov80 | +0.053pp | regressed |
| target CRPS | +0.41% | regressed |
| qb_workload cov80 | −0.059pp | improved |
| qb_workload cov95 | −0.075pp | improved |
| carry MAE | −0.58% | improved |
| qb_workload MAE | −0.65% | inconclusive (sign varies) |
| pass_qb MAE | +0.37% | inconclusive (sign varies) |

Point accuracy is a wash and CRPS is much worse. `mean_preserving_innovation`
stays `False`. The code is kept because the bias it removes is a mathematical
fact rather than a judgement call, and because the reason it fails is
informative.

## Why it failed: the scale is fed in on the wrong side of the softmax

`_estimate_role_innovation` measures the RMS log-share residual between observed
and expected shares — that is, the *realized* dispersion of shares around the
deterministic allocation. That number is then assigned directly to
`role_innovation_scale` and used as the standard deviation of the noise added to
`eta`, on the **input** side of the softmax.

Those are not the same quantity. Renormalization compresses, so a given input
scale realizes less dispersion than it started with, by an amount that depends
on room shape:

| room | input scale | realized dispersion | ratio |
|---|---:|---:|---:|
| two-man | 0.60 | 0.4235 | **0.70** |
| quarterback, 3-deep concentrated | 0.60 | 0.4886 | **0.82** |
| three-man even | 0.60 | 0.4897 | 0.82 |
| seven-deep target room | 0.60 | 0.5533 | 0.93 |

So the pipeline realizes 70–93% of the churn it measured in the data, and it is
worst in the small concentrated rooms — which is to say, worst at quarterback.
The model is systematically under-dispersed there by construction.

That is visible in the validation output and had been read as noise. Quarterback
workload coverage against an 80% nominal interval:

| fold | cov80 |
|---|---:|
| 2022 | 0.647 |
| 2023 | 0.619 |
| 2024 | 0.726 |

Six to eighteen points of undercoverage, in the same direction every year.

It also explains the rejection above. The mean-preserving correction happens to
pass slightly more dispersion through — 0.5081 against 0.4886 at scale 0.60 —
which is why it is the one change that *improved* quarterback workload coverage
while making everything else worse. It moved a mis-scaled distribution's mean
without fixing the scale.

## The fix, and why it is not in this branch

Invert the map: numerically solve for the input scale whose realized dispersion
matches what `_estimate_role_innovation` measured. The inversion is stable and
close to linear over the relevant range — for the concentrated quarterback room
a target of 0.45, 0.60 or 0.75 needs an input of 1.23×, 1.23× and 1.22×
respectively — so a per-room-shape correction factor computed once at fit time
would do it.

This is left open deliberately. It is a bigger change than anything else in this
branch, it moves the calibration of every allocation layer at once, and it needs
its own walk-forward rather than being folded into a branch that has already
been validated. The mean-preserving correction should be re-tested on top of it
rather than separately: the two interact, and the evidence above is that
correcting the mean while the scale is wrong makes the distribution worse.

---

# The scale fix, measured (2026-08-03)

`calibrated_innovation` inverts the map: it bisects on the rooms the model was
actually fitted over to find the input scale whose *realized* log-share spread
equals what the estimator measured.

On real data the problem is narrower than the toy rooms above suggested, because
room size drives everything and only one room is small:

| layer | mean room | measured | realized | |
|---|---:|---:|---:|---|
| QB workload | **2.37** | 1.2437 | 0.9465 | **76%** |
| target | 19.1 | — | — | 97% |
| carry | 22.9 | — | — | 98% |

Calibration raises the quarterback input scale to 1.6361 (1.32×), which realizes
1.2440. Target and carry barely move and are capped anyway.

## Three configurations, against the promoted baseline

| config | wl cov80 | wl cov95 | wl MAE | wl CRPS | qb MAE | qb CRPS |
|---|---:|---:|---:|---:|---:|---:|
| mean-preserving only | −5.94pp | −7.50pp | −0.65% | +5.42% | +0.37% | +4.30% |
| **calibrated only** | **−9.23pp** | **−9.32pp** | +1.02% | **+0.60%** | +0.91% | **+0.27%** |
| calibrated + mean-preserving | −6.33pp | −8.93pp | −0.65% | +7.29% | +0.36% | +5.47% |

Per fold, quarterback workload coverage under calibration:

| fold | cov80 (nominal 0.80) | cov95 (nominal 0.95) |
|---|---|---|
| 2022 | 0.647 → 0.824 | 0.788 → 0.906 |
| 2023 | 0.619 → 0.774 | 0.810 → 0.940 |
| 2024 | 0.726 → 0.881 | 0.869 → **1.000** |

A layer that ran six to eighteen points under nominal now sits within a few
points of it. It is not uniformly better: 2024 reaches 1.000 at 95%, so on that
fold the intervals cover every observation and the correction over-widens rather
than merely repairing.

## A correction to what this document previously claimed

It recorded that the mean-preserving correction failed *because* the scale was
wrong, and that the two "interact, and the evidence is that correcting the mean
while the scale is wrong makes the distribution worse" — implying it would fare
better once the scale was fixed.

It does not. On top of calibration its CRPS cost is **larger**, not smaller:
+7.29% against +5.42% on workload share, +5.47% against +4.30% on pass
attempts. The cost is intrinsic to the correction rather than a consequence of
the scale.

What the pairing does show is the other half of the mechanism. Calibration
raises the input scale, which enlarges the softmax renormalization bias, which
is why calibration alone costs about 1% of point accuracy. Mean preservation
removes exactly that — workload MAE goes +1.02% → −0.65%, pass MAE +0.91% →
+0.36% — and charges 5 to 7% of CRPS for it. That trade is not worth taking.

## Status

**`calibrated_innovation` is promoted, 2026-08-03**, over the protected-stream
allowance and with that override recorded in
[acceptance-gate.md](acceptance-gate.md). It fails the gate on one thing only —
pass MAE +0.91% and workload MAE +1.02% against 0.5% — and buys nine coverage
points on the layer with the pipeline's worst calibration, with CRPS flat. The
owner accepted that trade explicitly.

### The accepted cost did not reach the product

Measured after the fact, on total fantasy points against the promoted baseline:

| metric | standard | half-PPR | PPR |
|---|---:|---:|---:|
| CRPS | −0.67% | −0.68% | −0.68% |
| RMSE | −0.68% | −0.61% | −0.55% |
| MAE | −0.54% | −0.47% | −0.33% |

Every one improves on all three folds, coverage moves are negligible, and the
scoring gate accepts with no exception required.

So the quarterback-stream MAE regression is real at the layer the gate measures
and does not survive to the layer that matters. Shares are multiplied by team
totals and integrated against efficiency downstream, and a better-calibrated
allocation pays back more than the percent it costs. The override was needed to
promote the change and was not needed to justify it.

`mean_preserving_innovation` stays off. It is the only way to recover the MAE
that calibration costs, and it charges 5 to 7% of CRPS to do it, which is a
worse deal than the one being made here.

What follows was written before that call.

Neither promotes under the current gate. `calibrated_innovation` fails only on
the protected-stream allowance: pass-attempt MAE +0.91% and workload MAE +1.02%
against 0.5%, with workload CRPS +0.60% just over. Everything else it touches is
unchanged or better, and it buys nine coverage points on the layer with the
pipeline's worst calibration.

That is a policy question rather than a measurement one — how much point
accuracy on the quarterback streams is nine points of coverage worth — so both
flags stay off pending that call.

---

# The 2024 fold was not over-widening (2026-08-03)

This document previously read the 2024 fold's `cov95` of 1.000 as the
correction over-widening rather than merely repairing. That was wrong, and it
was wrong in an avoidable way: a coverage *rate* on one fold is a poor statistic
when the event it counts is rare.

In counts, over 84 quarterback rows, a 95% interval expects about four misses
with a standard deviation of two:

| level | fold | n | misses | expected | z |
|---|---|---:|---:|---:|---:|
| cov95 | 2022 | 85 | 8 | 4.3 | +1.87 |
| cov95 | 2023 | 84 | 5 | 4.2 | +0.40 |
| cov95 | 2024 | 84 | **0** | 4.2 | −2.10 |
| cov95 | **pooled** | 253 | **13** | **12.7** | **+0.10** |
| cov80 | pooled | 253 | 44 | 50.6 | −1.04 |

Pooled, the calibrated model lands on nominal almost exactly at 95% — 13 misses
against 12.7 expected, inside the [6, 20] binomial range — and mildly
conservative at 80%. The baseline it replaced recorded 45 and 85, both far
outside.

P(0 misses | n=84, p=0.05) is 0.0135, so seeing a fold like 2024 in at least one
of three has probability 0.04. Unusual, and entirely ordinary noise. There is no
over-widening to explain.

`scripts/coverage_calibration.py` reports this for any run so the next such
number is not judged by eye.

## What that tool then found

Run across every stream, the quarterback room this work repaired is now the
**best-calibrated layer in the pipeline**, and the skill-position streams — each
carrying seven times the sample — are worse. Pooled over 1,754 rows, and
unchanged by the calibration work:

| stream | cov80 misses / expected | z | cov95 misses / expected | z |
|---|---:|---:|---:|---:|
| carry | 209 / 350.8 | **−8.46** | 76 / 87.7 | −1.28 |
| target | 272 / 350.8 | **−4.70** | 119 / 87.7 | **+3.43** |
| snap | 298 / 350.8 | −3.15 | 88 / 87.7 | +0.03 |
| qb_workload | 44 / 50.6 | −1.04 | 13 / 12.7 | +0.10 |

Carry's 80% intervals contain 88% of outcomes: eight points too wide, at z=−8.46.
Target is miscalibrated in **both directions at once** — 84.5% coverage at the
80% level and 93.2% at the 95% level — which is too wide in the body and too
thin in the tails, and that is a distributional shape problem rather than a
width one. Neither is caused by anything in this branch; the before and after
counts differ by two.

This is left open. It is a modelling question about the response distributions
themselves rather than a parameter to invert, and it is now the largest known
calibration defect in the pipeline.

---

# The innovation cap, validated (2026-08-03)

`innovation_cap` bounds how much role churn the target and carry allocators
represent. It sat at 0.50 and **bound on every fit** — measured dispersion is
1.43 for targets and 2.00 for carries — so it was never a safety rail catching
the occasional outlier, it was the operative parameter, and its value had never
been validated. The carry figure of exactly 2.00 is the estimator's own internal
clip, so the true measurement is unknown and only known to be larger: two
stacked ceilings, neither checked.

The sweep is cheap because of where the parameter lives. `role_innovation_scale`
is consumed at prediction time and enters no likelihood, so one posterior serves
every candidate — one pipeline fit per fold rather than one per candidate.
Selection is nested, and the criterion is distance from nominal coverage rather
than point accuracy, because the cap governs width.

| inner fold | None | 0.15 | 0.25 | 0.35 | 0.50 | 0.75 | picks |
|---|---:|---:|---:|---:|---:|---:|---|
| 2021 | 0.0424 | **0.0292** | 0.0292 | 0.0322 | 0.0331 | 0.0360 | 0.15 |
| 2022 | 0.0312 | 0.0296 | **0.0292** | 0.0317 | 0.0342 | 0.0350 | 0.25 |
| 2023 | 0.0451 | 0.0363 | **0.0346** | 0.0363 | 0.0372 | 0.0410 | 0.25 |

The penalty rises on both sides and uncapped is worst on every fold, so the
minimum is interior rather than an artifact of where the candidates stop.
**Promoted at 0.25**: the modal pick, optimal or within 0.0017 on all three.

The gate accepts it, against the promoted calibration: target CRPS −1.57%,
carry CRPS −1.21%, carry MAE −0.48%, target cov80 −1.81pp toward nominal, each
on all three folds, with the quarterback streams untouched.

## What the sweep separated

Tightening the cap moves target `cov80` from 0.868 to 0.819 against a 0.80
nominal — close to repaired. It moves carry from 0.880 to 0.869, and carry
remains about seven points over-covered **even uncapped**. So the innovation is
not carry's width source and no cap value will fix it; that excess comes from
elsewhere in the stack — the any-carry hurdle, the team total, or the
multinomial allocation. That is a separate defect from the one this sweep
addressed.

It also trades one part of the target stream's calibration for another. Pooled
over 1,754 rows the 80% level improves, z −4.70 → −2.91, while the 95% level
worsens, z +3.43 → +5.29. The change itself is not material at that level —
+0.44pp against a per-fold binomial standard error of 0.90pp — but the level was
already bad and remains so. Target is miscalibrated in both directions at once,
which is a distributional shape problem rather than a width one, and narrowing
the body cannot fix a thin tail.

---

# The skill-stream "miscalibration" was the diagnostic, not the model (2026-08-03)

The section above reported carry at z=−7.99 and target miscalibrated in both
directions, and called them the largest known calibration defect in the
pipeline. That was wrong, and wrong the same way the 2024 fold was: a statistic
applied to data it does not fit.

Interval coverage assumes a continuous outcome. Carries and targets are counts
with a large atom at zero — **49.6% of carry rows and 27.9% of target rows have
none**, which is 89% of tight ends and 62% of receivers for carries. Any
interval containing zero covers every one of those rows. The population rate
therefore cannot reach nominal no matter what the model does, and the excess
reads as over-wide intervals.

Split by the atom, on the 2024 holdout:

| stream | rows | cov80 | cov95 |
|---|---|---:|---:|
| carry | all 569 | 0.905 | 0.970 |
| carry | truth = 0 (282) | **1.000** | **1.000** |
| carry | **truth > 0 (287)** | **0.812** | **0.941** |
| target | all 569 | 0.863 | 0.954 |
| target | truth = 0 (159) | 0.962 | 0.975 |
| target | **truth > 0 (410)** | **0.824** | **0.946** |

On the rows that actually have carries or targets — the only rows a projection
is about — both streams sit within about two points of nominal at both levels.
The zero rows are covered with probability 1.000 by construction. There is no
width defect to find.

The variance decomposition that preceded this reached the same conclusion from
the other direction. Ablating each noise source on one fitted posterior:

| ablation | cov80 | mean 80% width |
|---|---:|---:|
| full model | 0.905 | 2.878 |
| no role innovation | 0.902 | 2.805 |
| no multinomial allocation noise | 0.898 | 2.860 |
| no any-carry hurdle | 0.877 | 2.596 |
| neither multinomial nor innovation | 0.898 | 2.786 |

Removing *every* stochastic component still leaves coverage near 0.90. Nothing
in the model produces that number, because the model is not what produces it.

`scripts/coverage_calibration.py` now warns on any stream with an atom at zero,
and the non-zero split is recorded in
`scripts/validation_runs/zero_atom_coverage.json`.

## What this leaves

The two "open findings" this document raised against the skill streams are
withdrawn. The estimator's internal 2.0 clip is also moot in practice: with the
innovation cap promoted at 0.25, `min(measured, cap)` is 0.25 whether the
measurement saturates at 2.0 or not, so the clip can only matter if the cap is
removed.

That is three of three closed as measurement artifacts rather than defects,
which is worth stating plainly given how confidently the opposite was written
here a few hours earlier.

---

# The mean-preserving correction does not fix the carry bias (2026-08-04)

The carry allocator over-projects quarterbacks and under-projects running backs
in every fold: QB +31.1% / +25.7% / +36.4% and RB −6.9% / −6.4% / −5.2% on the
2023, 2024 and 2025 holdouts. Aggregate carry MAE never flagged it because the
two errors partly cancel against the conserved room total.

I attributed that to softmax renormalization — mass leaving the room's leader
and spreading to its minor members, which in a carry room means running backs
down and quarterbacks up. The per-layer `mean_preserving_innovation` flag exists
to remove exactly that. This measures whether it does.

## The test

The flag is read only in `_role_share_prediction` and never during fit, so one
posterior serves both arms. Predicting twice with the same seed makes this
exactly paired: the innovation draws are identical and the sole difference is
the renormalization correction. The target stream is bit-identical across arms,
confirming the isolation, and carry draws move by 0.114 per team game on
average, confirming the correction is not a no-op.

| pos | n | obs/gm | baseline | mp | change |
|---|---:|---:|---:|---:|---:|
| QB | 85 | 1.523 | **+36.5%** | **+35.4%** | −1.0pp |
| RB | 149 | 4.430 | −5.2% | −4.9% | +0.3pp |
| WR | 216 | 0.092 | +7.2% | +5.1% | −2.1pp |
| TE | 125 | 0.036 | −78.2% | −78.8% | −0.6pp |

**One point of thirty-six.** The correction is mean-preserving by construction
and removes the renormalization bias exactly, so this is a measurement of how
much of the carry bias renormalization was ever responsible for: about a
thirtieth of it. The mechanism I named is real, present, and almost entirely
beside the point.

Per-position error moves the same negligible amount — QB MAE −0.49%, RB −0.14%,
WR −0.81%, TE −0.12% — and the room total is unchanged to three decimals
(27.158 both arms), as a mass-neutral correction must be.

## Where the bias is not

The exposure asymmetry I first expected is not there: median snap exposure is
0.110 for quarterbacks against 0.111 for running backs. Renormalization is now
measured at one point of thirty-six. That leaves the allocator's other inputs —
the role prior, the projected snap exposure, the any-carry hurdle and the team
rush total.

An earlier version of this section named the role prior, on the strength of a
comparison between the *median* role prior per position and the *aggregate*
observed carries per snap. Those are not comparable quantities: the median is a
typical row, which at quarterback is a backup, while the aggregate is dominated
by the handful of players taking most of the carries. The direction it appeared
to show does not survive weighting the prior the way the softmax weights it, so
that attribution is withdrawn.

`scripts/decompose_carry_bias.py` settles it properly, by substituting each
input with the truth in turn and reporting where the quarterback bias collapses.

## Status

`mean_preserving_innovation` stays off, now for a second and better reason. The
first was cost: on all three layers it charges 4–7% CRPS on the passing streams.
The per-layer split removes that objection for carry specifically — the matched
four-fold comparison shows every other stream identical to the last digit — but
what carry buys is −0.21% MAE pooled, under the gate's 0.25% materiality floor,
and none of the bias it was built to remove.

The flag and the per-layer machinery stay. The correction is mathematically
right and cheap to carry, and it is the only clean instrument for asking "how
much of this is renormalization?" of any allocation layer. The answer for carry
happens to be "almost none".

The QB/RB split remains open. It is not an allocation-noise question, which is
what this measurement was for; which of the allocator's remaining inputs owns it
is the next one.

---

# It was the exposure, not the prior (2026-08-04)

Substituting each of the allocator's inputs with the truth in turn, on 2025:

| rung | QB | RB | WR | TE |
|---|---:|---:|---:|---:|
| as served | **+36.5%** | −5.2% | +7.2% | −78.2% |
| + observed snap share | **+12.9%** | +1.8% | −6.1% | −83.2% |
| + observed eligibility | +7.1% | +0.4% | +25.6% | −79.6% |
| + observed team total | +6.1% | −0.6% | +25.1% | −80.9% |

Knowing how much each player actually played removes **23.6 of the 36.5
points**, and running backs go from −5.2% to +1.8% at the same rung. The
any-carry hurdle removes another 5.8. What survives all three substitutions is
about six points, which is the role prior and the softmax together.

The role prior is not the problem. Measured the way the allocator uses it —
weighted by snaps within the room rather than as a median over rows — it is
close to exact where it matters:

| pos | prior (snap-weighted) | realized | ratio |
|---|---:|---:|---:|
| QB | 0.06577 | 0.06453 | **1.019** |
| RB | 0.29161 | 0.30858 | 0.945 |
| WR | 0.00864 | 0.00439 | 1.968 |
| TE | 0.00079 | 0.00163 | 0.484 |

Both earlier attributions were wrong, in opposite directions, and both came from
comparing quantities that were not comparable. This is the third statistic in
this document to fail that way.

## What the exposure gets wrong

Quarterback snap share is bimodal and the projection is not. Observed on 2025:

| pos | n | mean | p10 | p50 | p90 | share below 0.05 |
|---|---:|---:|---:|---:|---:|---:|
| **QB** | 117 | 0.275 | 0.000 | 0.091 | **0.913** | **44.4%** |
| RB | 181 | 0.195 | 0.000 | 0.105 | 0.527 | 38.7% |
| WR | 248 | 0.317 | 0.000 | 0.264 | 0.737 | 23.4% |
| TE | 157 | 0.279 | 0.000 | 0.227 | 0.627 | 27.4% |

A quarterback either plays nearly every snap or almost none; the median of 0.091
describes almost no actual quarterback. A unimodal predictive over that shape
hands backups exposure they will never get, and `log(exposure)` enters `eta`
directly, so that exposure buys carries. It also explains why the bias was flat
in absolute terms across usage bands while reaching +131% in relative terms on
backups — the error is concentrated on players whose true exposure is near zero.

The other two rows are worth noting even though they are small in absolute
terms. Receivers are over-allocated by a quarter once the hurdle stops hiding
it, and tight ends are under-projected by four fifths at every rung, prior
included.

Recorded as task 33, against the snap model rather than the allocator.
