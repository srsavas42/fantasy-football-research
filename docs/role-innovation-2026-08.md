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

Neither promotes under the current gate. `calibrated_innovation` fails only on
the protected-stream allowance: pass-attempt MAE +0.91% and workload MAE +1.02%
against 0.5%, with workload CRPS +0.60% just over. Everything else it touches is
unchanged or better, and it buys nine coverage points on the layer with the
pipeline's worst calibration.

That is a policy question rather than a measurement one — how much point
accuracy on the quarterback streams is nine points of coverage worth — so both
flags stay off pending that call.
