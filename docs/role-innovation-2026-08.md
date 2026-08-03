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
