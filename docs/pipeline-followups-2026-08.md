# Follow-up work after the pipeline review

Run 2026-08-02 against merged `8446d75`. nflverse point-in-time rosters,
holdouts 2022/2023/2024, the volume-v3 protocol at 1000 draws / 1000 tune /
4 chains unless stated. Scoring comparisons run at 400/400/2 to match the
baseline they are measured against. Every configuration reads one cached build
of the season-average frames, so arms differ only in the flag under test.

## E1 — the sampler now uses more than one core

`sample_model` hardcoded `cores=1`, so every model ran its chains one after
another; the season-average pipeline fits eight components, so a three-fold
walk-forward paid that twenty-four times. Chains are independent given their
seeds, so it was pure wall clock.

Verified the posterior does not move: same seed and data give `mu` and `sigma`
identical to 1e-12 at one core and four. A small model went 22.8s → 3.8s.
`FFMODEL_SAMPLING_CORES` pins it, including back to 1.

## Step 1 — the team model's R-hat, resolved

Two structural causes and one measurement artifact.

**Redundant dispersion.** `plays_obs` is NegativeBinomial — a Poisson-Gamma
mixture already carrying overdispersion through `play_alpha_pg` — and a per-row
log-normal term was added to the same mean. Two dispersion sources on one
likelihood are not separately identified, and the fit showed it: the variance
posterior sat at 0.008 with sd 0.006, pressed against zero, with the intercept's
bulk ESS at 376. The pass, sack and target likelihoods are Binomial and have no
free dispersion parameter, so their transition terms stay.

**Intercept/team-effect confounding.** Each stream's intercept and its 32 team
effects are additively confounded; `Normal(0, small)` identifies that only
through the prior. The effects now sum to zero, the construction
`_position_effect` already used for position effects.

| configuration | team max R-hat | min bulk ESS |
|---|---:|---:|
| merged main | 1.0177 | 376 |
| + one dispersion term | 1.0116 | 287 |
| + sum-to-zero team effects | 1.0107 | **1069** |

**The residual 1.0107 is R-hat's own Monte Carlo noise, not convergence.** It
appears on one fold and one parameter, `pass_persistence` on the 2023 fold. At
1000 draws, seed 42 gives 1.01 and seed 7 gives 1.00 on identical data. At 2000
draws both seeds give 1.00, with bulk ESS 2576 and 2919.

Predictive effect of the two changes, full budget: target MAE -0.05% (3/3),
carry MAE -0.03% (2/3), pass MAE -0.09% (2/3); everything downstream unchanged.
Sum-to-zero is metric-neutral on its own (all responses within ±0.06%) and is
worth having for the 3.7x ESS.

## Step 2 — why the 2023 fold behaves differently

The same fold carries both the coupling's MAE losses and the residual R-hat, and
the parameter involved is `pass_persistence` — the slope on the lagged team pass
rate. That points at the data rather than the model.

League pass rate by season shows why. 2022 is the second-largest year-over-year
move in the window, and it is the **last training season for the 2023 holdout**:

| season | league pass rate | change |
|---|---:|---:|
| 2021 | 0.5636 | -0.0031 |
| **2022** | **0.5504** | **-0.0132** |
| 2023 | 0.5565 | +0.0062 |

So the 2023 fold fits its persistence slope on a history ending in a sharp
pass-rate drop, and then scores a season that partially reverts. The pass stream
is exactly where that fold's error concentrates, and `pass_persistence` is
exactly the parameter the likelihood and prior disagree about.

This is a property of the holdout, not a defect in the coupling. It also means
the gate's "win two of three holdouts" rule is being applied across folds that
are not exchangeable — one of them straddles a regime shift.

Splitting quarterback error by how much of the job each passer actually held
shows the mechanism directly. Error is per team game; bias is predicted minus
observed, so negative means under-projected.

| fold | group | MAE uncoupled → coupled | bias uncoupled → coupled |
|---|---|---|---|
| 2022 | starter (≥20 att/gm) | 5.45 → 5.43 | -1.11 → -2.48 |
| 2022 | committee (2-20) | 6.02 → **5.47** | -0.37 → -0.93 |
| 2023 | starter | 7.49 → 7.98 | **-4.42 → -5.92** |
| 2023 | committee | 7.35 → **6.74** | +1.35 → +0.82 |
| 2024 | starter | 5.20 → 5.73 | -2.04 → -3.75 |
| 2024 | committee | 6.99 → **6.50** | -0.49 → -1.06 |

Three things follow.

Starters are under-projected in **every** fold, before the coupling touches
anything. That is an upstream bias, and 2023's is twice the size of either
other fold's — which is what a persistence slope fitted through a pass-rate
drop and then scored on a partial rebound should produce.

The coupling moves share off starters and onto the committee, which is the
correction it was built to make: committee MAE improves in all three folds. The
cost is that starters move further into a negative bias they already had, by
about 1.4 to 1.7 in each fold.

So the coupling's cost lands on a population that is already biased low, and it
is largest where that bias is largest. Overall quarterback MAE still improved in
all three folds in this diagnostic (4.82→4.56, 6.28→6.17, 5.04→4.97).

**Correction: the starter bias is not mainly a team-total problem.** An earlier
draft of this section blamed the team pass-attempt layer. Checking it, the
persistence baseline's team-level bias alternates sign and stays within about
one attempt per game (+1.04 in 2022, -0.32 in 2023, +0.93 in 2024), while the
starter bias is negative in every fold and up to 4.4. Most of the gap is
therefore in the *allocation*, not the team total.

The likely mechanism is the roster softmax itself. Predictions add innovation
noise in log space and then renormalise, and for a concentrated room that is not
mean-preserving — by Jensen, symmetric noise moves mass off the leader and onto
the tail. Simulating the prediction path at the workload model's default
`role_innovation_scale = 0.60`, on a room split 0.90 / 0.08 / 0.02:

| innovation sd | leader E[share] | bias |
|---:|---:|---:|
| 0.00 | 0.9000 | — |
| 0.30 | 0.8933 | -0.0067 |
| **0.60** | **0.8733** | **-0.0267** |
| 0.90 | 0.8418 | -0.0582 |

At 34 team attempts per game, -0.0267 of share is about **0.9 attempts per game
of systematic starter under-projection from the noise alone**, before any data
enters — and real quarterback rooms are often more concentrated than 0.90, where
the pull is larger. The quarterback room is the most concentrated simplex in the
pipeline, which is why this shows up there first; target and carry rooms are
flatter and should be affected far less.

That makes the roster-share innovation, not the team layer, the place to work
next: the noise needs to be applied so it does not move the expected share.

## Step 3 — postseason features, promoted for skill positions only

The features earn their place among the skill positions and lose it at
quarterback.

| response | change | fold wins |
|---|---:|---:|
| carry MAE | **-2.80%** | 3/3 |
| carry CRPS | **-1.48%** | 3/3 |
| snap MAE | **-0.82%** | 3/3 |
| snap CRPS | **-0.63%** | 3/3 |
| target CRPS | -0.28% | 2/3 |
| target MAE | +0.10% | 1/3 |
| pass_qb MAE | **+3.39%** | 0/3 |
| qb_workload MAE | **+3.27%** | 0/3 |

The pass-stream regression is far through the gate's 0.5% protected limit, so
the features no longer reach the workload layer. Re-run with that restriction,
the quarterback layers are untouched to five decimal places and the skill-
position gains are intact:

| response | change | fold wins |
|---|---:|---:|
| carry MAE | **-2.77%** | 3/3 |
| carry CRPS | **-1.47%** | 3/3 |
| snap MAE | **-0.82%** | 3/3 |
| snap CRPS | **-0.63%** | 3/3 |
| target CRPS | -0.29% | 2/3 |
| target MAE | +0.07% | 1/3 |
| pass_qb, qb_workload | **±0.00%** | protected |

Target 95% coverage improves 0.924 → 0.932; carry and snap coverage are flat.
Sampling is unchanged (max R-hat 1.0107, min ESS 1110, no divergences).

**Promoted**, with one documented exception: target MAE moves +0.07% winning one
holdout of three, against target CRPS -0.29% at 2/3. That is well inside fold
noise and is dwarfed by a 2.77% carry MAE gain won on every holdout, but it is
recorded rather than waved through — the same treatment the availability
coupling's exception got. The quarterback room is close
to winner-take-all and already well determined by depth chart and prior snap
share, so a signal present on 18% of rows and correlated with team strength
rather than with who takes the snaps is noise there. Availability and the team
layer were already excluded, for the related reason that qualifying for the
postseason is a fact about team quality.

## Step 4 — the matched `oof_*` estimator, measured but not promoted

Training-time `oof_*` covariates built by cross-fitting the production volume
pipeline, rather than by ridge, remove the coefficient attenuation the review
measured (the two constructions correlate 0.90-0.92, but the training-side one
was the less accurate).

| scoring | MAE | fold wins | CRPS | fold wins |
|---|---:|---:|---:|---:|
| standard | -0.07% | **3/3** | -0.01% | 1/3 |
| half-PPR | -0.06% | **3/3** | -0.01% | 2/3 |
| PPR | -0.06% | **3/3** | -0.02% | 3/3 |

The direction is exactly what removing attenuation predicts, and MAE wins every
holdout on every scoring system — but the effect is 0.06%, CRPS misses the
two-of-three rule on standard, and the run costs 3.5x the wall clock (995-1370s
per fold against 295-379s). `volume_feature_estimator` therefore stays `"ridge"`.
The matched path is implemented, validated as directionally correct, and worth
switching on if the covariate is ever given more weight than it currently
carries.

## Step 5 — the scoring coverage puzzle, resolved

season-scoring v1 recorded 95% coverage of 0.879-0.892 against a 0.90 floor.
Re-fitting both layers gives 0.916-0.920 on the same folds — on this branch and
on unmodified main alike — so whatever changed is not recent. v1 paired a
volume-v2 checkpoint with separately fitted efficiency posteriors.

The decisive test is where the interval width comes from.
`scale_efficiency_dispersion(0)` removes latent season-to-season efficiency
variation while keeping event noise, so what survives is volume plus sampling:

| holdout | 95% coverage, full | efficiency dispersion off |
|---|---:|---:|
| 2022 | 0.899 | 0.899 |
| 2023 | 0.903 | 0.895 |
| 2024 | 0.950 | 0.948 |

**Removing latent efficiency dispersion entirely moves 95% coverage by at most
0.008.** Total-scoring interval width is a property of the volume layer, not the
efficiency layer.

Two conclusions follow.

The v1 coverage number describes a volume layer that no longer exists. It was
measured against volume-v2; the volume layer has since been replaced by v3 and
then by the availability coupling and the fixes here. Comparing a current number
to it is not meaningful, and the gate should be re-run rather than referenced.

And v1's calibration ablations were pulling the wrong lever. Sweeping efficiency
dispersion from 0.75x to 1.50x could not have fixed coverage, because efficiency
dispersion is not what sets the width — which is exactly what that sweep found
("did not produce a candidate that passed accuracy, CRPS stability, and coverage
together"). The `1.10x` total-dispersion result that did move coverage worked by
inflating everything, including the volume component, which is why it cost CRPS.

Coverage also varies far more by fold (0.899, 0.903, 0.950) than by any
configuration tested here, which is worth remembering before reading a pooled
number as a property of the architecture.

## Step 6 — S5, S7, S8

**S7 (fixed).** `vacated_opportunity` counted a player who stayed but recorded
nothing as departed, because `season_usage` is built from stat rows. That
inflates vacated opportunity with exactly the injury events the model exists to
predict. Roster membership now separates the two, and degrades to the old
behaviour where a source cannot supply it. The membership test is vectorised,
which also fixes a `KeyError` on an empty current season.

**S8 (fixed, metric-neutral).** Passing and receiving yards were built by
rescaling the rate by a clipped completion or catch probability and multiplying
by the event count. That preserved the mean but added roughly 120 yards of
standard deviation per efficiency draw that no part of the model estimates — the
efficiency posterior already carries per-opportunity noise through its
`sqrt(season_sigma^2 + opportunity_sigma^2 / exposure)` scale. Rushing already
used the matched form, so the three pathways disagreed about what a rate means.

On the scoring gate the change is neutral: MAE +0.02% to +0.05%, CRPS ±0.00%,
95% coverage -0.001 to -0.003. It lands as a correctness fix — one arbitrary
noise source and three hardcoded clips removed — not as an improvement.

*(My review described this as yardage having no residual variance. That was
right for rushing and backwards for the other two.)*

**S5 (measured, not promoted).** Each efficiency response is fitted only on rows
clearing its `min_exposure` and then scored on every row: 57% of quarterback
rows, 58% of receiving rows and 82% of rushing rows sit below their own floor,
so the fitted mean describes high-usage players and is extrapolated onto a
majority that never entered the fit. Both likelihoods already downweight a thin
row correctly, so the hard floor is doing by exclusion what the likelihood does
by weighting, and pays for it by selecting on usage.

Lowering the floor to 5 improves point accuracy and costs a little calibration:

| metric | floor as specified | floor 5 | fold wins |
|---|---:|---:|---:|
| standard MAE | 33.1113 | 32.9778 (-0.40%) | 2/3 |
| PPR MAE | 43.4152 | 43.2787 (-0.31%) | 2/3 |
| standard CRPS | 24.1028 | 24.0806 (-0.09%) | 2/3 |
| PPR CRPS | 31.8708 | 31.8523 (-0.06%) | 1/3 |
| PPR 95% coverage | 0.918 | 0.912 | — |

MAE improves consistently, CRPS is a wash and misses the two-of-three rule on
PPR, and coverage slips about half a point while staying above the floor. That
does not clear the gate, so `exposure_floor` stays `None`. Searching for a
better floor value on these same holdouts would be selecting on the test set,
which is the process risk recorded as S9 in the review — the right next step is
an inner fold, not a sweep.
