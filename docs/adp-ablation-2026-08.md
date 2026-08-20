# Preseason ADP as a covariate (2026-08-19)

Every feature in this pipeline is a transform of something that happened on a
field. Average draft position is not. It aggregates what drafters believed
before a season started — camp bodies, contracts, holdouts, a new coordinator's
stated plan — none of which reaches a box score until the season it describes is
over, and it is published before the season it forecasts. It is the first
genuinely outside source the model can read.

Implementation in `ffmodel.features.market`, gated by
`SeasonAverageVolumePipeline.market_adp_features`. Off by default; see
[Recommendation](#recommendation).

## What was measured

Three columns — `adp_log_rank`, `adp_position_log_rank`, `adp_drafted` — added
to the four layers that decide who holds a role: snap share, carry eligibility,
target allocation, carry allocation. Availability, the team layer and the
quarterback workload layer were deliberately left out, so a result here is
attributable to the role rooms rather than to everything at once.

Walk-forward over 2022–2024, both arms on `.cache/ffmodel-wf-2025-adp`.

### The controls

Three things had to hold before the numbers meant anything, and all three were
checked rather than assumed.

**The candidate actually read the feature.** Each fold records the feature names
every submodel put in its design matrix. The candidate carries all three ADP
columns in all four rooms; the baseline carries none. A flag sets a name on a
list and `_matrix` drops names it cannot find, and twice on this branch a clean
`+0.00%` null turned out to be that gap rather than a result.

**The cache is inert.** The augmented frames were produced by copying the
existing ones and adding columns, not by rebuilding — two caches here once
differed in 69 of 289 columns with identical row counts. The baseline arm
reproduces `scoring_coldmeasured.json`, the shipping configuration, with **no
metric differing in any fold or scoring format**. Whatever the candidate does is
the feature.

**Sampler health did not move.** Zero divergences in both arms, minimum bulk ESS
621. The one watch item — `volume/team` R-hat 1.0107 at 2023 — is **1.0107 in
both arms**: the pre-existing statistic the gate documents as noise on the seed,
not something this feature introduced.

## Result: accepted

`compare_validation_runs.py` exits 0. Every accuracy metric improves, 3/3 folds,
across all three scoring formats and both populations. No metric regresses.

| population | Δ MAE | Δ RMSE | Δ CRPS |
|---|---:|---:|---:|
| all rostered | −2.27% | −1.99% | −2.49% |
| ADP drafted | −0.73% | −1.39% | **−2.75%** |
| undrafted (derived) | **−5.02%** | — | −2.00% |

The undrafted row is recovered arithmetically, not refitted: MAE, CRPS and both
coverages are means over rows, so the complement follows exactly from the two
recorded groups and their counts.

## What the feature is actually doing

Two different things to two different populations, and reporting only the pooled
number would have hidden both.

**On undrafted players it moves the mean.** MAE −5.02% against CRPS −2.00%. ADP
tells the model which players will not matter, and their point projections come
down. This is most of the pooled MAE gain — the pooled figure is three times the
drafted-pool figure, which is arithmetically impossible unless the complement is
carrying it.

**On drafted players it moves the spread, not the location.** The gains order
themselves by how much each metric weights the distribution:

| metric | drafted-pool gain |
|---|---:|
| MAE (centre only) | −0.73% |
| RMSE (weights large errors) | −1.39% |
| CRPS (whole distribution) | −2.75% |

A projection whose mean barely moves while its CRPS falls 2.75% has not learned
where to point. It has learned how confident to be — narrower for a player the
market ranks eighth, wider for one it declines to rank at all. That is a real
improvement on exactly the players who get drafted, and it is invisible to MAE.

## The one place it is worse

Drafted-pool 80% coverage rises in all three folds, and the interval was already
over-covering:

| holdout | base | candidate | distance from 0.80 |
|---|---:|---:|---:|
| 2022 | 0.848 | 0.852 | 0.048 → 0.052 |
| 2023 | 0.821 | 0.830 | 0.021 → 0.030 |
| 2024 | 0.830 | 0.860 | 0.030 → 0.060 |

The gate calls this negligible and it is — under two coverage points, against a
95% interval that stays put. But it is consistent in sign across three folds and
it moves the wrong way, so it belongs in the record rather than in the rounding.

## What the result does not say

**It is not evidence the model beats the market. The model loses to the
market.** See [the next section](#the-model-does-not-beat-the-draft-board),
which was measured after this ablation and is the more important result.

**It has not been confirmed on 2025.** The selection was made on 2022–2024, so
2025 remains untouched and can still serve as the one honest confirmation. That
run has not been done.

## The model does not beat the draft board

Measured by `scripts/benchmark_adp_only.py` after the ablation, on the same
drafted-pool rows the walk-forward scored, pooled over 2022–2024:

| projection | MAE | RMSE | CRPS |
|---|---:|---:|---:|
| **ADP alone, per-position curves** | **54.48** | **68.24** | **38.22** |
| ADP alone, one pooled curve | 60.73 | 75.63 | 42.48 |
| model, no ADP | 59.33 | 74.53 | 44.96 |
| model, with ADP | 58.90 | 73.49 | 43.72 |

The ADP-only estimator is a per-position log fit of points on draft rank, with
predictive draws resampled from the curve's own residuals at nearby ranks, all
fitted on seasons strictly before each holdout. It is a lookup table with error
bars.

It beats the model by **8.1% MAE and 14.4% CRPS**.

The curve is doing real work rather than exploiting the scoring. Replacing rank
with no information at all — resampling prior seasons' drafted-pool outcomes —
gives MAE 75.24 and CRPS 51.86, so the rank signal is worth about 21 points of
MAE and the model captures only part of it.

### Correcting an earlier number

`benchmark_adp_baselines.py` reported the model beating ADP by 6.5% on 2025.
That comparison was wrong in two ways, both of which flattered the model.

| 2025, drafted pool | n | MAE | CRPS |
|---|---:|---:|---:|
| model (no ADP feature) | 238 | 55.29 | 42.40 |
| ADP, one pooled curve | 237 | 60.26 | **41.84** |
| ADP, per-position curves | 237 | **55.36** | **38.52** |

The baseline did not know position, so it could not tell a quarterback taken
40th overall from a running back taken 40th — a distinction any drafter reads
straight off the board. And it was scored on point metrics only, so nobody
computed its CRPS. Against a position-aware curve the model ties on MAE and
loses CRPS by 10%; against even the weak curve it was already losing CRPS.

### What this means, and what it does not

The uncomfortable reading is the right one: **on the players people draft, this
model is not beating consensus.** Adding ADP as a covariate narrows the gap from
8.9% to 8.1% MAE, which is progress in the right direction and nowhere near
enough. The model now has the better signal in its design matrix and is still
turning it into a worse forecast.

The obvious explanation — that the feature is being shrunk on its way through
four submodels under a shared prior — was written here first and is **wrong**.
It was measured afterwards and refuted: the ADP columns keep 113% of their
unregularized magnitude while everything else keeps 58%. See
[Following the three leads](#following-the-three-leads-2026-08-20).

Three things the comparison does not establish, stated so they are not used to
explain the result away:

- **The baseline's training distribution is better matched.** The curve is
  fitted on drafted players who appeared in the frames; the model is fitted on
  every rostered player, most of whom are fringe. Both are scored on identical
  rows, so the evaluation is fair, but the curve is specialised to the
  population it is tested on and the model is not.
- **The curve cannot do anything else.** It has nothing to say about a player
  the market did not rank, and the undrafted rows are where the ADP feature
  helped the model most (MAE −5.02%). It carries no team structure, no
  volume-and-efficiency decomposition, and no way to respond to news after the
  board is published.
- **Beating realised points is not the only question.** A drafter wants to know
  who is mispriced relative to ADP, and a projection that reproduces ADP
  perfectly would score well here while being useless for that.

None of which changes the headline. The next question for this package is not
another feature; it is why a model with ADP in it does worse than ADP.

### Where the gap actually comes from

`scripts/decompose_adp_gap.py` builds a ladder, each rung adding one thing, all
fitted on prior seasons and scored on the same rows.

| rung | MAE | CRPS |
|---|---:|---:|
| 1. intercept only | 75.31 | 51.83 |
| 2. one curve, no position | 60.54 | 42.41 |
| 3. shared slope, position intercepts | 55.58 | 38.88 |
| 4. per-position slopes | 54.48 | 38.22 |
| 5. per-position, no functional form | 54.88 | 38.66 |
| 3b. rung 3 fitted on all rostered | 57.56 | 42.52 |
| model, with ADP | 58.90 | 43.72 |

Three things fall out, and two of them contradict the obvious hypotheses.

**Nonlinearity is not what is missing.** Rung 5 replaces the log curve with a
local mean over nearby ranks — no functional form at all, free to bend however
the data likes — and it scores *worse* than the log fit (54.88 against 54.48).
There is no hidden nonlinear structure in the rank-to-points relationship. A log
curve is the right shape, and a more flexible learner would be fitting noise.

**The position interaction is real but minor.** Rung 3 to rung 4 is 1.09 MAE,
a quarter of the model's deficit. Worth adding; not the explanation.

**Most of the apparent gap was the comparison, not the model.** Rung 3b refits
rung 3 on every rostered player — the mixture the model is actually trained on,
undrafted rows included — and its edge collapses: 55.58 to 57.56 MAE, and 38.88
to 42.52 CRPS. Training on a fringe-heavy population costs 3.6 CRPS points on
drafted players, almost all of the baseline's distributional advantage.

Against that fair comparison the model is **2.3% worse on MAE and 2.8% worse on
CRPS**, not 8.1% and 14.4%. The honest headline is smaller than the one above.

It is still a bad result. Rung 3b is four parameters — a log-rank slope and
three position dummies — fitted by least squares, and it beats a hierarchical
Bayesian pipeline carrying fifteen features of usage history. But it relocates
the problem: the model is not failing to represent the relationship, it is
diluting a signal it already has, and a large part of what looked like dilution
is the price of being trained on everybody. That points at population weighting
or a draft-status-aware calibration, not at a different model family.

## Recommendation

Promote, conditional on a single 2025 confirmation run, and describe it
accurately when reporting: the drafted-pool gain is a calibration improvement,
not a sharper point projection, and the headline MAE number is mostly about
players nobody drafts.

Whether to accept a market-following model at all is a product judgement the
gate cannot make. The metrics say yes; the cost is that the projection stops
being independent of consensus, and any future claim about beating ADP has to
be worded around that.

## Following the three leads (2026-08-20)

The gap decomposition left three candidate causes. All three were tried.

### Interaction terms: implemented, and they do not help

`market_adp_interactions` gives each position its own rank slope and its own
drafted effect — the terms a linear probe said would close the population gap
entirely on MAE (55.67 against a 55.68 drafted-only target). All nine ADP
columns verified present in all four rooms.

On the first holdout the pipeline moved the **wrong way**: drafted-pool MAE
+1.57% and CRPS +0.64% against plain ADP, and the fold took 1382s against 734s.
One fold, and an indicative rather than controlled comparison — the arms sit on
caches differing by the six new columns, and `adpon2`/`adpoff2` on the matching
cache are the real control. But the probe predicted a gain and the model
delivered a loss, which is the interesting direction.

### Attenuation: refuted

`scripts/measure_adp_attenuation.py` fits the snap model as shipped, reads the
implied per-feature coefficients back through the SVD projection, and compares
them against an unregularized least-squares fit of the same response on the same
rows, with the same nuisance structure.

| | median coefficient kept |
|---|---:|
| ADP columns (3) | **1.13x** |
| every other feature (15) | 0.58x |
| every other, excluding the collinear pair (13) | 0.59x |

The prior is binding on the ordinary features — they keep 58% of what free
least squares would give them — and **not binding on ADP at all**. The three ADP
columns come through at full strength or slightly amplified: `adp_drafted` 0.310
against 0.362, `adp_log_rank` −0.306 against −0.271, `adp_position_log_rank`
0.121 against 0.101.

Nothing is diluting the ADP signal. The hypothesis behind task 42 was wrong.

A note on reading this table: a root-mean-square comparison reports the non-ADP
features as 7% kept, which is an artifact. `depth_rank` and
`is_replacement_player` both separate backups from starters and are nearly
collinear, so free least squares hands them cancelling coefficients of −15.8 and
+16.2 whose difference is identified and whose levels are not. The median ratio
is the honest statistic; the magnitude ratio measures the collinearity.

Scope: one submodel, one fold. The carry and target allocators have not been
checked the same way.

### What is left: the target, not the features

By elimination and by evidence, the deficit is neither the prior nor the terms.
It is that these columns are being asked the wrong question.

ADP predicts **fantasy points** — that is what a draft board is a forecast of,
and it is what the rank curve regresses on directly. The pipeline never
regresses points on anything. It uses ADP to predict *snap share*, then
composes: snap share to role share to volume, multiplied by a separately fitted
efficiency posterior, aggregated to points. Every stage is individually
defensible and the composition is where the accuracy goes.

That reading was measured next, and it is **also wrong**. See below.

### Composition: refuted too

`scripts/test_composition_cost.py` holds the estimator, the information set and
the machinery fixed — a per-position log fit on draft rank with residuals
resampled from nearby ranks — and varies only the route to points.

| route to points | MAE | CRPS | bias | cov95 |
|---|---:|---:|---:|---:|
| direct: points on rank | 54.70 | 38.35 | −3.07 | 0.946 |
| composed, independent draws | 55.01 | 38.44 | +2.63 | 0.967 |
| composed, dependence kept | 54.90 | 38.42 | −2.26 | 0.953 |

Projecting opportunity and points-per-opportunity separately and multiplying
costs **0.6% MAE** drawing them independently and **0.4%** drawing both
residuals from the same training player. Essentially nothing.

The dependence arm was included because independent draws lose the covariance
term in `E[XY]` outright, which is a bias rather than a spread — and the bias
does flip, +2.63 against −2.26. But the residual correlation between opportunity
and rate is **−0.040**, so there is almost no covariance to lose, and the two
arms land in the same place.

Multiplying volume by efficiency is not what costs the model four points of MAE.

### Where that leaves it

Four explanations have now been tested and eliminated:

| hypothesis | verdict | evidence |
|---|---|---|
| the rank curve is nonlinear and the model is linear | no | rung 5 scores worse than the log fit |
| the model lacks a position-by-rank interaction | small | worth 1.09 MAE of a 4.41 deficit |
| the ADP coefficients are shrunk by the prior | no | 1.13x kept, against 0.58x for everything else |
| composing volume times efficiency is lossy | no | 0.4% |

What remains is the least exciting and most likely explanation: the pipeline's
component projections are simply less accurate than a rank curve's, and the
architecture is not the reason. Testing that needs the pipeline's own per-row
volume and efficiency predictions scored against realised volume and
efficiency — which needs a fit that saves them, and has not been done.

## Why the interaction backfired (2026-08-20)

The interaction arm lost on both scored holdouts — pooled MAE +0.78% and +0.82%,
drafted-pool MAE +1.57% and +1.37%, zero divergences. Consistent sign, consistent
magnitude. Two hypotheses for why, both tested, both wrong, and the third is the
answer.

**Not the target.** The obvious reading was that the probe optimised the wrong
thing: it regressed *points* on rank, while the submodel regresses *snap share*.
Running the identical ladder on the snap model's own response, its own rows and
its own filter says otherwise — the interaction helps snap share **more** than it
helps points.

| terms | logit snap share | season points |
|---|---:|---:|
| rank + position + drafted | 1.0612 | 42.81 |
| + position × rank | −2.28% | −1.91% |
| + drafted × position | **−4.11%** | −3.03% |

The terms are genuinely informative about exactly what the submodel is fitting.

**It is the prior, once the columns are collinear.** Re-running the attenuation
measurement with interactions enabled:

| | median coefficient kept |
|---|---:|
| ADP columns, no interactions (3) | **1.13x** |
| ADP columns, with interactions (9) | **0.25x** |
| every other feature (15) | 0.62x |

Adding the interactions collapses the whole ADP block. And it does not only
shrink the new columns — it damages the main effects that were previously
untouched: `adp_log_rank` falls from 1.13x to 0.46x, `adp_position_log_rank` from
1.20x to 0.28x.

The free coefficients say why. Unregularized least squares wants
`adp_log_rank` −0.574 with `_x_qb` −0.554, `_x_rb` +0.513, `_x_wr` +0.388: large
opposing values whose *differences* are identified and whose levels are not. This
is the same pathology `depth_rank` and `is_replacement_player` already exhibit at
−16.8 and +17.2. A shared `Normal(0, 0.35)` over every coefficient cannot supply
that, so it pulls the entire block toward zero and takes the working main effect
down with it.

So the sequence is: the terms are real, the encoding makes them collinear, the
prior cannot express collinear terms, and the net result is less ADP signal than
before the interaction was added.

Two candidate fixes, neither tested:

- **Re-encode.** Position-masked slopes with no shared main effect span the same
  space but parameterise it as four absolute slopes rather than a level plus
  three deviations. An isotropic prior is not invariant to that choice.
- **Widen the prior for the ADP block.** `feature_prior_scale` is one number for
  every coefficient, so this needs a per-block prior the submodels do not
  currently have.

Caveat on the numbers above: two chains at 400 draws, and the sampler warned on
R-hat. The block-level contrast (0.25x against 1.13x) is large enough to survive
that; individual coefficients from this run should not be quoted.

### The re-encoding does not rescue it, and the terms were never worth much

`scripts/probe_adp_encoding.py` compares the two parameterisations under ridge —
which *is* the Gaussian prior: the posterior mean of a Normal likelihood with a
`Normal(0, tau)` coefficient prior is the ridge solution at
`lambda = sigma^2 / tau^2` — sweeping lambda so the answer does not depend on
pinning the noise scale down. Scored on the snap model's own response, rows and
filter, **with the model's full feature set present**.

| lambda | no interaction | deviations | absolute |
|---:|---:|---:|---:|
| 0 | 0.9348 | 0.9352 | 0.9352 |
| 10 | 0.9483 | 0.9470 | **0.9448** |
| 100 | 0.9507 | 0.9479 | **0.9473** |
| 1000 | 0.9491 | **0.9464** | 0.9500 |
| best over lambda | **0.9348** | 0.9352 | 0.9352 |

At `lambda = 0` all three agree to four decimals, as they must — the encodings
span the same space. The divergence at moderate lambda is real and the absolute
encoding does handle the penalty better, which confirms the parameterisation
diagnosis. It does not matter, because **both interaction encodings are worse
than no interaction at every lambda**, including their own optima.

#### The earlier −4.11% was an artifact

The ladder that motivated all of this fitted `rank + position + drafted` and
nothing else. In that feature-poor setting the interaction is worth 4.11% on
logit snap share. With `prior_snap_share`, `prior_snap_share_3yr`, `depth_rank`,
`is_replacement_player`, `qb_listed_starter` and the rest of `SNAP_FEATURES`
present — the setting the submodel actually fits in — it is worth **+0.04%**,
which is to say nothing.

Whatever the market's positional structure knows about exposure, the usage
history already knows. The interaction was measured in a room the model does not
live in.

That is the honest cause. The collinearity finding above is real and reproducible
— the block does collapse from 1.13x to 0.25x — but it is the mechanism for a
term that had nothing to contribute in the first place. Both encodings are
recorded as tested and rejected; `market_adp_interactions` should stay off and
`ADP_INTERACTION_FEATURES` is dead weight unless something else motivates it.

**Method note for the next feature.** Two probes in this document reached
opposite conclusions about the same terms, and the difference was which other
features were in the design. A probe run against a subset of the model's inputs
measures the subset, not the model.

## A projection that beats the draft board (2026-08-20)

Blending the pipeline with the rank curve beats the curve out of sample on both
scorable holdouts, at weights chosen only from earlier holdouts:

| holdout | w(model) | model MAE | curve MAE | blend MAE | model CRPS | curve CRPS | blend CRPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | — | 57.62 | 49.07 | — | 43.75 | 34.54 | — |
| 2023 | 0.10 | 64.19 | 58.95 | **58.19** | 48.51 | 41.42 | **41.05** |
| 2024 | 0.15 | 54.75 | 55.98 | **55.01** | 41.67 | 39.04 | **38.39** |

Pooled over the two scored folds, paired-draw averaging gives **MAE −1.50% and
CRPS −1.28%** against the curve; mixing gives −1.39% and −1.49%. Both clear the
0.25% materiality floor several times over, and both improve on each fold
individually. 2022 is unscored on purpose: it has no earlier holdout to pick a
weight from, and an in-sample weight there would be the least meaningful number
in the table.

Worth noting separately: **on 2024 the model beats the curve outright**, 54.75
against 55.98. The fold-to-fold spread is large and the curve still wins the
other two.

### Why this works: the model knows which way the board is wrong

The board's error regressed on the model's disagreement with it:

    observed - curve = alpha + beta * (model - curve)

Pooled over the two test folds, **beta = +0.409 ± 0.071** (n=459). Roughly two
fifths of the model's contrarian opinion is correct — when it likes a player by
100 points over his ADP-implied total, he beats it by about 41. That is why a
blend works even though the model loses on absolute error, and it is the
quantity a drafter actually wants.

### Every subgroup story from the exploratory fold was noise

Four predictions were registered from 2022 and committed before 2023 and 2024
finished exporting. One survived. Three were falsified, and two of them
reversed:

| prediction (from 2022) | test folds | verdict |
|---|---|---|
| pooled slope positive: +0.195 | **+0.409 ± 0.071** | survives, larger |
| QB above the pool: +0.564 | +0.064 vs pool +0.409 | **falsified, reversed** |
| later board above early: +0.296 vs +0.059 | +0.286 vs **+0.513** | **falsified, reversed** |
| RB and TE carry none: +0.057, −0.035 | RB **+0.617 ± 0.138**, TE +0.243 | **falsified** |

Running backs went from the emptiest subgroup on 2022 to the richest on the test
folds. Quarterbacks went the other way. Without the pre-registration this
document would have recommended a quarterback-and-late-round mispricing signal,
confidently, and been precisely backwards.

### What is not established

- **Two scored folds.** The weight rule needs a third before anyone should trust
  its stability, and 2022's absence from the scored set is a real cost.
- **The weight rule is conservative.** It grid-searches CRPS on earlier
  holdouts, which picked 0.10 and 0.15 while the measured optimum is near 0.41,
  so the blend is under-using the model. Switching to a slope-based weight —
  which is the variance-optimal estimator, and better motivated a priori — would
  probably improve on this. It must be validated on 2025, untouched, rather than
  chosen here: the reason to prefer it is now contaminated by having seen that
  higher weights help.
- **The error correlation is rising**: 0.766, 0.795, 0.823. If that is a trend
  rather than noise, the blend's headroom is shrinking.
