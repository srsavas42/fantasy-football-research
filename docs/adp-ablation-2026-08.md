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

**It is not evidence the model beats the market.** ADP is a forecast, and a
model that reads it is partly following consensus. Before this feature, "the
model beats ADP by 6.5% on the drafted pool" (see
[out-of-sample-2025.md](out-of-sample-2025.md)) was a claim about independent
information. With it, the claim becomes "the two together beat the history
alone", which is a different and weaker statement about the model's own
contribution — though a more useful one for a drafter, who has access to both.

**It has not been confirmed on 2025.** The selection was made on 2022–2024, so
2025 remains untouched and can still serve as the one honest confirmation. That
run has not been done.

## Recommendation

Promote, conditional on a single 2025 confirmation run, and describe it
accurately when reporting: the drafted-pool gain is a calibration improvement,
not a sharper point projection, and the headline MAE number is mostly about
players nobody drafts.

Whether to accept a market-following model at all is a product judgement the
gate cannot make. The metrics say yes; the cost is that the projection stops
being independent of consensus, and any future claim about beating ADP has to
be worded around that.
