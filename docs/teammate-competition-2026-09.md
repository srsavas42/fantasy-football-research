# Teammate competition in a player's history

*September 2026*

The question: a player's history is a list of weeks, and the weeks are not
comparable. Some of them he played with the man ahead of him on the depth chart
inactive. Those are his best weeks, they enter every average at full weight, and
when the teammate comes back the average is still carrying them.

That story is true, and measurable, and the model already handles it. The season
layer is the one place it might not, and there the evidence is a coin flip. This
is the record of both.

## The contamination is real

Skill players 2016-2025, weeks played, at least four prior games. Within a
player — so this is not a comparison between players who get such weeks and
players who do not — a week in which somebody listed ahead of him at his position
is ruled out is worth **+1.25 points** (857 players seen both ways, t = +7.99).

The average then carries it. Residual against
`prior_points_recent_given_played`, restricted to weeks in which the man ahead is
back on the field:

| week | n | points − prior_points_recent |
|---|---:|---:|
| no absence in the last two | 38,263 | −0.098 |
| he was absent one week ago | 1,451 | **−0.602** |
| he was absent two weeks ago | 1,179 | **−0.706** |

Half a point high, and the second week is no better than the first: a half-life
of four games still has most of its weight inside the contaminated window. By
position, against each position's own baseline: RB **−0.86**, TE −0.62, WR −0.43,
worst where the work is most zero-sum. Backfield carries are handed from one
specific man to one specific other man; targets spread across a route tree.

## The repair, and why it is a weight rather than a feature

`src/ffmodel/weekly/competition.py` down-weights such weeks by a factor `kappa`
when the averages are formed. `_prior` gained a `weights` argument on its
exponential path, implemented through the identity

```
ewm(w · x) / ewm(w)  =  Σᵢ (1−a)^(t−i) wᵢ xᵢ  /  Σᵢ (1−a)^(t−i) wᵢ
```

which is the weighted average exactly rather than an approximation of it. Two
properties of that form are load-bearing. Only *relative* weights survive, so a
pure handcuff back — every one of whose weeks came with the starter hurt — is
returned unchanged, and the adjustment reaches only players with a mix of both,
which is the population the table above describes. And the numerator and
denominator must skip the same rows, which is why the mask is taken from the
values and not from the weights; otherwise an unplayed week spends denominator
and every average is biased toward zero. `tests/test_weekly_competition.py`
pins all of it, including a brute-force check against the definition written out
one row at a time.

The alternative was another covariate saying "his room was depleted".
`docs/target-competition-2026-09.md` is the record of that failing: a
room-structure feature offered to the target allocator collided with an offset
the model already carried and cost 4.63% of target MAE. A weight adds no degree
of freedom, so it cannot double-count.

## The weekly arm is a null

`scripts/sweep_competition_weight.py`, inner window 2021/2022, population fixed
once at the unweighted arm (`relevant_population` reads one of the averages under
test, so letting it move would score each candidate on different rows):

| kappa | MAE | CRPS |
|---:|---:|---:|
| 0.00 | +0.05% | +0.28% |
| 0.25 | −0.08% | +0.01% |
| 0.50 | **−0.10%** | **−0.08%** |
| 0.75 | +0.04% | +0.04% |
| 1.00 | — | — |

A tenth of a percent, on a curve that is not monotone. That is noise, and the
diagnostic says why it has to be.

**The model already repairs this.** It carries `ahead_out`, `ahead_out_lagged`
and `depth_promoted_lagged`, and it spends them. The same split measured against
the *fitted prediction* rather than against the raw average:

| week | n | points − prediction |
|---|---:|---:|
| no absence in the last two | 7,914 | +0.135 (se 0.074) |
| he was absent one week ago | 220 | +0.106 (se 0.382) |
| he was absent two weeks ago | 170 | −0.280 (se 0.446) |
| the man ahead is out now | 372 | +0.384 (se 0.324) |

Every cell inside one standard error of the baseline. The bias visible in the
feature is gone by the time it is a prediction.

And the ceiling was never large. The affected rows are **4.0%** of the scored
population, and they are *easier* than average (MAE 4.17 against 4.71), so
removing their residual bias perfectly is worth **at most 0.05%** of pooled MAE.
The sweep's ±0.1% wobble is the noise floor around a zero, not a signal.

This is the correction worth stating plainly: the measurement that opened this
investigation was taken against a raw feature and read as a defect in the model.
It is a defect in the feature that the model routes around. `DEFAULT_KAPPA` is
1.0 — the identity — and the code stays for the season layer.

## The season layer: carries maybe, targets no

The season layer has no escape hatch. Its prior-season shares are plain sums over
weeks (`ffmodel.features.crossseason.season_usage`) and nothing in the preseason
feature set names which weeks were played against a depleted room. So the same
weights were screened there by `scripts/screen_season_competition.py`, which
builds both forms of the prior-season share —

```
raw       Σ xᵢ      / Σ Tᵢ
adjusted  Σ wᵢ xᵢ   / Σ wᵢ Tᵢ
```

— and scores each against what the player actually did the following season. The
adjusted form is a weighted average of the same weekly shares, so it is still a
share, and it collapses to the raw one under uniform weights.

Restricted to the rows the weights actually move (~85% of rows have no boosted
week and cannot change), MAE against the next season:

| kappa | target share | carry share |
|---:|---:|---:|
| 0.75 | −0.09% | −0.47% |
| 0.50 | +0.02% | **−0.92%** |
| 0.25 | +0.56% | **−1.35%** |
| 0.00 | +2.16% | −0.91% |

Carries improve, with an interior optimum, which is a better shape than a
monotone edge. Targets do not.

### The control that matters

The adjusted share is mechanically smaller for an affected player — he is being
down-weighted on his biggest weeks — and shrinking any prior-season share toward
the mean improves MAE against the next season purely through regression to the
mean. So each arm was compared against a plain shrink toward the population mean,
scaled to move the same average distance:

| share | population | competition | matched shrink |
|---|---|---:|---:|
| target | weights bite | +0.02% | **−0.79%** |
| carry | weights bite | **−0.92%** | +0.62% |

*(kappa 0.5; at 0.25 the gap is wider still — carry −1.35% against +1.83%.)*

This splits the two cleanly. For **targets**, the entire apparent story is
shrinkage, and plain shrinkage does it better — competition weighting is a worse
way to shrink than shrinking. For **carries**, moving the same distance uniformly
makes things *worse*, so the weighting is placing individual backs correctly
rather than pulling everyone down. The direction is real information.

### It still does not clear the gate

| prior season | n | delta (kappa 0.25) |
|---:|---:|---:|
| 2016 | 52 | −2.37% |
| 2017 | 41 | +3.45% |
| 2018 | 44 | −3.32% |
| 2019 | 54 | −7.50% |
| 2020 | 74 | +3.42% |
| 2021 | 69 | −2.17% |
| 2022 | 52 | +0.19% |
| 2023 | 62 | +1.62% |
| 2024 | 65 | −6.27% |
| **pooled** | **513** | **−1.35%**, folds better **5/9** |

Five of nine, on folds of forty to seventy rows, with the pooled number carried
by three of them. The house standard elsewhere in these docs is unanimity or
close to it. A pooled 1.35% that is 5/9 is a direction, not a result.

The binding constraint is population, not feature design: there are only ~500
player-seasons in a decade where a back holds a real mix of both kinds of week.
No refinement of the weight fixes that, so none was attempted — the honest move
is to stop rather than to tune until a fold flips.

## What shipped

Nothing changed in any production projection. `DEFAULT_KAPPA` is 1.0, no caller
passes anything else, and `add_features` collapses a uniform weight to the
unweighted path so the baseline stays bit-identical rather than merely equal.

What is now in the tree is the machinery and the measurement: the weighted-EWMA
identity with its tests, the two screens, and this record. The carry-share result
is the one thing here worth revisiting — if the panel ever reaches back far
enough to double that 513, it is a real question again.
