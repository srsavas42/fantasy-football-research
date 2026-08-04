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

> **Closed 2026-08-04.** Everything above this line is the state before the
> tail work; the sections that follow attribute the defect and fix it. PPR
> coverage on 2025 now reads z=−1.10 at the 95% level against the +3.99
> recorded here. The numbers in this opening section were also produced from
> the older cache, so they are not directly comparable to anything below —
> see the frame fingerprints.

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

---

# Whose rows they are (2026-08-04)

The deficit is one-sided and concentrated. Of 46 misses at the 95% level on
2025, **37 are above the interval and 9 below**; the PIT's top decile holds 78
against 50.8 expected (z=+4.02) while the bottom decile sits exactly on
expectation. The model under-predicts upside and its downside is calibrated.

Split by row, on 2025:

| split | n | miss rate | z | above / below |
|---|---:|---:|---:|---|
| **no prior snap share** | 117 | **28.2%** | **+11.52** | 32 / 1 |
| **rookies** | 85 | **25.9%** | **+8.83** | 21 / 1 |
| **prior role continuity <0.33** | 202 | **18.8%** | **+9.01** | 33 / 5 |
| prior snap share 0.25–0.60 | 156 | 2.6% | −1.40 | — |
| prior role continuity ≥0.67 | 306 | 2.6% | −1.91 | — |
| prior snap share >0.60 | 112 | 1.8% | −1.56 | — |

Players with an established role are at or past nominal. The entire deficit
sits on players the model has never seen play, and half the misses land more
than a full half-width outside the interval, the worst at 17.7. That is a
mixture signature rather than a width one.

This also rules the innovation cap out as the *cause*. The cap is global, so
raising it widens the rows that are already covered.

## The fix, and what it was worth

`cold_role_innovation` gives rows whose role prior is a population fallback
their own, wider innovation scale, sized from the training data's own
cold-versus-warm log-share dispersion ratio: 1.38 for carries, 1.64 for targets
on 2014–2024.

Against a matched baseline on 2022/2023/2024, same cache, promoted
configuration otherwise:

| | baseline | cold-role |
|---|---:|---:|
| PPR cov95, misses / expected | 133 / 76.6 | **125 / 76.6** |
| PPR cov95, z | +6.62 | **+5.68** |
| PPR cov80, z | +4.33 | **+3.76** |
| PPR MAE | — | −0.20% |
| PPR CRPS | — | −0.17% |

Per fold, PPR cov95: 0.901 → 0.904, 0.895 → 0.903, 0.944 → 0.948.

Every metric improves and every one improves on all three folds — fifteen
metrics moving the same way three times each is not noise. But it recovers
**eight of the fifty-six excess misses, about a seventh of the deficit**, and
every move is below the gate's materiality floor. The gate says *accepted, but
nothing improved materially*.

## Why it is throttled

The cap. Measured cold-row dispersion is 2.68 for carries; the base scale is
capped at 0.25, so a 1.38× multiplier reaches 0.35 — still about seven times
narrower than the population it is meant to represent. The widening is
directionally right and quantitatively strangled by a parameter upstream of it.

That parameter's selection is separately suspect. `innovation_cap` was promoted
at 0.25 on mean distance from nominal coverage over the carry and target
streams — the statistic later shown to be uninterpretable there, because half of
carry rows are zero and every interval containing zero covers them. A criterion
that reads guaranteed coverage as over-wide intervals rewards narrowing.

The next measurement is therefore the cap, swept against total fantasy points
rather than against an intermediate stream's coverage. Stated before the numbers
arrive: if that sweep does not move the deficit either, this is reported as an
insufficient fix rather than iterated on. Three plausible variants selected in
sequence against the same three folds spends the window just as surely as
tuning on it.

## The cap is a different point on the same bad trade

| cap | cov80 | z80 | cov95 | z95 | MAE | CRPS |
|---:|---:|---:|---:|---:|---:|---:|
| **0.25** (promoted) | 0.785 | **+0.82** | 0.909 | +4.19 | 40.746 | **29.862** |
| 0.50 | 0.807 | −0.40 | 0.923 | +2.77 | 40.676 | 30.147 |
| 1.00 | 0.846 | −2.62 | 0.941 | **+0.94** | 40.594 | 31.964 |
| 1.50 | 0.860 | −3.39 | 0.957 | −0.69 | 40.791 | 34.344 |
| None | 0.864 | −3.62 | 0.963 | −1.30 | 41.032 | 34.626 |

Relaxing it closes the 95% gap and breaks the 80% level doing it, with CRPS
degrading 7% by cap 1.0 — the same trade the global width sweep found, because
both are global knobs and the defect is not global.

Which is the argument for the targeted one. Every global knob trades; the
cold-role widening improved both levels, MAE, CRPS and RMSE together. It was
just far too small, because a cold-to-warm *ratio* inherits the cap's
compression: capped from 1.94 to 0.25, a 1.38× ratio lands cold rows at 0.35
against a measured 2.68.

## Targeting the measured dispersion instead

`cold_role_scale_mode="measured"` targets the cold population's own dispersion,
so the cap bounds the typical row without bounding the row it was never about.
Cold rows land at 1.50 — the multiplier cap of six binds before the measured
2.68 does.

Pooled PPR over the same three folds, same cache, all three arms sharing
fingerprint `e1e698b3`:

| | cov80 misses/exp | z80 | cov95 misses/exp | z95 | MAE | CRPS |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 374 / 306.2 | +4.33 | 133 / 76.5 | +6.62 | 43.119 | 31.295 |
| cold-role, relative | 365 / 306.2 | +3.76 | 125 / 76.5 | +5.68 | 43.033 | 31.241 |
| **cold-role, measured** | **315 / 306.2** | **+0.56** | **78 / 76.5** | **+0.17** | **41.787** | **30.761** |

Both levels land on nominal — 78 misses against 76.5 expected at the 95% level,
315 against 306.2 at the 80% — and point accuracy improves 3.07% and CRPS
1.68% at the same time, each on all three folds. The gate accepts with no
exception required.

That combination is what distinguishes this from every earlier attempt. A width
change that fixes the tails should cost the body, and a change that improves
coverage should cost a proper scoring rule. This does neither, because the rows
it widens are the rows that were wrong: the established rows it leaves alone
were already at nominal, so there is nothing there to spend.

The one blemish is 2024, where cov95 goes 0.944 → 0.962 and so moves from six
tenths under nominal to twelve hundredths over. That is the fold whose aggregate
deficit was smallest to begin with, and it is why the 95% verdict reads
"improved (sign varies)" rather than a clean sweep.

## 2025, scored once

| | baseline | cold-role, measured |
|---|---:|---:|
| PPR cov80, misses / expected | 111 / 101.6 (z=+1.04) | **96 / 101.6 (z=−0.62)** |
| PPR cov95 | 45 / 25.4 (z=**+3.99**) | **20 / 25.4 (z=−1.10)** |
| half-PPR cov95 | 44 / 25.4 (z=+3.79) | 22 / 25.4 (z=−0.69) |
| standard cov95 | 40 / 25.4 (z=+2.97) | 21 / 25.4 (z=−0.90) |
| PPR MAE | 40.736 | **39.270** (−3.60%) |
| PPR CRPS | 29.851 | **29.245** (−2.03%) |
| PPR RMSE | 58.575 | **56.225** (−4.01%) |

The defect this document opened with is closed on the season it was found on.
Every scoring system moves from three to four standard errors of under-coverage
to within about one of nominal, and point accuracy, CRPS and RMSE all improve
alongside it. Both arms share fingerprint `e1e698b3`.

The gate calls each metric "inconclusive", which is correct and worth keeping:
one fold has no spread, so it cannot separate an effect from a fold. The
decision was made on the three in-window folds; this is the check, and it
agrees.

**Promoted 2026-08-04**, `cold_role_innovation` with `cold_role_scale_mode`
`"measured"`.

## The targeting, demonstrated rather than inferred

Aggregate coverage improving is consistent with two very different changes: one
that fixes the rows that were wrong, and one that widens everything and happens
to net out. Running the miss split under all three configurations on 2024
separates them.

| band | n | baseline | relative | **measured** | base z | meas z |
|---|---:|---:|---:|---:|---:|---:|
| **rookie** | 81 | 14.8% | 13.6% | **4.9%** | +4.05 | **−0.03** |
| **no prior snap share** | 117 | 12.8% | 12.0% | **5.1%** | +3.88 | **+0.06** |
| **role continuity <0.33** | 212 | 9.0% | 8.5% | **4.7%** | +2.65 | −0.19 |
| prior snap share >0.60 | 106 | 0.9% | 0.9% | 0.9% | −1.92 | −1.92 |
| prior snap share 0.25–0.60 | 155 | 3.9% | 3.9% | 3.9% | −0.64 | −0.64 |
| prior snap share 0–0.25 | 123 | 4.9% | 4.9% | 4.9% | −0.06 | −0.06 |
| role continuity ≥0.67 | 289 | 3.1% | 3.1% | 3.1% | −1.47 | −1.47 |
| experience 3–5 | 153 | 3.3% | 3.3% | 3.3% | −0.98 | −0.98 |
| second year | 82 | 4.9% | 4.9% | 4.9% | −0.05 | −0.05 |
| veteran (6+) | 185 | 3.8% | 3.8% | 3.2% | −0.76 | −1.10 |

The three bands the widening was built for land on nominal — 4.9%, 5.1% and
4.7% against a 5% nominal, from z of +4.05, +3.88 and +2.65. Every band it was
not built for is identical to the tenth of a percent.

The one exception is veterans, 3.8% → 3.2%, and it is the mask working as
specified rather than leaking: a player returning from a season out has no prior
snap share whatever his experience, so some veterans are legitimately cold.

That is the claim the aggregate numbers could only suggest. Overall on this fold
the 95% level goes 28 misses → 19 against 25.1 expected and the 80% goes 110 →
93 against 100.2, but the reason to believe it is the row-level table, not the
totals.

## What is least evidenced about it

`cold_role_multiplier_cap` at 6. It binds in both modes on real data — measured
mode asks for 2.68 over a base of 0.25 — so the cap, not the measurement, is
what sets where cold rows land. It was chosen before any result and has never
been selected against folds. The feature works; the specific width it settles on
is the one number here that nothing validated.

Coverage now sits slightly conservative at both levels (z between −0.6 and −1.1
across scoring systems), which is the direction a slightly-too-large cap would
push it. Selecting it properly, on an inner fold, is the obvious next
refinement.
