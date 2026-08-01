# Total-season scoring: where coverage actually fails

Scoring v1 was rejected on coverage
([season-scoring-v1-validation.md](season-scoring-v1-validation.md)): 95%
coverage sat at 0.879-0.892 against a 0.90 promotion floor while CRPS stayed
flat. Four repairs followed, and every one was a global transform — dispersion
scaling from 0.75x to 1.50x, total-point shrink and expansion, an empirical
Gaussian copula, and a draw-conditioned volume-to-efficiency handoff. All were
rejected, and the conclusion drawn was that simple marginal rescaling cannot
jointly repair sharpness and coverage.

Each of those repairs assumes the miscalibration is uniform. That assumption
was never tested: the validation reports a pooled number per scoring format and
never decomposes it. This note tests it.

## Method

A refit rather than a reproduction. The published artifacts live under
`.cache/`, which is git-ignored and therefore absent from a fresh clone, so the
frozen volume-v2 posterior and the v3 component replacements are not available
to score against. Instead the current pipeline is fit on 2015-2023 and used to
project 2024 from a real week-1 roster snapshot, and coverage is decomposed by
position and by projected size.

```powershell
python scripts/diagnose_scoring_coverage.py --holdout 2024 --scoring ppr
```

This is one holdout at 300 draws over two chains, not the three-fold panel at
production sampling. Treat the levels as indicative and the *shape* as the
result.

## First, the evaluation was dropping the failures

An inner join from projections to realized stat rows discards every projected
player who never recorded one. Those are not missing observations. A player
projected onto a roster who produced nothing produced zero, and that zero is the
outcome the projection got wrong — so excluding it grades the model only on
players whose role materialised, which flatters precisely the behaviour a
preseason projection most needs to get right.

For 2024 that is 86 of 569 real projected players, 15% of the population, spread
across every position: 35 receivers, 20 tight ends, 17 backs, 14 quarterbacks.
Scoring them as zeros rather than dropping them:

| Alignment | n | 80% coverage | 95% coverage | misses below |
| --- | ---: | ---: | ---: | ---: |
| inner join | 483 | 0.795 | 0.954 | 7 |
| every real projection | 569 | 0.798 | 0.947 | 15 |

Misses below the interval double. Quarterback moves from 0.900 to **0.893**,
which is to say from sitting exactly on the 95% floor to failing it. The rate of
absent players is similar across positions, but the cost is not: a projection of
83 points for a backup quarterback who never plays is a far larger error than a
small projection for a fringe receiver.

`align_projection_to_outcomes` in `ffmodel.evaluation.holdout_alignment` performs
this alignment, excluding only the synthetic replacement buckets, which are a
modelling device rather than people and have no realized counterpart. Every
figure below uses it.

## Result: the miscalibration is not uniform

PPR, 2024, n=569 real projected players. Pooled coverage is 0.798 at the 80%
level and 0.947 at the 95% level — both inside the gate. Decomposed:

| Position | n | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: |
| QB | 84 | **0.679** | **0.893** |
| RB | 140 | 0.850 | 0.971 |
| TE | 123 | 0.829 | 0.967 |
| WR | 222 | 0.793 | 0.941 |

The gate requires 0.70-0.90 at the 80% level and 0.90-0.99 at the 95% level.
Quarterbacks fail both. Every other position is comfortably inside both.

By projected points, coverage rises monotonically with size:

| Quartile | n | median projection | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 121 | 1.6 | 0.727 | 0.901 |
| Q2 | 121 | 25.2 | 0.760 | 0.950 |
| Q3 | 120 | 70.4 | 0.800 | 0.975 |
| Q4 | 121 | 176.2 | 0.893 | 0.992 |

Small projections are under-covered and large ones are over-covered. At 0.992
the top quartile is wider than it needs to be.

That combination explains the earlier results directly. Widening globally
improves the under-covered low tail while making an already over-wide top
quartile worse, and the top quartile carries most of the fantasy points, so CRPS
degrades. Narrowing does the reverse. The observed trade-off is not a property
of the distribution family; it is the consequence of applying one multiplier to
two groups that need opposite corrections.

## The misses name a mechanism

Of 30 observations outside the 95% interval, 15 fall above and 15 below. The
overshoots are dominated by one recognisable case:

| Player | Position | Projected | Actual |
| --- | --- | ---: | ---: |
| Sam Howell | QB | 83.4 | -0.8 |
| Jake Browning | QB | 79.0 | -0.2 |
| Skyy Moore | WR | 50.8 | 0.0 |
| Tyson Bagent | QB | 34.1 | -0.3 |
| Clayton Tune | QB | 21.0 | -2.1 |

These are backup quarterbacks given a plausible share of a season's passing
work who then never played. The model is not mildly over-confident about their
rate statistics; it is missing an outcome in which the projected role does not
happen at all. A quarterback room is close to winner-take-all, which is why the
effect concentrates there: a back or receiver who loses a job still absorbs some
volume, while a quarterback who loses one absorbs almost none.

## The backup quarterback distribution, and the missing hurdle

The realized shares say the quarterback room is not one population. Splitting
2016-2023 quarterback seasons by listed week-1 depth:

| Listed depth | n | share below 0.02 | median | p90 | skew |
| --- | ---: | ---: | ---: | ---: | ---: |
| QB1 | 253 | 0.01 | 0.901 | 0.997 | -1.33 |
| QB2 or lower | 624 | 0.62 | 0.002 | 0.291 | +2.77 |
| unlisted | 80 | 0.82 | 0.000 | 0.125 | +4.01 |

The shape inverts across the room. A listed starter is left-skewed against a
ceiling; a backup is right-skewed against a floor, taking nothing in 62% of
seasons and a substantial share when the starter goes down. No single unimodal
distribution fits both, and the marginal that results fits neither.

Conditioning on actually playing isolates the cause:

| Position | n with share >= 0.02 | mean | median | skew |
| --- | ---: | ---: | ---: | ---: |
| QB | 499 | 0.507 | 0.451 | **0.09** |
| RB | 917 | 0.227 | 0.165 | 0.88 |
| WR | 1267 | 0.116 | 0.100 | 0.71 |
| TE | 606 | 0.083 | 0.066 | 1.08 |

Given play, quarterback is the *most* symmetric position in the model — less
skewed than any of the others. The quarterback problem is therefore not the
shape of the distribution conditional on playing. It is entirely the zero/
non-zero mixture, and that mixture is the part the model does not represent.

`QBWorkloadShareModel` allocates the room with a softmax over a Multinomial
likelihood, with Gaussian innovation added to the linear predictor. A softmax
cannot emit an exact zero: every quarterback on the roster receives positive
share on every draw, and the innovation widens the allocation symmetrically in
log-odds space rather than adding mass at zero. A backup therefore gets a
unimodal distribution centred between the two outcomes that actually occur,
which explains both halves of the failure — the central projection is far too
high for the 62% who never play, and the 80% interval sits in a region the
realized distribution rarely visits.

Availability already uses a Bernoulli/Beta-Binomial hurdle for appearing, and
carries already use a draw-level any-carry hurdle whose eligibility samples zero
out the allocation before renormalising. Quarterback workload is the pathway
that never received the same treatment.

## Implication for the next challenger

The v1 note already proposes latent role states — `replacement`, `inactive`,
`committee`, `lead` — sampled jointly into availability, opportunity and
efficiency ([latent-regime-ablation.md](latent-regime-ablation.md)). This
diagnosis supports that direction and narrows it:

1. The defect is a missing role-collapse mode, not a mis-set spread, so the
   `inactive` state is the mechanism that matters and it must be *sampled*
   rather than blended into a mean. A post-hoc tilt has already been rejected
   for exactly this reason. The narrowest form of this is a quarterback
   workload hurdle: a per-draw Bernoulli gate deciding whether a quarterback
   takes meaningful snaps at all, with the existing softmax allocating among
   those who clear it and renormalising over the room. That is the same shape
   as the carry-eligibility hurdle already in the volume stack, which zeroes
   the linear predictor for ineligible rows before renormalising, so it needs
   no new machinery. It is also testable on its own before any broader shared
   regime state, and the earlier all-pathways regime screen specifically
   regressed target and QB-pass accuracy, which argues for changing one
   pathway rather than bundling.
2. It concentrates at quarterback and in small projections, so a candidate
   should be judged by decomposed coverage. A pooled number can pass while
   quarterbacks fail, and a global correction that fixes the pool will damage
   the top quartile.
3. Because the top quartile is already over-covered, a successful candidate
   should be expected to *narrow* there while lengthening the lower tail
   elsewhere. Any change that widens everything is going the wrong way.

`coverage_by_group` in `ffmodel.evaluation.metrics` reports coverage and
miss direction per group so future candidates can be gated on the decomposition
rather than the pooled figure alone.

## Caveats

- One holdout, one refit, reduced sampling. The pooled 95% coverage here (0.954)
  is well above the published v1 figure (0.879-0.892), so this is not a
  reproduction of that baseline and the two numbers are not comparable. The
  likely cause is the volume layer: this uses the current pipeline rather than
  the frozen volume-v2 plus v3 components the published run scored against.
- Points are scoped to the preseason team, so mid-season movers are penalised;
  see the Davante Adams case in the pull request that added forward projection.
- Extreme quantiles from 600 draws are noisy, which affects the 95% level more
  than the 80% level.
