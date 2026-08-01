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

## Result: the miscalibration is not uniform

PPR, 2024, n=483 matched players. Pooled coverage is 0.795 at the 80% level and
0.954 at the 95% level — both inside the gate. Decomposed:

| Position | n | 80% coverage | 95% coverage |
| --- | ---: | ---: | ---: |
| QB | 70 | **0.657** | **0.900** |
| RB | 123 | 0.854 | 0.976 |
| TE | 103 | 0.825 | 0.971 |
| WR | 187 | 0.791 | 0.952 |

The gate requires 0.70-0.90 at the 80% level and 0.90-0.99 at the 95% level.
Quarterbacks fail the first outright and sit exactly on the boundary of the
second. Every other position is comfortably inside both.

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

Of 22 observations outside the 95% interval, 15 fall above and 7 below. The
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

## Implication for the next challenger

The v1 note already proposes latent role states — `replacement`, `inactive`,
`committee`, `lead` — sampled jointly into availability, opportunity and
efficiency ([latent-regime-ablation.md](latent-regime-ablation.md)). This
diagnosis supports that direction and narrows it:

1. The defect is a missing role-collapse mode, not a mis-set spread, so the
   `inactive` state is the mechanism that matters and it must be *sampled*
   rather than blended into a mean. A post-hoc tilt has already been rejected
   for exactly this reason.
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
- 483 of 697 projected players matched a realized row. Shares and points are
  scoped to the preseason team, so mid-season movers are dropped or penalised;
  see the Davante Adams case in the pull request that added forward projection.
- Extreme quantiles from 600 draws are noisy, which affects the 95% level more
  than the 80% level.
