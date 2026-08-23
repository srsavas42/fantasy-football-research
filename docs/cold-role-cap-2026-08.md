# The cold-role cap, selected (2026-08-04)

`cold_role_multiplier_cap` was the one number in the promoted cold-role widening
that nothing had validated. It binds on real data — measured mode asks for a
ratio of about 10.7 on carries and 7.4 on targets, and the cap is 6 — so the cap
rather than the measurement decides where cold rows land. It was chosen before
any result.

**It stays at 6.0.** Not because it is optimal, but because a proper nested
selection could not beat it, and the value that selection kept choosing is worse
on the seasons it was scored against.

## The criterion, fixed before the numbers

Mean CRPS over the three scoring formats on total fantasy points. CRPS is a
proper scoring rule, so it penalises over- and under-dispersion alike and can
select a width without needing a coverage statistic to behave — which matters
because the criterion that picked `innovation_cap` at 0.25 was coverage on
carry and target counts, half of carry rows are zero, and that criterion
rewarded narrowing. Coverage was reported here and did not select.

## Every inner fold preferred a larger cap

| fold (selected on) | 6.0 | 8.0 | uncapped | picked |
|---|---:|---:|---:|---|
| 2022 (on 2021) | 26.3291 | 26.2613 | **26.2592** | None |
| 2023 (on 2022) | 26.3426 | **26.2671** | 26.2676 | 8.0 |
| 2024 (on 2023) | 28.5401 | **28.5328** | 28.5778 | 8.0 |

Consistently, and by margins that looked material: 6.0 sat +0.266%, +0.288% and
+0.025% above the best candidate. Two of three folds picked 8.0; the third
picked uncapped by 0.008%, which is noise. The 2024 fold is the only one where
removing the cap entirely is *worse*, so the cap does have a job.

## None of it reproduced

| holdout | picked | incumbent | selected | delta | inc cov95 | sel cov95 |
|---|---|---:|---:|---:|---:|---:|
| 2022 | None | 26.3426 | 26.2676 | −0.28% | 0.941 | 0.958 |
| 2023 | 8.0 | 28.5401 | 28.5328 | −0.03% | 0.945 | 0.949 |
| 2024 | 8.0 | 26.3166 | 26.5387 | **+0.84%** | 0.962 | 0.966 |

Mean **+0.18%**: the selected caps are worse than the incumbent on average, and
the one fold where the difference is large goes against them. Coverage agrees on
the scored folds — incumbent cov95 averages 0.949 against nominal 0.95, the
selection 0.958.

The clearest single case is holdout 2023. Its inner fold measured 6.0 → 8.0 as
worth 0.288%. On the season actually scored, the same change delivered 0.026% —
an eleven-fold shrinkage. Holdout 2024 is worse: an inner gain of 0.025% became
a 0.84% loss.

## What this is an example of

A sweep run directly on 2022/2023/2024 would have found 8.0 beating 6.0 by
around 0.28% on every fold and reported a material improvement. That improvement
does not exist. It is the selection fold's own noise, and nesting is what
separates the two.

This is the second time in this package that a cap looked well-chosen and was
not: `innovation_cap` was selected on a criterion that could not work, and this
one was nearly selected on gains that did not survive. The value being defended
here is the one that was picked arbitrarily — it just happens to hold up.

## What was not settled

- **The margin is small in both directions.** Nothing here shows 6.0 is right,
  only that 8.0 and uncapped are not demonstrably better and are probably
  slightly worse. Any future change should clear the 0.25% floor on scored
  folds, which nothing did.
- **`--drop-seasons 2016` was not run.** It would only matter if the value were
  moving; see docs/data-quality-2026-08.md.
- **One sampler failure is in this run.** An efficiency `concentration` in the
  2022 inner fit took 600 seconds and finished with 304 divergences, a chain at
  maximum tree depth, R-hat above 1.01 and ESS under 100 per chain. Within a
  fold one posterior serves every candidate, so it biases that fold's whole
  table together rather than its ranking — but it is the fold that picked
  uncapped, which is a further reason to discount that pick. The scoring
  walk-forward and this selection now record per-component diagnostics; this run
  predates that, so the warning is transcribed here from the log.
- **The loop recomputes a fit it already has.** The inner fold of holdout H+1 is
  the same training window and the same scored season as the outer fold of
  holdout H — fold 2024's inner numbers (28.5401 / 28.5328) are identical to
  fold 2023's outer numbers. That is roughly 30 of the 90 minutes. A cache keyed
  on the training window would remove it.
