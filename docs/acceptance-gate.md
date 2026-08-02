# The acceptance gate

`python scripts/compare_validation_runs.py <baseline.json> <candidate.json>`

Exits non-zero when the candidate is not acceptable. Implementation in
`src/ffmodel/evaluation/acceptance.py`, tests in `tests/test_acceptance_gate.py`.

Until now this gate was a convention: read the two JSON files, tabulate the
metrics you care about, promote if the candidate wins two holdouts of three.
Three things were wrong with that, and each one had already cost something.

## It only watched what somebody tabulated

The team model's R-hat of 1.0177 was in the diagnostics block of every
validation run for weeks. It never appeared in a comparison table, because no
table had a column for it, and it was eventually found by reading raw JSON
rather than by the gate. A gate that depends on remembering to look is not a
gate.

The comparator now enumerates every `(stream, metric)` pair present in both
runs and every component in both `diagnostics` blocks. A metric the baseline
reports and the candidate does not is itself a blocker — silently narrowing the
comparison is the exact failure this is meant to prevent.

## It counted fold wins as if the folds were exchangeable

They are not. The 2023 holdout trains on a history ending in the second-largest
year-over-year pass-rate move in the window (−0.0132 across 2022) and then
scores a season that partially reverts. Treating it as one independent vote out
of three is treating a regime shift as a coin flip.

With three folds there is no honest standard error, so the gate does not pretend
to one. It reports the pooled change against the fold-to-fold spread, and a
metric earns `improved` or `regressed` only when the pooled move exceeds that
spread. A change that is −4% on two folds and +9% on the third is
`inconclusive`, which is what it is. Under the old rule it was a promotion.

`inconclusive` does not block. It also does not sell.

## It gated R-hat at a bright line inside R-hat's own noise

The team model's residual 1.0107 was chased as a convergence defect before a
reseed showed the statistic itself moves: seed 42 gives 1.01, seed 7 gives 1.00
on the same model and the same data, and both settle at 1.00 by 2000 draws. A
threshold at 1.01 is therefore a coin flip on the seed.

Divergences are gated hard, because a divergence is a real event rather than an
estimate. R-hat is gated at 1.02, and anything in the 1.01–1.02 band is reported
as needing a reseed rather than failed. Bulk ESS is gated at 400.

## Two things the gate learned about itself

Both surfaced the first time it ran against a promotion already accepted by
hand, which it initially rejected on three counts. Two of the three were the
gate's fault.

**Coverage cannot be scored as a ratio.** Coverage is judged by distance from
nominal, and that distance is routinely near zero — which makes the relative
change a divide-by-almost-zero. A 95% interval moving from 0.960 to 0.966 is six
tenths of a coverage point; as a ratio it reads +62%, and the gate blocked on it.
Coverage is now measured in coverage points, with a materiality floor of one
point.

**A verdict needs a materiality floor as well as a spread test.** A pass-stream
MAE move of +0.06% was consistent across all three folds and comfortably larger
than its 0.03% spread, so it qualified as a regression — while being far too
small to act on. Non-coverage metrics now need a pooled move of at least 0.25%
before the gate will rule on them at all.

The third rejection was correct and stands as a `watch`: the team model's R-hat
of 1.0107 on the 2023 fold.

## Protected streams

`--protected` names streams that may not be damaged in exchange for gains
elsewhere, defaulting to `pass_qb` and `qb_workload`, with
`--protected-tolerance` at 0.5% relative. This is the one place the gate is
deliberately asymmetric: those two streams drive the highest-scoring position in
every scoring system the pipeline supports, so a change that buys carry accuracy
by spending quarterback accuracy is not a trade the gate will make silently.
