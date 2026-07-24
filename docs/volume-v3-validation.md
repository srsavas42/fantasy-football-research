# Season-average volume v3 validation

Validation date: 2026-07-18. Volume v3 is the accepted incremental successor to
the frozen volume-v2 architecture. It changes two player-volume pathways and
leaves team volume, availability, QB workload/propensity, target allocation,
and carry allocation unchanged.

## Promoted pathways

1. The conditional snap model adds a leakage-safe exponentially weighted
   three-year snap-share state, one-year snap-share trend, and three-year
   availability state. The history weight is `alpha=0.5` and follows player
   identity across team changes.
2. The any-carry eligibility hurdle adds prior rushing EPA per carry. This is
   the only efficiency-to-volume input accepted by default; it does not enter
   the positive carry-share allocator.
3. Standardized feature matrices are projected onto their full-rank SVD basis,
   and position effects use a sum-to-zero basis. Both transforms are persisted
   with posterior checkpoints and production model artifacts.

## Decision protocol

The walk-forward evaluation holds out 2022, 2023, and 2024, trains only on
earlier seasons, and uses nflverse Week 1 point-in-time rosters. Each candidate
fit used 1,000 tuning and 1,000 retained draws in each of four nutpie chains.
Diagnostics use all 4,000 samples. Predictive scoring deterministically thins
longer candidate posteriors to the frozen volume-v2 pipeline's 600 draws so all
independently fitted components share one draw axis.

Acceptance requires all of the following:

- lower pooled MAE/CRPS or Brier score for every required metric;
- a win in at least two of three holdouts for every required metric;
- no protected pass-stream regression beyond 0.5%;
- R-hat below 1.01, bulk ESS of at least 100, and zero divergences for every
  promoted component in every holdout.

## Results

| Pooled metric | Volume v2 | Volume v3 | Relative improvement | Fold wins |
|---|---:|---:|---:|---:|
| Snap MAE | 0.150879 | 0.149196 | 1.12% | 3/3 |
| Snap CRPS | 0.104375 | 0.103241 | 1.09% | 3/3 |
| Carry-eligibility Brier | 0.160270 | 0.152406 | 4.91% | 3/3 |
| Target MAE | 0.888362 | 0.885993 | 0.27% | 2/3 |
| Target CRPS | 0.660603 | 0.658344 | 0.34% | 2/3 |
| Carry MAE | 0.834271 | 0.826082 | 0.98% | 2/3 |
| Carry CRPS | 0.608741 | 0.601129 | 1.25% | 3/3 |
| Pass MAE | 5.552860 | 5.552860 | unchanged | protected |
| Pass CRPS | 4.053063 | 4.053063 | unchanged | protected |

The combined architecture passed. Across all six promoted component fits, max
R-hat was 1.004635, minimum bulk ESS was 1,210.1, and divergences were zero.

## Availability calibration decision

Availability draws now report exact `games_active / team_games`. Baseline 80%
coverage is 88.2%, 93.3%, and 95.4% for 2022-2024, revealing overcoverage.
Multi-year availability history improved pooled MAE by 0.60%, CRPS by 0.56%,
and any-appearance Brier score by 1.80%, but worsened coverage error by 4.96%
and regressed pooled pass MAE/CRPS. Position-specific concentration did not
repair the full predictive gate. Both availability variants are rejected, so
volume v3 retains the volume-v2 availability model.

## Artifacts and status

The resumable machine-readable decision is stored at
`.cache/season-average-validation/volume-v3-promotion-final/report.json`. Every
candidate posterior, checksum, model transform, diagnostic record, and metric
record is checkpointed beneath the same directory.

Volume v3 feature defaults are promoted in code. A final all-data production
fit remains before publishing live projections; holdout posteriors are
validation evidence and must not be reused as a live final fit.
