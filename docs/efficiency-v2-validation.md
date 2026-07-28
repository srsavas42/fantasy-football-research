# Season-average efficiency v2 validation

Validation date: 2026-07-18. Efficiency v2 converts the accepted v1 point
forecasts into exposure-aware posterior predictive distributions for ten
season efficiency responses. It is validated as an input to total-season
scoring, with a deliberately narrow promotion rule for changing conditional
means.

## Architecture

- Completion, passing-touchdown, interception, catch, receiving-touchdown,
  rushing-touchdown, and fumble-lost rates use Beta-Binomial likelihoods.
- Passing yards per attempt, receiving yards per target, and rushing yards per
  carry use bounded Student-t likelihoods with separate season-level and
  opportunity-dependent dispersion.
- Training uses efficiency history through `Y-1` and volume projections for
  `Y` that are cross-fitted using response seasons strictly before `Y`.
- Feature transforms, position effects, accepted point models, posterior
  samples, and prediction metadata are persisted for reproducible inference.
- Fumbles lost use passing attempts + targets + carries as the exposure proxy;
  the raw season numerators are retained through the feature contract.

The flexible posterior mean was allowed to replace the accepted efficiency-v1
point forecast only when it improved pooled opportunity-weighted MAE and won
at least two of the three holdouts. Otherwise, the posterior likelihood wraps
the accepted ridge or pooled-prior mean and supplies future-season uncertainty.

| Response | Production mean | Posterior weighted MAE | Point benchmark MAE | Change | Fold wins |
|---|---|---:|---:|---:|---:|
| Completion rate | ridge | 0.02570 | 0.02570 | unchanged | 1/3 |
| Pass yards/attempt | ridge | 0.57729 | 0.57729 | unchanged | 1/3 |
| Pass TD/attempt | ridge | 0.01025 | 0.01025 | unchanged | 1/3 |
| Interceptions/attempt | ridge | 0.00671 | 0.00671 | unchanged | 0/3 |
| Catch rate | pooled prior | 0.05822 | 0.05822 | unchanged | 2/3 |
| Receiving yards/target | posterior regression | 1.09659 | 1.11007 | **1.21% better** | **2/3** |
| Receiving TD/target | pooled prior | 0.02272 | 0.02272 | unchanged | 0/3 |
| Rushing yards/carry | ridge | 0.62704 | 0.62704 | unchanged | 2/3 |
| Rushing TD/carry | pooled prior | 0.01581 | 0.01581 | unchanged | 1/3 |
| Fumbles lost/opportunity | pooled prior | 0.00444 | 0.00444 | unchanged | 0/3 |

Receiving yards per target is the only response whose flexible posterior mean
cleared the accuracy and stability gate. The fixed production mean policy
reproduces every other accepted point benchmark exactly.

## Predictive calibration and diagnostics

The final walk-forward run holds out 2022, 2023, and 2024. Each target/holdout
fit used four nutpie chains with 1,000 tuning and 1,000 retained draws. The
2023 receiving-yards-per-target fit was conservatively rerun with 2,000 tuning
and 2,000 retained draws after its first effective sample size was marginal.

| Response | Weighted CRPS | 80% coverage | 95% coverage |
|---|---:|---:|---:|
| Completion rate | 0.01903 | 0.893 | 0.993 |
| Pass yards/attempt | 0.40778 | 0.780 | 0.946 |
| Pass TD/attempt | 0.00734 | 0.868 | 0.965 |
| Interceptions/attempt | 0.00478 | 0.834 | 0.980 |
| Catch rate | 0.04112 | 0.870 | 0.972 |
| Receiving yards/target | 0.78777 | 0.778 | 0.971 |
| Receiving TD/target | 0.01542 | 0.899 | 0.983 |
| Rushing yards/carry | 0.45550 | 0.770 | 0.948 |
| Rushing TD/carry | 0.01109 | 0.917 | 0.971 |
| Fumbles lost/opportunity | 0.00276 | 0.936 | 0.990 |

Across all 30 final fits, maximum R-hat was 1.00957, minimum bulk ESS was
296.7, and divergences were zero. Seven of ten strict marginal coverage gates
passed. Completion rate, rushing-touchdown rate, and fumble-lost rate were
flagged for overcoverage rather than dangerous undercoverage; their discrete,
sparse support makes central interval coverage conservative. They remain
visible as calibration warnings and were carried into the downstream total
scoring gate rather than silently narrowed.

## Decision and artifacts

Efficiency v2 is accepted as the marginal efficiency distribution layer for
the total-season scoring experiment. This does not promote the combined total
fantasy-point distribution; that requires its own downstream CRPS, accuracy,
and coverage gate.

The resumable report, per-target posterior checkpoints, checksums, diagnostics,
and fold metrics are stored beneath
`.cache/season-average-validation/efficiency-v2-final/`. The 2023 receiving
yards per target refit is documented in
`holdout-2023/rec_yards_per_target/refit.json`.

```powershell
python scripts/validate_efficiency_posteriors.py `
  --seasons 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 `
  --holdouts 2022 2023 2024 --draws 1000 --tune 1000 --chains 4 `
  --nuts-sampler nutpie `
  --output-dir .cache/season-average-validation/efficiency-v2-final --resume
```
