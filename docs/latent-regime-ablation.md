# Player-season regime ablation

## Decision

Treat the proposed architectures as nested, not competing:

```text
preseason inputs -> shared player-season regime
                       |                 |
                   availability,      efficiency
                   snap, role,         residual rate
                   opportunity         distribution
```

The first implementation uses a discrete, interpretable state. A continuous
within-regime factor is deferred until the discrete model earns a gain in
walk-forward tests. This avoids an unidentifiable two-latent design and lets us
attribute any improvement to role, efficiency, or their shared draw.

## Regime contract

The states are `replacement`, `inactive`, `committee`, and `lead`.

- Replacement rows are the existing synthetic later-season call-up bucket.
  They are fixed to `replacement` and excluded from classifier fitting.
- For actual preseason rostered players, `inactive` means realized season
  availability below 25% or negligible position-specific role use.
- `lead` is the upper quartile of a training-fold, position-specific blend of
  realized playing time and direct opportunity. QB uses workload share; RB,
  WR, and TE use snap share plus carry or target share.
- The remaining active players are `committee`.

Those outcomes construct labels only in training. Future predictions use a
regularized multinomial model over lagged roles and availability, age and
experience, Week-1 roster/depth status, team change/cold start, and draft
priors. No realized current-season availability, snap, target, carry, or
fantasy-point field enters its feature matrix.

## Initial predictor screen

Walk-forward folds fit threshold and classifier on seasons before the holdout.
The probability objective is unweighted, prioritizing calibrated simulations
over balanced hard classifications.

| Holdout | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: |
| 2022 | 0.739 | 0.572 | 0.348 |
| 2023 | 0.788 | 0.505 | 0.300 |
| 2024 | 0.773 | 0.520 | 0.327 |

Predicted state frequencies were close enough to proceed, but not yet perfect:
the largest discrepancy is a 7.6 percentage-point underprediction of the 2022
committee state. It is a calibration target for the role-only ablation, not a
reason to introduce another latent dimension before testing the shared state.

Reproduce the screen with:

```powershell
python scripts/validate_season_regime.py --report-json reports/season-regime.json
```

## Remaining ablations

1. **Role only:** sample one regime per player/draw and use it only to adjust
   availability and team-conserving target/carry/pass allocation.
2. **Efficiency only:** retain the accepted volume posterior and use the same
   regime definition only for conditional efficiency residuals and dispersion.
3. **Joint:** use the *same sampled regime index* for both paths. This is the
   actual shared-state challenger.

Every candidate must beat the accepted volume/scoring posterior on the frozen
2022--24 walk-forward panel. A continuous residual state is only a follow-up
if the joint discrete model improves calibration but leaves structured residual
dependence.

## Role-only frozen-posterior result (rejected)

The first role-only screen reused the accepted volume-v3 posterior for each
2022--24 holdout and changed only the shared regime draw plus the
team-conserving pass/target/carry allocation tilt. Draw-level player alignment
and team-count conservation were asserted on every prediction.

| Stream | Mean CRPS change | Mean 80% coverage change |
| --- | ---: | ---: |
| Pass attempts | +0.0174 | -0.0097 |
| Targets | +0.0810 | -0.0412 |
| Carries | +0.0448 | -0.0286 |

Positive CRPS deltas are worse. The direct empirical tilt degraded every
stream in every holdout, especially targets, so it is rejected as a volume
candidate. The regime classifier remains useful research infrastructure, but a
future joint model must estimate regime effects inside the volume likelihood
rather than post-process already-fitted shares.

Reproduce this screen with:

```powershell
python scripts/validate_role_regime_volume.py --report-json reports/role-regime-volume.json
```

## Upstream likelihood challenger (in progress)

The next candidate supplies `inactive`, `committee`, and `lead` probabilities
to the availability, snap, QB-workload, eligibility, target, and carry
likelihoods. Training rows receive chronological out-of-fold probabilities:
for season `Y`, the regime classifier is fit only on seasons before `Y`.
Future rows use the classifier fit on all available historical seasons.

This is intentionally an upstream, leakage-safe screen—not yet the final
sampled latent-state model. It tests whether the regime signal has incremental
value once the probability is learned inside the existing role likelihoods.
The evaluator exposes it through:

```powershell
python scripts/validate_season_average.py --source nflverse --roster-mode point_in_time `
  --holdout-season 2024 --regime-likelihood-features
```

### 50-draw / one-chain screening result (not promotable)

The 2022--24 screen refit the full volume stack with the upstream regime
features. It is directional evidence only: the 50-draw, one-chain posterior
does not meet sampling diagnostics. Relative to the accepted v3 artifacts, the
mean deltas were:

| Metric | Mean MAE/Brier change | Mean CRPS change | Fold direction |
| --- | ---: | ---: | --- |
| Availability | -0.0046 MAE | -0.0026 | MAE 3/3; CRPS 2/3 |
| Snap share | -0.0087 MAE | -0.0057 | 3/3 |
| Carry eligibility | -0.0169 Brier | -- | 3/3 |
| Carries | -0.0220 MAE | -0.0038 | MAE 3/3; CRPS 2/3 |
| Targets | +0.0118 MAE | -0.0053 | MAE only 1/3 |

Negative values are improvements. The broad bundle therefore has useful
availability/snap/carry signal but fails the screen as a whole: target MAE
regressed in 2022 and 2023, and the protected QB-pass MAE also regressed in two
folds. Do not run a full promotion fit for this all-pathways version. The next
volume refinement, if pursued, should restrict the regime inputs to the
validated availability/snap/carry pathways and leave target/QB allocation
unchanged.

## Deferred history experiment

The current regime screen uses summarized lagged history. A separate future
experiment should compare it with fuller player-history representations, such
as multi-season sequence summaries or a learned history encoder, using the same
walk-forward folds. That experiment is intentionally deferred until the
regime-coupling ablations are complete.
