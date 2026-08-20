# The snap model's feature prior (2026-08-04)

Opened while chasing task 33 — the carry allocator over-projects quarterbacks by
a third, and substituting observed snap share removes two thirds of that, so the
error is in the exposure the allocator reads. It did not fix task 33. It found
something else.

## What the prior does

`_matrix` projects the standardized features onto their principal directions
before the likelihood sees them, so `beta` has one entry per direction — 14 for
15 features, the set being rank deficient by one — and there is no per-feature
coefficient to widen. `feature_prior_scale` is the only lever, and it moves
every position at once.

Read through `implied = projection @ beta`, the historical scale of 0.35 leaves
two features pinned: `depth_rank` at 4.1 implied prior standard deviations and
`is_replacement_player` at 5.1, both with posterior widths below the prior's
own. Both separate backups from starters.

## Widening does move them

Trained on seasons before 2024, scored on 2024:

| scale | depth_rank | is_replacement_player | qb_listed_starter |
|---:|---:|---:|---:|
| **0.35** | −1.42 | +1.77 | +0.449 |
| 0.75 | −5.24 | +5.57 | +0.440 |
| 1.5 | −10.40 | +10.71 | +0.425 |
| 3.0 | −13.63 | +13.93 | +0.415 |

So the prior was constraining, and by an order of magnitude.

## It does not fix what it was aimed at

| group | n | MAE 0.35 | MAE 3.0 | change |
|---|---:|---:|---:|---:|
| RB | 152 | 0.1499 | 0.1406 | **−6.2%** |
| WR | 225 | 0.1776 | 0.1665 | **−6.3%** |
| QB starter | 32 | 0.1202 | 0.1162 | −3.3% |
| TE | 130 | 0.1399 | 0.1375 | −1.7% |
| **QB backup** | 51 | 0.2109 | 0.2147 | **+1.8%** |

Backup quarterbacks — the population this was for — are the only group that
gets worse, on MAE and CRPS both. Their mean bias improves from +0.017 to
+0.011 and their spread does not.

This was predicted backwards. The stated expectation was "a real improvement on
backup quarterbacks bought with a broad regression everywhere else". It is
exactly inverted: a broad improvement, and nothing for backups.

## Two corrections

**The 21% figure was in-sample.** The backup over-projection of 0.253 against
0.209 came from fitting on seasons before 2025 and evaluating on those same
rows. Held out on 2024 it is 0.268 against 0.251 — about 6.8%, a third the size.
The defect is real and smaller than first reported.

**The metric gains may be ill-conditioning, not learning.** An implied
coefficient of −13.6 means a one-standard-deviation move in `depth_rank` drops
the logit by 13.6, which is not a real effect. The SVD retains every rank-14
direction including ones with tiny singular values, and a wide prior lets those
blow up while barely changing predictions. That the held-out numbers improve
anyway does not make the mechanism sound, and it points at the rank tolerance
rather than the prior as the thing to change.

## Status

`feature_prior_scale` defaults to 0.35, unchanged. Nothing here is promoted.

Two separate follow-ups, deliberately not merged into one:

- The ~6% of held-out snap MAE on running backs and receivers is worth having
  and needs its own walk-forward, at a moderate scale rather than 3.0, gated,
  with the conditioning question settled first.
- Backup quarterback exposure is untouched and task 33 stays open. Nothing in
  this sweep moved it, so the mechanism is elsewhere.

## The gate rejects prior 3.0 (2026-08-20)

The sweep selected 3.0 by a plateau rule on **snap** MAE. The acceptance gate
looks at every stream, and one of them moves the other way.

`compare_validation_runs.py wf_snapconfbase.json wf_snap300.json` exits 1:

```
target/mae     +1.95%   0/3   regressed
snap/mae       -3.37%   3/3   improved
snap/crps      -3.18%   3/3   improved
carry/mae      -1.37%   3/3   improved
target/cov80  -3.66pp   3/3   improved
```

Per fold:

| fold | snap MAE | carry MAE | target MAE |
|---|---|---|---|
| 2022 | 0.14624 → 0.14242 (−2.6%) | 0.80783 → 0.80367 (−0.5%) | 0.83763 → 0.85825 (**+2.5%**) |
| 2023 | 0.14912 → 0.14367 (−3.7%) | 0.76603 → 0.74073 (−3.3%) | 0.86028 → 0.88864 (**+3.3%**) |
| 2024 | 0.14932 → 0.14359 (−3.8%) | 0.75959 → 0.75747 (−0.3%) | 0.78960 → 0.79027 (**+0.1%**) |

Widening the prior buys snap and carry accuracy on every fold and costs target
accuracy on every fold. Pooled, the target regression is 1.95% against a gate
that allows no consistent regression, so the change does not ship.

This is the failure mode the gate was rebuilt to catch, in its own words: *"It
only watched what somebody tabulated."* The sweep tabulated snap MAE. Nobody
looked at the target stream until the gate did.

**The on-config concern was unfounded.** `wf_snapconfbase.json` reproduces the
sweep's `wf_snapbase.json` to five decimals on all three folds, so the sweep was
already running the shipping configuration in this harness — the
`--cold-role-scale-mode` default that had corrupted a *scoring* run never
affected the volume walk-forward. Three container restarts were spent chasing a
confirmation that was already in hand.

**Where this leaves task 40.** The plateau at 3.0 is real for snap share and
irrelevant, because snap share is not what the package ships. Any future attempt
at this needs to select on a metric the gate will accept — the total-scoring
walk-forward, or a snap-and-target objective — rather than on one stream.
