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

## The gate accepts prior 3.0 (2026-08-20)

`compare_validation_runs.py wf_snapconfbase.json wf_snapconf300.json` exits 0.
Every accuracy metric improves on three folds of three, nothing regresses:

```
snap/mae      -3.37%   3/3   improved
snap/crps     -3.18%   3/3   improved
target/mae    -2.38%   3/3   improved
carry/mae     -1.94%   3/3   improved
target/crps   -1.39%   3/3   improved
carry/crps    -0.94%   3/3   improved
```

Coverage is negligible everywhere. The one sampler watch, `team` R-hat 1.0107 at
2023, is the pre-existing statistic present in every arm.

### This section first said the opposite, and was wrong

An earlier version reported the gate **rejecting** prior 3.0 on `target/mae`
regressing +1.95% across 3/3 folds. That verdict came from comparing
`wf_snapconfbase.json` against `wf_snap300.json` — a baseline re-run on the
current configuration against a candidate from the original sweep. The two were
fitted under different configurations, so the comparison attributed a
configuration difference to the prior.

The check that was supposed to catch this compared **only `snap.mae`**, found it
identical to five decimals across all three folds, and concluded the old runs
were on-config. `snap.mae` is the one stream `cold_role_scale_mode` does not
touch. Every other stream differed:

| stream | old sweep vs re-run, baseline arm |
|---|---|
| snap | identical on all folds |
| availability, pass_qb, qb_workload | identical on all folds |
| **target** | mae 0.87555 → 0.83763, cov95 0.908 → **0.947** |
| **carry** | mae 0.81555 → 0.80783, cov95 0.940 → 0.943 |

The target cov95 moving 0.908 → 0.947 is the exact signature of the
`--cold-role-scale-mode` tri-state bug, which is how that bug was found in the
first place. Both sweep arms were off-config on the allocation layers; both
re-runs are correct; the sweep's *snap* numbers were always right, which is why
the plateau selection itself stands.

So the confirmation runs were not wasted after all — they were the whole reason
this is right. What was wasted was the intermediate verdict, published from a
one-stream check.

**The lesson is the gate's own.** Its rebuild note says it was previously
"watching what somebody tabulated". Verifying two runs match by tabulating one
metric is the same error in the same document, committed an hour after quoting
it.

### Where this leaves task 40

Prior 3.0 is selected by the plateau rule, confirmed on the shipping
configuration, and accepted by the gate on every stream. It is ready to promote.
