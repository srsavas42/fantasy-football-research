# Season-average efficiency v1 validation

Validation date: 2026-07-17. The walk-forward evaluation uses nflverse weekly
stats from 2014-2024 and Week 1 point-in-time rosters. Response seasons are
2015-2024, comprising 320 team-seasons and 7,234 roster player-seasons. Every
model-generated volume feature for season `Y` is fit using response seasons
strictly earlier than `Y`.

## Data and architecture

- Canonical weekly ingestion retains air yards, YAC, EPA, and first downs while
  leaving unavailable optional observations missing rather than zero.
- Passing interceptions map from nflverse `passing_interceptions`; the previous
  alias mismatch had produced a zero-valued canonical field.
- Player efficiency is a ratio of season totals, not an average of weekly
  ratios. Each rate is partially pooled toward its same-season position mean
  with an opportunity-equivalent prior, then shifted from `Y` to `Y+1`.
- Volume challengers receive only lagged efficiency. Production keeps all
  broad share-allocation gates closed, while volume v3 admits lagged rushing
  EPA per carry only in the carry-eligibility hurdle. Future-efficiency
  training receives lagged efficiency plus cross-fitted volume projections.
- The first efficiency model is an opportunity-weighted ridge stack. It is a
  fast point-estimate gate before distributional Beta/Binomial and continuous
  likelihoods are added.

## Efficiency into volume

The first screen used a regularized direct-share allocator. Metrics cover all
available walk-forward seasons, exclude synthetic replacement buckets, and use
QB-only rows for pass attempts.

| Stream | Volume-only MAE | Refined challenger MAE | Change | Fold stability |
|---|---:|---:|---:|---:|
| Pass attempts | 6.209 | 5.818 | -6.3% | 9/10; 6/6 since 2019 |
| Targets | 1.146 | 1.146 | flat | efficiency excluded |
| Carries | 1.135 | 1.096 | -3.4% | 8/10; 6/6 since 2019 |

The QB challenger uses a reliability-weighted rank of prior yards/attempt, EPA,
first-down rate, and completion rate plus a pass-TD rate already pooled with a
200-attempt positional prior. The carry challenger uses only prior rushing EPA
per carry. Broad feature bundles were worse: the all-candidate target model
scored 1.169 MAE and the all-candidate carry model scored 1.144.

These gains did **not** clear the production-architecture gate. Volume v2
allocates QB attempts through separate snap-workload and attempts-per-snap
layers. A structurally matched point ablation produced:

| QB layer | MAE | RMSE | Fold wins | Recent wins |
|---|---:|---:|---:|---:|
| Production proxy | 5.567 | 9.068 | baseline | baseline |
| Efficiency in both layers | 5.545 | 9.085 | 4/8 | 2/6 |

The 0.4% MAE gain was inconsistent and slightly worsened RMSE. Passing
efficiency therefore remains a direct-share challenger only.

For carries, the accepted volume-v2 posteriors were reused for team volume,
availability, snaps, and carry eligibility while only the carry-share model was
refit with rushing EPA. Across the 2022-2024 holdouts:

| Carry model | MAE | RMSE | CRPS |
|---|---:|---:|---:|
| Volume v2 | 0.8343 | 1.9286 | 0.6087 |
| Volume v2 + rushing EPA | 0.8373 | 1.9307 | 0.6095 |

Rushing EPA lost in every carry-share holdout. At this stage no efficiency
metric entered the share allocator; the later hurdle-specific test below is a
separate pathway.

### Conditional role-context screen

A second screen tested whether lagged efficiency becomes more useful when its
effect depends on preseason role context. Position-room competition is
`1 - largest prior-role share` within the current team-position room; player
uncertainty is `1 - player prior-role share` within that room. Continuity
separates returning players from team changers and cold starts. These features
use only lagged roles, draft priors, current preseason rosters, and lagged
efficiency.

| Stream | Best contextual direct-share model | MAE | Reference MAE | Stability versus reference |
|---|---|---:|---:|---:|
| Pass attempts | unconditional + room interaction | 5.8156 | 5.8184 | 3/10; 1/6 since 2019 |
| Targets | no efficiency | 1.1462 | 1.1580 quality-only | no contextual model beat no-efficiency |
| Carries | unconditional rushing EPA + room interaction | 1.0910 | 1.0960 | 7/10; 5/6 since 2019 |

The passing improvement is only 0.05% and is not fold-stable. Targets again
reject every efficiency/context interaction. The carry interaction cleared the
direct-share screen, so it alone advanced to the posterior-controlled volume-v2
test.

| Carry model, 2022-2024 | MAE | RMSE | CRPS |
|---|---:|---:|---:|
| Volume v2 | 0.8343 | 1.9286 | 0.6087 |
| Volume v2 + rushing EPA + room interaction | 0.8408 | 1.9360 | 0.6110 |

The contextual carry challenger lost in all three posterior holdouts. This
confirms that its direct-share gain is absorbed by volume v2's playing-time,
snap, eligibility, and carry-propensity layers. The production share gate
therefore remains closed for all unconditional and conditional efficiency
features.

### Posterior pathway promotion

A resumable layer-matched experiment then tested efficiency in the sparse
any-carry hurdle rather than in carry allocation. It also tested multi-year
production history in the conditional snap model. The final run used 1,000
tuning and 1,000 retained draws in each of four chains for every candidate and
holdout. Full 4,000-sample posteriors were used for diagnostics; deterministic
thinning aligned predictions with the frozen volume-v2 baseline's 600 draws.

| 2022-2024 pooled metric | Volume v2 | Volume v3 | Change | Fold wins |
|---|---:|---:|---:|---:|
| Snap MAE | 0.15088 | 0.14920 | -1.12% | 3/3 |
| Snap CRPS | 0.10438 | 0.10324 | -1.09% | 3/3 |
| Carry-eligibility Brier | 0.16027 | 0.15241 | -4.91% | 3/3 |
| Target MAE | 0.88836 | 0.88599 | -0.27% | 2/3 |
| Target CRPS | 0.66060 | 0.65834 | -0.34% | 2/3 |
| Carry MAE | 0.83427 | 0.82608 | -0.98% | 2/3 |
| Carry CRPS | 0.60874 | 0.60113 | -1.25% | 3/3 |

The combined model passed every predictive, protected-stream, and sampling
gate. Across its six component fits, max R-hat was 1.0047, minimum bulk ESS was
1,210, and divergences were zero. Volume v3 therefore promotes exactly two
additions: three-year/trend snap and availability history in the snap model,
and prior rushing EPA per carry in the carry-eligibility hurdle. No other
efficiency feature is enabled.

## Future efficiency performance

The table compares the full cross-fitted model with the opportunity-weighted
MAE of the lagged, partially pooled persistence prior. The accepted ridge
regularization was 500.

| Response | Full weighted MAE | Pooled-prior weighted MAE | Change | Gate |
|---|---:|---:|---:|---|
| Completion rate | 0.0294 | 0.0296 | -0.7% | model, marginal |
| Pass yards/attempt | 0.568 | 0.611 | -7.1% | model |
| Pass TD/attempt | 0.0112 | 0.0114 | -1.8% | model, marginal |
| Interceptions/attempt | 0.00687 | 0.00717 | -4.3% | model |
| Catch rate | 0.0604 | 0.0588 | +2.7% | pooled prior |
| Receiving yards/target | 1.131 | 1.159 | -2.5% | model |
| Receiving TD/target | 0.0256 | 0.0226 | +13.1% | pooled prior |
| Rushing yards/carry | 0.611 | 0.638 | -4.3% | model |
| Rushing TD/carry | 0.0177 | 0.0161 | +10.4% | pooled prior |

Advanced air-yard/EPA/first-down inputs provide most of the incremental model
gain. Cross-fitted volume adds a smaller secondary improvement, most clearly
for passing yards per attempt. Sparse touchdown rates remain better handled by
the pooled prior; they should move next to exposure-aware Binomial or
Beta-Binomial likelihoods rather than receive a more flexible point regressor.

## Participation / route-data ablation

nflverse participation for 2016-2024 was expanded into a pass-play
participation proxy: the number of targeted pass plays on which a player was on
offense. This is not true routes run because blockers and non-route assignments
cannot be separated.

The proxy covered 3,360 of 7,234 rows. Adding it to the all-candidate target
model increased target MAE from 1.179 to 1.207, and receiving-yards-per-target
weighted MAE moved from 1.1307 to 1.1318. The proxy is rejected. No additional
provider is needed for efficiency v1; paid route data should be reconsidered
only if it provides complete player-level routes and can beat this baseline in
the same walk-forward test.

## Reproduction

```powershell
python scripts/validate_season_efficiency.py --participation
python scripts/validate_conditional_efficiency.py
python scripts/compare_efficiency_carry.py
python scripts/compare_efficiency_carry.py --feature-set rush_epa_room --output-dir .cache/season-average-validation/efficiency-carry-room
python scripts/compare_efficiency_pathway_posteriors.py --holdouts 2022 2023 2024 --gate-holdouts 2022 2023 2024 --draws 1000 --tune 1000 --chains 4 --candidates snap_history carry_eligibility_efficiency --output-dir .cache/season-average-validation/volume-v3-promotion-final --resume
```

Machine-readable output is written under
`.cache/season-average-validation/efficiency-v1/` and
`.cache/season-average-validation/efficiency-carry/`. The conditional screens
write to `.cache/season-average-validation/efficiency-conditional/` and
`.cache/season-average-validation/efficiency-carry-room/`. The final promotion
report is
`.cache/season-average-validation/volume-v3-promotion-final/report.json`.
