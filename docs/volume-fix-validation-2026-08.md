# Walk-forward validation of the S0/S1/S2 fixes and the gate coupling

Run 2026-08-02. nflverse point-in-time rosters, holdouts 2022/2023/2024,
training only on earlier seasons — the volume-v3 protocol, at a **reduced
sampler budget of 400 draws / 400 tune / 2 chains** rather than the protocol's
1000/1000/4. Pooled figures weight folds by observation count.

Baseline is unmodified `7867cba`. Three configurations were fitted end to end:
`before` (main), `after` (S0+S1+S2), and `coupled` (S0+S1+S2 plus
`couple_gate_to_availability=True`).

## Headline

The bug fixes are close to neutral on the production path. **The availability
coupling is the change that actually moves the model**, and it only became
expressible once S2 made the gate coherent.

| metric | main | branch + coupling | change | fold wins |
|---|---:|---:|---:|---:|
| pass_qb MAE | 5.35277 | 5.22889 | **-2.31%** | 2/3 |
| pass_qb CRPS | 4.19177 | 4.02711 | **-3.93%** | 3/3 |
| qb_workload MAE | 0.15250 | 0.15217 | -0.22% | 1/3 |
| qb_workload CRPS | 0.12477 | 0.11848 | **-5.03%** | 3/3 |
| target MAE / CRPS | 0.86901 / 0.63923 | unchanged | +0.00% | — |
| carry MAE | 0.80793 | 0.81113 | +0.40% | 0/3 |
| carry CRPS | 0.57869 | 0.57853 | -0.03% | 2/3 |
| snap, availability | unchanged | unchanged | +0.00% | — |

Coverage, pooled:

| interval | main | branch + coupling |
|---|---:|---:|
| pass_qb 80% / 95% | 0.739 / 0.925 | 0.759 / 0.937 |
| qb_workload 80% / 95% | 0.628 / 0.802 | 0.672 / 0.822 |
| carry 80% / 95% | 0.878 / 0.949 | 0.883 / 0.957 |

## Correction: S1 is much smaller on nflverse than the review claimed

The review graded S1 **critical** on the strength of a legacy build — the only
one runnable offline at the time — where 48% of carry rows entered the
allocation at a saturated role prior of 1.0, about 5x an established back's
0.219. That measurement does not transfer to the production path, and the review
should not have implied it did.

`_role_prior` falls through `per_snap → lagged role → draft prior → cold`, so
the cold branch only binds where all three earlier terms are absent. On the
nflverse training frame (6,537 rows):

| stream | rows reaching the cold branch | who they are | pre-fix cold prior | post-fix |
|---|---:|---|---|---|
| target | 987 | QB only | RB .052, WR .061, TE .043 | RB .067, WR .096, TE .065 |
| carry | 2,924 | WR 1,579, TE 1,345 | WR .0053, TE .0003 | WR .0083, TE .0005 |
| pass | 5,302 | WR/RB/TE | all .0001 | all .0001 |

The target stream **cannot** change: the only rows that reach its cold branch
are quarterbacks, and `_design` masks quarterbacks out of the target multinomial
entirely. That is why target MAE and CRPS are identical to five decimal places
rather than merely close. The pass stream's cold branch is reached only by
non-quarterbacks, whose prior is 0.0001 either way.

So on nflverse the whole of S1 lands on WR and TE carry priors, and moves them
by a factor of about 1.6 — not the 468x and 2492x measured on legacy. The
legacy figures were real, but they came from a frame where the snapless rows
carried volume; on nflverse the snapless rows are mostly zero-volume, so the
mis-scaled arm pulled the mean *down* rather than up.

The unit mixing was still a genuine defect and the post-fix numbers are the
arithmetically correct ones. But as a *performance* change on the production
path, S1 costs 0.40% carry MAE with 0/3 fold wins and buys 0.03% CRPS and about
half a point of carry coverage at each level. **That does not clear the
volume-v3 promotion gate**, which requires a MAE win in at least two of three
holdouts. It should land as a correctness fix, with the carry MAE noted, not
advertised as an accuracy improvement.

## S0 is a provable no-op on nflverse

163 of 1,187 quarterback rows carry `snap_counts_observed = 1` with zero
`offense_snaps` — the population the fix redirects to the pass-attempt fallback.
All 163 have zero pass attempts, so the fallback recovers nothing and the fitted
response is bit-identical. The fix matters only where the snap feed omits a
passer who threw, which on nflverse never happens in this window and on the
committed CSVs happens to all 289 of them.

## S2 on its own

Isolating S0+S1+S2 against main, the QB layers improve distributionally while
point accuracy is flat: pass_qb CRPS -0.96% (2/3), qb_workload CRPS -1.56%
(2/3), pass_qb MAE +0.00%, qb_workload MAE +0.63%. That is the expected
signature of a fix that reshapes a distribution rather than moving its centre.
On its own it is marginal. Its value is that it makes the gate a coherent
factorisation, which is what lets the coupling work.

## The coupling, isolated

Against the branch baseline (so S0/S1/S2 held constant):

| metric | uncoupled | coupled | change | fold wins |
|---|---:|---:|---:|---:|
| qb_workload MAE | 0.15347 | 0.15217 | -0.85% | 2/3 |
| qb_workload CRPS | 0.12282 | 0.11848 | **-3.53%** | 3/3 |
| pass_qb MAE | 5.35288 | 5.22889 | **-2.32%** | 3/3 |
| pass_qb CRPS | 4.15170 | 4.02711 | **-3.00%** | 3/3 |
| target, carry | unchanged | unchanged | +0.00% | — |

Coverage: qb_workload 80% 0.640 → 0.672 and 95% 0.783 → 0.822; pass_qb 80%
0.735 → 0.759 and 95% 0.901 → 0.937. Target and carry are untouched, which is
the expected blast radius — the coupling only enters the quarterback room.

This clears every volume-v3 acceptance criterion on accuracy, CRPS, fold wins
and coverage direction. It does **not** yet clear the sampling-quality
criterion: max R-hat across folds reached 1.0435 (team), 1.0155 (snap) and
1.0125 (availability) against a 1.01 threshold, with zero divergences. Those
are budget artifacts — 400 draws in 2 chains against the protocol's 1000 in 4 —
and they affect all three configurations equally, including unmodified main.

## Recommendation

1. Land S0, S1 and S2. They are correctness fixes and are jointly neutral to
   slightly positive; state the carry MAE cost rather than hiding it.
2. **Promote the coupling** after one confirmation run at the full
   1000/1000/4-chain budget, purely to satisfy the R-hat and ESS criterion.
   Everything else already passes on three folds.
3. Re-run the total-scoring gate afterwards. `docs/season-scoring-v1-validation.md`
   failed on 95% coverage, and pass_qb 95% coverage moves 0.925 → 0.937 here,
   so the scoring layer's inputs are meaningfully better calibrated than when
   that gate was last evaluated.

---

# Total-season scoring gate, re-run

`docs/season-scoring-v1-validation.md` failed its gate on 95% coverage
(0.879-0.892 pooled against a 0.90 floor) and on CRPS, which "wins only one
holdout for each scoring system". Since the quarterback inputs are now better
calibrated, the gate was re-run.

`validate_season_scoring_posteriors.py` loads pre-fitted posteriors from
`.cache/season-average-validation/...`, which do not exist in a fresh container.
This run instead fits **both layers from data per holdout** through
`SeasonAverageScoringPipeline`, at 400/400/2. Absolute levels therefore are not
comparable to the v1 table; the paired comparison within this harness is.

| scoring | metric | main | + coupling | change | fold wins |
|---|---|---:|---:|---:|---:|
| standard | MAE | 33.2043 | 33.0958 | -0.33% | 2/3 |
| standard | CRPS | 24.2523 | 24.1020 | **-0.62%** | **3/3** |
| half-PPR | MAE | 38.2897 | 38.1769 | -0.29% | 2/3 |
| half-PPR | CRPS | 28.0801 | 27.9317 | **-0.53%** | **3/3** |
| PPR | MAE | 43.5221 | 43.4077 | -0.26% | 2/3 |
| PPR | CRPS | 32.0176 | 31.8712 | **-0.46%** | **3/3** |

Coverage moves slightly up everywhere: 80% from 0.773-0.775 to 0.776-0.777, 95%
from 0.917-0.918 to 0.918-0.920.

Two things to read carefully.

**CRPS now wins 3/3 on every scoring system.** That is precisely the criterion
v1 failed, and it is the criterion a distributional model should be judged on.
The MAE gain is small but consistent at 2/3.

**The 95% coverage failure does not reproduce in this harness — and that is not
a claim that the coupling fixed it.** Both configurations sit at 0.917-0.920,
comfortably above the floor, *including unmodified main*. The difference from
the v1 table is the harness, not the architecture: v1 combined a volume-v2
checkpoint with separately-fitted efficiency-v2 posteriors, whereas this fits
both layers together on the same folds. That is worth chasing on its own —
it suggests the recorded coverage failure may be partly an artifact of the
checkpoint combination rather than a property of the scoring architecture. The
honest next step is to re-run the original harness with its checkpoints and see
which explanation survives, not to declare the gate passed.

## Files

`scripts/validate_volume_fix_walkforward.py` and the JSON under
`scripts/validation_runs/` reproduce the volume comparison. The scoring
comparison uses the same cached nflverse frames through
`SeasonAverageScoringPipeline`.
