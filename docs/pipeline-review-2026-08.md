# Pipeline review — ingestion through modeling

Reviewed at `7867cba` (main, after PR #5). Scope: `src/ffmodel/**`, `scripts/**`.
Findings are ordered scientific → code quality → efficiency, and within each
group by severity. Every claim marked *measured* was reproduced against a
`build_season_average_data(range(2016, 2021), source="legacy",
roster_mode="inferred")` build on this checkout; the fast test suite passes
(163 passed, 10 skipped, 1 xfailed) once `.cache/` exists.

## Status

| Finding | State |
|---|---|
| S1 cold-start role prior | **fixed**, regression tests in `tests/test_role_prior_units.py` |
| S2 QB workload hurdle | **fixed**, regression tests in `tests/test_qb_workload_hurdle.py` |
| S0 team-level snap flag read per player | **fixed** — found while verifying S1/S2; see below |
| everything else | open |

### S0 — `snap_counts_observed` is a team-level flag read as a per-player one  *(found during the fix, high)*

`_merge_snap_usage` sets `snap_counts_observed` from the **team-season's**
presence in the snap feed and zero-fills `offense_snaps` for players the feed
omits. `QBWorkloadShareModel._observed_counts` then reads that flag as "were
this player's snaps measured", so an omitted passer looks like a measured zero
and the pass-attempt fallback never fires.

Measured on the legacy build: all 300 QB rows carry `snap_counts_observed = 1`
and `offense_snaps = 0`, while 289 of them have positive `pass_att`. Every QB
workload count is therefore zero, `_design`'s `counts.sum(axis=1) > 0` filter
empties the design, and the fit dies inside PyTensor with
`MemoryError: failed to create Softmax iterator` on a `(0, 4)` softmax.

The practical consequence is bigger than the crash: **the primary volume
validation script cannot run its Bayesian stage at all under its own default
flags** (`--source legacy`, which resolves to `roster_mode="inferred"`). This
was confirmed against unmodified `7867cba`, so it predates the fixes here. It
also compounds S6 — the default configuration of the acceptance gate is both
the leakage-unsafe mode *and* a non-running one.

Fixed by requiring the player's own positive snap count before preferring snaps,
plus a clear error in `fit` in place of the PyTensor `MemoryError`. This matches
what every sibling model already does — `SeasonSnapShareModel.fit` gates on
`snap_observed & snap_share.gt(0)` and `QBPassPropensityModel.fit` on
`observed & snaps.gt(0)` — so `_observed_counts` was the outlier, not the
convention. On nflverse the change is a no-op except for passers the snap feed
omits entirely, but it does change what the workload model fits on, so it is
kept as its own commit and needs an nflverse revalidation before promotion.

### What could not be verified here

There is no nflverse access in this environment, so **no end-to-end holdout
comparison was run**. `scripts/validate_season_average.py --source legacy`
cannot reach the Bayesian stage even after S0: `QBPassPropensityModel.fit`
raises `QB propensity fitting requires observed quarterback snaps`, because the
committed snapcount CSVs contain no quarterback rows at all. Both unmodified
`7867cba` and this branch fail there identically.

The fixes are therefore verified by:

- unit-level measurement against real legacy feature frames (the tables above,
  re-measured after the fix);
- a sampled synthetic QB-room fit for S2, comparing the conditional fit against
  a reproduction of the merged unconditional-then-gated behaviour;
- regression tests that fail against pre-fix source and pass after
  (`tests/test_role_prior_units.py`, `tests/test_qb_workload_hurdle.py`);
- the full suite: 216 fast tests plus 5 sampler-heavy `-m slow` tests passing.

Running the nflverse walk-forward is the first thing to do on a machine with
data access, and is listed as step 1 below.

## What holds up

Worth stating first, because the leakage discipline is genuinely better than
most repos of this kind:

- The lag contract is real and consistently enforced. `lagged_efficiency_rows`
  shifts `Y → Y+1`; `add_volume_efficiency_features` ranks only within season on
  already-lagged inputs; `season_injury._expected_recovery` and the 3-year
  injury history both filter `episodes["season"] < season`;
  `efficiency_volume_pathways.early_competition_rows` lags its room aggregates
  before exposing them as `prior_room_*`. I looked specifically for
  same-season contamination in each and did not find it.
- `PROJECTION_BLANK_LABELS` correctly restores zero-filled label columns to
  missing on an unplayed season — a subtle failure mode handled properly.
- `schema.conform` distinguishing "unmeasured" (NaN) from "measured zero" (0.0),
  and `pass_sacks_available` / `sacks_observed` propagating that distinction all
  the way into `TeamSeasonAverageModel.fit`, is exactly right.
- `empirical_crps` uses the correct sorted-sample PWM identity (verified by hand
  for n=2).
- The validation docs record gate *failures* (season-scoring v1, the copula
  candidate) rather than quietly promoting. That is the right instinct.

---

## Scientific issues

### S1 — Cold-start role prior is unit-mixed, then clipped flat, and dominates the roster softmax  **(critical)**

`models/volume_season_average.py:611-643` (`SeasonRosterShareModel._fit_metadata`):

```python
exposure = snaps.where(snaps.gt(0), availability.clip(0.03, 1.0))
opportunity_rate = d[self.count_col].to_numpy(float) / np.clip(exposure, 0.03, None)
```

`snaps` is `offense_snaps` — a **season snap count** (median 307, p95 892).
`availability` is `observed_availability` — a **fraction ≤ 1**. Both go into one
array, so `opportunity_rate` carries two units differing by ~3 orders of
magnitude. Measured on the 2017-2020 build: 1732 rows take the snap branch,
541 (24%) take the fraction branch.

| position | mean rate, snap branch | mean rate, fraction branch |
|---|---:|---:|
| RB | 0.082 | 24.3 |
| WR | 0.099 | 54.2 |
| TE | 0.066 | 44.4 |

The per-position means become `cold_role_prior`:

| stream | fitted `cold_role_prior` | unit-consistent equivalent | ratio |
|---|---|---|---|
| target | QB 1.53, RB 1.80, WR 7.40, TE 7.44 | RB .082, WR .099, TE .066 | 22× / 74× / 113× |
| carry | QB 38.1, RB 8.95, WR 3.09, TE 1.50 | RB .296, WR .0066, TE .0006 | 30× / 468× / 2492× |

The inflation is **position-dependent**, so the softmax's within-team
normalization does not cancel it.

It then gets worse. `_role_prior` (line 692) ends with
`np.clip(prior, 1e-5, 1.0)`, so every one of those inflated values collapses to
exactly **1.0**. Any player falling through to the cold branch enters the roster
softmax with role prior 1.0 — versus a median established RB carry prior of
0.219. Measured share of rows landing on that clipped fallback:

| stream | rows at role prior 1.0 |
|---|---|
| carry | 1083 / 2273 (48%) — every WR/TE without a prior carry-per-snap |
| target | 270 / 2273 (12%) |
| pass | 7 / 2273 (0%) |

So in the carry allocator, a cold-start WR is scored ~5× an established RB
before any covariate applies. The carry-eligibility hurdle masks much of this by
gating most WR/TE out per draw, but conditional on being gated in they soak up
carries. This is very likely a real contributor to the top-quartile
over-dispersion recorded in `docs/season-scoring-coverage-diagnosis.md`.

**Fixed.** `opportunity_rate` is now per-snap throughout: rows without an
observed snap count contribute nothing to it rather than dividing by an
availability fraction, and the position estimate is clipped onto the per-snap
scale where it is persisted, so a saturated value stays visible in the metadata
instead of silently becoming the same number for every position. Re-measured on
the same 2017-2020 frame:

| stream | before | after |
|---|---|---|
| target | QB 1.53, RB 1.80, WR 7.40, TE 7.44 | QB 0.0001, RB 0.072, WR 0.092, TE 0.057 |
| carry | QB 38.1, RB 8.95, WR 3.09, TE 1.50 | QB 0.080, RB 0.281, WR 0.008, TE 0.0008 |

Rows sitting on the `_role_prior` upper clip fall from 1083/2273 (48%) to 1/2273
for carries and from 270/2273 to 0 for targets. A cold-start receiver's carry
prior goes from 1.0 — about 5x an established back's 0.219 — to 0.008, about 27x
below it, which is the right side of the comparison.

### S2 — The new QB workload hurdle biases the mean share it gates  **(high)**

Introduced in the just-merged PR (`models/season_availability.py`, `fit` /
`_hurdle_gate` / `predict_share_samples`). The Multinomial is fit
**unconditionally** over the whole QB room, including zero-attempt backups —
`_design` filters only groups whose total is zero, not individual rows — so its
`p` is already the *marginal* mean share. The hurdle is then fit separately on
`counts >= 25` and applied at prediction as a gate over that same softmax,
followed by renormalization.

Gating a distribution whose mean is already marginal moves the mean. Simulated
with starter/backup `p = 0.85 / 0.15` and gate probabilities `0.97 / 0.30`:

```
fitted multinomial mean share : [0.85  0.15 ]
post-gate realized mean share : [0.9473 0.0527]   -> backup share falls 2.8x
```

The hurdle does fix the bimodality it was written for, but it pays with a
systematic reallocation of pass attempts from backups to starters. Because
team pass attempts are conserved by `_allocate_season_counts`, the error shows
up purely as within-room misallocation — which is precisely the quantity the
backup-QB projections depend on.

**Fixed.** The softmax is now fit over the gated-in room only, via
`_conditional_design`, so it estimates the share *conditional* on clearing the
hurdle and multiplying by the gate restores the marginal. Rooms where nobody
cleared the bar carry no conditional information and leave the share fit, but
still label the hurdle. `_estimate_role_innovation` takes the same support, so
a sub-threshold passer's near-zero share is no longer charged to dispersion as
well as to the gate. The unused `draws` parameter is gone.

On a sampled synthetic QB panel (14 seasons x 24 rooms, starters missing 25% of
seasons), predicting with the pipeline's flat availability fallback:

| role | realized | before (bias) | after (bias) |
|---|---:|---:|---:|
| starter | 0.7826 | 0.8017 (+0.019) | 0.7602 (-0.022) |
| backup | 0.2149 | 0.1926 (-0.022) | 0.2162 (+0.001) |

The backup bias — the fantasy-relevant quantity — drops from -10.4% to +0.6%.

**Still open in the same layer.** Two things this fix does not address:

1. Availability now enters the workload softmax twice: once as
   `availability_offset`, and once through the gate, and at prediction time the
   two are drawn *independently*. Feeding realized availability into the same
   synthetic panel leaves a large residual backup bias (-53% before, -47%
   after) that is dominated by this incoherence rather than by the mean shift.
   Coupling the gate to the availability draw is a modelling change that needs
   its own validation.
2. The hurdle regresses only on `QB_WORKLOAD_FEATURES` and not on
   `prior_qb_snap_share`, which lives in the role *offset* rather than the
   design matrix — so a proven starter and an unproven one with the same
   depth-chart entry get the same gate probability.
3. `hurdle_min_attempts` is compared against `_observed_counts`, which is snaps
   where measured and attempts otherwise, so the threshold is not a pure attempt
   count. Both are small-workload bars; putting the gate on one basis is worth
   revisiting. Documented in the class docstring.

### S3 — Ridge roster baseline regresses on an arbitrary log-share floor  **(high)**

`evaluation/season_average.py:77`:

```python
y = np.log(np.clip(observed, 1e-5, None)) - np.log(role)
```

Measured on the 2017-2020 target stream: **18.1% of rows have exactly zero
share** and are pinned at the floor. Mean `y` at the floor is −4.13 versus +0.63
elsewhere; response sd rises from 1.22 (non-floored) to 2.34 (all). The
regression is substantially fitting a zero/non-zero indicator through a constant
chosen for numerical convenience, and the fitted value is sensitive to the
choice of `1e-5`.

This is not confined to a baseline: `add_walk_forward_volume_features` uses this
same `RidgeRosterBaseline` to build the `oof_*` columns that the **promoted**
efficiency models consume as covariates.

**Fix.** Either model the hurdle explicitly (as the Bayesian path already does)
and fit the ridge only on positive-share rows, or move to a Poisson/multinomial
likelihood on counts so zeros are representable without a floor.

### S4 — Train/serve construction gap in the efficiency volume covariate  **(high)**

`oof_*_per_team_game` is built two different ways under one name:

| | construction |
|---|---|
| fit — `evaluation/efficiency_season_average.py:78-85` | ridge softmax share × the team's **prior-season** per-game rate |
| serve — `simulation/season_scoring.py:105-113` | posterior mean of the **full Bayesian pipeline's** allocated counts ÷ team games, with team totals from the team model's **current-season projection** |

Two different estimators and two different team-rate sources. The efficiency
coefficient on `oof_targets_per_team_game` is estimated against ridge-quality
projections and applied to pipeline-quality ones. The repo already recognises
this class of bug — `tests/test_projection_feature_parity.py` was added for a
different instance of it — so this one deserves the same treatment.

**Fix.** Generate the training `oof_*` columns from cross-fitted runs of the
*production* volume pipeline, or (cheaper) add a parity test asserting the two
constructions agree in distribution on a common season.

### S5 — Efficiency is trained on an exposure-selected sample and applied unselected  **(medium)**

`PosteriorSeasonEfficiencyModel._eligible` filters training rows on **realized
current-season** exposure (`min_exposure` 20-50 by response), but
`predict_samples` scores every row, and `simulate_season_scoring` then applies
those rates at whatever projected exposure the volume layer produced —
including the near-zero exposures the model never saw. Selecting on exposure is
unavoidable for measuring efficiency, but the extrapolation should be explicit:
the Beta-Binomial concentration is a single global parameter estimated on
high-exposure players.

### S6 — The primary volume validation script defaults to the mode the README calls not leakage-safe  **(medium)**

`scripts/validate_season_average.py` defaults to `--source legacy` and
`--roster-mode auto`. With `source == "legacy"`, `build_season_average_data`'s
`should_try = roster_mode == "point_in_time" or source != "legacy"` is False, so
no point-in-time snapshot is loaded and the run falls to
`roster_snapshot=None` → `inferred_postseason`. The README states those files
"cannot reconstruct a leakage-safe preseason roster."

The row universe also collapses under that mode: the 2016-2020 legacy build
yields 2273 player-seasons (~568/season) — only players who actually recorded
stats. The zero-volume population is absent from both fitting and scoring, which
is the same failure `evaluation/holdout_alignment.py` was written to prevent on
the other path. Separately, this script uses a **single** holdout
(`holdout = max(seasons)`), not the three-fold protocol
`docs/volume-v3-validation.md` describes.

**Fix.** Default `--roster-mode point_in_time` (or default `--source auto`), and
print a loud banner when a run is scored in inferred mode.

### S7 — `vacated_opportunity` counts injured returnees as departures  **(medium)**

`features/crossseason.py:222-240` marks a player departed when `(key, team)` is
absent from season `Y+1`'s usage frame. That frame is built from player-week
**stat rows**, so a player still on the roster who missed the season, or
recorded no stats, is counted as vacated opportunity. The vacated signal is
therefore inflated by exactly the injury events the model wants to predict.
(The same function also raises `KeyError: 'team'` on an empty current season —
reproduced — because `DataFrame.apply(axis=1)` on an empty frame returns an
empty DataFrame rather than a boolean Series.)

### S8 — Yardage is a deterministic function of the count in the simulator  **(medium)**

`simulation/season_scoring.py:443-466`:

```python
pass_yards_per_completion = (ypa / clip(completion_probability, 0.05, None)).clip(0, 40)
pass_yds = rint(pass_cmp * pass_yards_per_completion)
```

Conditional on the rate draws there is **no residual yardage variance** — all
spread comes from the completion count and the rate draw. The division also
injects spurious dependence between two independently drawn rates: a low
catch-rate draw inflates yards-per-reception up to the 40 cap. Both the missing
residual and the 40/`0.05`/`0.03` clips distort the tails, which is where
`docs/season-scoring-v1-validation.md` reports 95% coverage failing its floor.

### S9 — Post-hoc dispersion knobs are swept on the holdout  **(low)**

`scripts/validate_season_scoring_posteriors.py` sweeps
`--dispersion-scales × --dependence × --point-dispersion-scales` and reports
each combination on the holdout. The docs are disciplined about rejecting
post-hoc widening, but the harness structurally invites picking a scale on the
test fold. Consider selecting on an inner fold and reporting only the selected
configuration on the holdout.

---

## Code quality

### C1 — Unreachable code hiding a missing baseline

`evaluation/season_average.py:231-232` sits after `return out` in
`_softmax_by_group`. Those two lines plainly belong to `persistence_volume`,
which consequently never emits `pred_pass_attempt_share` /
`pred_pass_attempts_per_game` — it computes `pass_share` and `pass_pg` and
discards both. `persistence_volume` currently has no callers, so this is dead
code concealing a real defect rather than an active bug.

### C2 — Variable shadowing in `player_preseason_rows`

`features/season_average.py:604` rebinds `observed` — the list of observed
seasons from line 518 — to a pandas Series. Harmless only because nothing reads
the season list after that point. Rename to `roster_observed`.

### C3 — `_fit_metadata` rebinds `availability` inside its position loop
(`volume_season_average.py:634`) after using it at line 619 to build `exposure`.
Same-name reuse for two different quantities in one function.

### C4 — Broad `except Exception` swallowing

`_merge_draft_capital`, `_merge_combine`, `_safe_draft_capital`,
`_resolve_depth_identities`, and `load_season_snap_usage` all catch bare
`Exception` and degrade to empty/NaN. A genuine bug in the combine parser or
identity resolver silently becomes "no athletic features" rather than an error.
Narrow these to the expected I/O and provider exceptions, and log when a
degradation fires.

### C5 — `_volume_feature_column` returns a raw-design index

`efficiency_season_average.py:690-699` returns an index into the *pre-SVD*
matrix. It is correct only because `predict_samples` rotates `beta` back through
`feature_projection` before indexing — but nothing at the call site says so, and
a future edit that drops the rotation would silently apply the wrong
coefficient. Worth an assertion or a docstring line.

### C6 — `_estimate_role_innovation` uses a different support than the model

For the target stream, `_design` masks QBs out of the likelihood, but
`_estimate_role_innovation` (called from `_fit_metadata`, before masking)
includes them in the softmax denominator and in the observed-share
normalization. The innovation scale is estimated under a support the model does
not use.

### C7 — Test suite fails on a clean checkout

`pyproject.toml` sets `--basetemp=.cache/pytest-tmp`, and pytest's `mkdir` has
no `parents=True`, so the first run on a fresh clone errors with
`FileNotFoundError: '.cache/pytest-tmp'`. Reproduced here; `mkdir -p .cache`
then gives 163 passed / 10 skipped / 1 xfailed. Either commit a `.cache/.gitkeep`
or create the directory in a `conftest.py`.

### C8 — Dead helpers

`base.squeeze_unit` is used only by `volume_season.py`;
`base.convergence_summary` is referenced only in its own module docstring —
`sampling_quality` superseded it.

### C9 — Known projection-parity gap is pinned but open

`tests/test_projection_feature_parity.py` xfails (strict) on
`prior_snap_share_trend` being all-NaN for a single-season frame. In the
production path `build_season_average_data` builds history and projection
together so the trend does resolve; the exposure is to any caller that
featurizes a projection season alone. Worth a guard in
`add_player_pathway_features` that raises when a consumed `*_trend` column is
wholly missing.

---

## Efficiency

Ordered by expected wall-clock impact.

### E1 — `sample_model` hardcodes `cores=1`

`models/base.py:40`. Every model samples its four chains **serially**. The
season-average pipeline fits eight-plus components per run, so on any multi-core
machine this is roughly a 4× wall-clock cost for the whole validation loop.
Make `cores` a parameter defaulting to `min(chains, os.cpu_count())`.

### E2 — Row-wise `apply` / per-row Python in hot paths

- `features/season_average.py:776-788` — three `DataFrame.apply(..., axis=1)`
  calls over the full player-season frame just to evaluate
  `base * exp(-(pick - 1) / scale)`. This is a map over 12 `(position, stream)`
  constants and vectorizes trivially.
- `crossseason.vacated_opportunity:234` — `cur.apply(lambda r: ..., axis=1)` for
  a set-membership test that is a two-column `MultiIndex.isin`.
- `season_injury._expected_recovery` — `itertuples` plus two `.loc` lookups per
  snapshot row.

### E3 — `_normalize_teams` calls `team_identity` once per row

`features/season_average.py:1411-1418`, invoked nine times across a build, on
frames up to player-week scale. `team_identity` is a long `if/elif` chain over
strings. Measured: 0.25s per 200k rows; 0.07s with an `lru_cache`. The real win
is mapping over the ~250 unique `(team, season)` pairs instead of every row.
Small in absolute terms, but it is pure waste repeated nine times.

### E4 — `_allocate_season_counts` is a Python double loop

`volume_season_average.py:1711-1718` — groups × draws. At 32 team-seasons, 600
draws, 3 streams that is ~57,600 `rng.multinomial` calls per prediction.
`rng.multinomial` accepts array `n` and `pvals`, so the draw loop can be
collapsed.

### E5 — `apply_efficiency_copulas` loops per player row

`simulation/season_scoring.py:285-292` generates a `(draws, k)` normal matrix
and `k` argsorts for every supported row. Vectorizable across rows in one
`(rows, draws, k)` pass.

---

## Suggested order of work

S1, S2 and S0 are done. What is left, in the order I would take it:

1. **Re-run the nflverse walk-forward** (2022-2024, the volume-v3 protocol) on
   this branch. S1 changes the cold-start prior for roughly half of all carry
   rows and S2 changes every QB room, so the published volume-v3 numbers no
   longer describe this code. Nothing else should be promoted until that is
   re-measured. S0 needs the same run to confirm it is the no-op it should be
   on nflverse.
2. **Make the legacy path either work or fail loudly at the top.**
   `--source legacy` cannot fit the QB layers at all — the committed snapcount
   CSVs have no quarterback rows — so `validate_season_average.py` dies partway
   through under its own defaults. Either default to `--source auto` /
   `--roster-mode point_in_time`, or check the source up front and refuse with
   one clear message instead of failing three models in. This is S6 plus S0 and
   it is what makes every other experiment awkward to run.
3. **The availability/gate incoherence in the QB workload layer** (S2's open
   item 1). It is now the dominant remaining bias in that layer, larger than the
   mean shift just fixed. Coupling the gate to the availability draw is the
   natural fix and needs its own ablation.
4. **S4 / S3** — the efficiency covariate's train/serve gap and the ridge
   log-share floor behind it. These feed the promoted efficiency models, and the
   parity test pattern from `tests/test_projection_feature_parity.py` applies
   directly.
5. **E1** — make `cores` configurable in `sample_model`. One line, roughly 4x
   faster validation loops on a multi-core machine, which makes steps 1-4
   materially cheaper to iterate on.
6. **S8** — give simulated yardage its own residual instead of deriving it
   deterministically from the count. This is the most likely remaining
   contributor to the 95% coverage floor that season-scoring v1 failed.
