# Pipeline review — ingestion through modeling

Reviewed at `7867cba` (main, after PR #5). Scope: `src/ffmodel/**`, `scripts/**`.
Findings are ordered scientific → code quality → efficiency, and within each
group by severity. Every claim marked *measured* was reproduced against a
`build_season_average_data(range(2016, 2021), source="legacy",
roster_mode="inferred")` build on this checkout; the fast test suite passes
(163 passed, 10 skipped, 1 xfailed) once `.cache/` exists.

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

**Fix.** Compute `opportunity_rate` on one basis — per-snap throughout, with a
per-snap fallback for snapless rows rather than an availability fraction — and
either drop the upper clip on the cold branch or clip on the per-snap scale
(e.g. `1e-5, 0.5`). Add a test asserting `cold_role_prior[pos] < 1.0` for every
position and stream; today nothing catches this because every value silently
saturates to the same legal number.

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

**Fix.** Fit the softmax *conditional on the gate*: mask non-gated rows out of
the Multinomial likelihood at fit time (or model the mixture jointly), so the
softmax estimates the conditional-on-playing share and gating restores the
marginal. Two smaller points in the same code: `_hurdle_gate(design, draws, rng)`
never uses `draws`; and the hurdle regresses only on `QB_WORKLOAD_FEATURES`
(depth rank, listed starter, age, …) and not on `prior_qb_snap_share`, which is
in the role *offset* rather than the design matrix — so a proven starter and an
unproven one with the same depth-chart entry get the same gate probability.

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

1. **S1** — the cold-start prior. Largest measured distortion, affects the
   promoted volume path, and is cheap to fix and test.
2. **S2** — the QB hurdle, before it is relied on. It is new, so the fix is not
   yet load-bearing on any published result.
3. **S4 / S3** — the efficiency covariate's train/serve gap and the ridge floor
   behind it.
4. **S6** — change the validation default so the acceptance gate stops running,
   by default, in the mode the README disclaims.
5. **E1** — one-line change, ~4× faster validation loops, which makes everything
   above cheaper to iterate on.
