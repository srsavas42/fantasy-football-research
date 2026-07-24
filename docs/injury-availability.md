# Injury-informed availability candidate

This candidate adds injury occurrence and an expected recovery-duration signal
to the preseason games-played model. It is an availability feature layer, not a
medical prognosis and not a same-season performance feature.

## Data flow

Historical training uses nflverse weekly injury reports from 2009 through 2024
plus weekly roster statuses. The feature builder retains regular-season rows
only and maps player IDs through GSIS IDs; it never joins an injury by a fuzzy
name match.

For each player/team/season, contiguous availability-relevant injury reports
form an episode. Its realized recovery label is the number of regular-season
weekly roster states that are not `ACT`, from the episode start up to the first
later `ACT` status. An episode without a same-season return is right-censored
and is excluded from the recovery-time estimator. It still contributes to the
player's historical injury burden.

For a projected season `Y`, the returned features use only reports, episodes,
and roster outcomes from seasons before `Y`. The current-injury state comes
from the official Week-1 report in historical backtests. Live projections may
instead pass an archived Sleeper player snapshot. Sleeper's loader refuses to
create a missing historical snapshot, so an as-of date cannot silently pull
today's injury state into an old backtest.

The current expected recovery value is an empirical-Bayes estimate:

1. Start with completed injury episodes before `Y`.
2. Pool by injury body group and official status severity.
3. Shrink a player's own completed recovery history toward that pool.
4. Use the result only for a current availability-relevant injury report.

`Out`, `IR`, `PUP`, `NFI`, and `COV` map to the high-severity bucket. This
lets a current live reserve designation affect availability without pretending
that the model has a deterministic medical return date. Suspensions and
non-injury/rest designations are excluded.

## Feature contract

The builder adds these fields to `player_rows`:

```text
injury_history_available
prior_injury_report_weeks_3yr
prior_injury_out_weeks_3yr
prior_injury_episode_count_3yr
prior_injury_mean_recovery_weeks_3yr
prior_injury_weeks_since_last
current_injury_snapshot_available
current_injury_reported
current_injury_severity
current_injury_practice_severity
current_injury_expected_recovery_weeks
```

Zero is a valid healthy value only when the corresponding `*_available` flag
is one. The flags let the model distinguish a known healthy snapshot from a
missing injury data source.

## Acquisition and live use

Pull historical reports once; weekly rosters are already part of the normal
point-in-time roster workflow:

```powershell
ffmodel-data nflverse --seasons 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 --datasets injuries
ffmodel-data sleeper --snapshot-at 2026-07-18
```

For a live projection, label the archived snapshot with the season and the
prediction cutoff, then pass it into the season-data build:

```python
from ffmodel.features.season_average import build_season_average_data
from ffmodel.features.season_injury import load_live_injury_snapshot

injury_snapshot = load_live_injury_snapshot(2026, snapshot_at="2026-07-18")
data = build_season_average_data(
    seasons,
    source="nflverse",
    roster_mode="point_in_time",
    injury_snapshot=injury_snapshot,
)
```

The historical nflverse injury feed ends after 2024, so current archived
snapshots are the live path for later projections.

## Validation status

`scripts/validate_injury_availability.py` compares the accepted availability
model against the injury feature candidate in full-season walk-forward folds.
The first low-draw development screen was directionally favorable in 2022,
2023, and 2024: CRPS and MAE both declined in every fold, while 80% coverage
was effectively unchanged or improved. Those runs are not sufficient for a
promotion decision; the candidate remains opt-in until a longer multi-chain
posterior run and a downstream volume/season-scoring ablation are complete.

```powershell
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = (Resolve-Path '.cache').Path
python scripts/validate_injury_availability.py --draws 300 --tune 300 --chains 2 --nuts-sampler nutpie
```
