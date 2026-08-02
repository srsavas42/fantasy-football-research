# What the ingest and feature layers discard

Every filter between the nflverse feed and the model matrix was traced and
measured on the 2014–2024 pull. The question was whether any of them removes
something the model needed, or removes it from one side of a ratio but not the
other. The short answer is no, with one observability gap now closed.

## The position filter

`normalize_model_positions` keeps QB/RB/WR/TE and drops everything else.

| | rows | opportunities |
|---|---|---|
| kept | 62,859 (33.1%) | 543,510 (99.906%) |
| dropped | 126,770 (66.9%) | 514 (0.094%) |

Two thirds of the rows go and a tenth of a percent of the football goes with
them. What is in that tenth of a percent is worth naming, because it is the
check that the filter is not quietly removing a real role:

| dropped position | rows with touches | opportunities |
|---|---|---|
| P | 162 | 163 |
| CB | 35 | 75 |
| LB | 41 | 58 |
| OT | 58 | 58 |
| DB | 46 | 46 |
| everything else | ~100 | ~114 |

Punters on fakes, linemen on tackle-eligible plays, defenders on laterals and
onside recoveries. There is no season in the window where the loss exceeds
0.16%, and no trend across the window. Fullbacks are *not* in this table —
`opportunity_position` folds FB into RB, which keeps 2,398 opportunities that a
literal position match would have dropped.

The filter is applied to the team totals too. `team_season_volume` calls
`normalize_model_positions` before aggregating, so player numerators and team
denominators are built from the same rows and every usage share still sums to
one over its support. `team_game_totals` keeps the distinction explicit:
`team_targets` is the full recorded total for generic usage features, and
`team_target_support` is the QB/RB/WR/TE total the allocator is actually
responsible for.

## The roster-status filter

`ROSTER_STATUSES` keeps ACT, RES, INA and EXE, and the point-in-time snapshot
takes the last status at or before the cutoff week. Players who record
opportunity but hold none of those statuses at cutoff — a September signing, a
practice-squad elevation, a waiver claim — do not vanish. They land in the
replacement bucket, which carries a measured share of the season:

| season | share of opportunity in the replacement bucket |
|---|---|
| 2015 | 7.11% |
| 2016 | 3.07% |
| 2017 | 4.17% |
| 2018 | 6.62% |
| 2019 | 4.71% |
| 2020 | 3.36% |
| 2021 | 4.73% |
| 2022 | 4.78% |
| 2023 | 4.05% |
| 2024 | 3.37% |

Three to seven percent of the league's opportunity belongs to players a
preseason roster could not have named. That is accounted for rather than
discarded, which is the property the allocation layer needs — a simplex that
omitted it would have to inflate everyone else to compensate.

## Team identity across relocations

32 distinct team codes, 32 in every season, with OAK/SD/STL folded into
LV/LAC/LAR. Every `prior_team` value in the player rows joins to a team row;
there are no orphans. Cross-season lags survive the relocations intact.

## Seasons

`player_rows` spans 2015–2024 from a 2014–2024 pull. 2014 is not missing: the
feature contract is strictly lagged, so 2014 exists only as the prior-season
source for 2015 and has no row of its own.

## The one gap, now closed

`load_injuries` catches `DataUnavailableError`, `OSError` and `ValueError` per
season and continues. That behaviour is right — the feed's coverage window
moves, and one unpublished year should not cost the caller a decade of injury
history — but it was silent. The availability model reads injury history as a
covariate, so a skipped season is a season fitted on a differently-informed
feature with nothing in the output to say so.

It now warns, naming the seasons and the underlying error. On the current feed
nothing is skipped: 2012–2024 were all requested and all returned, 70,401 rows.
The warning is there for the day that stops being true.
