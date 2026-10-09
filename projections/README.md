# Projections

```
projections/
├── preseason/                 season-long draft projections, one set per season
│   ├── 2026_ppr.csv           player-season projection with p10/p50/p90 and its role inputs
│   ├── 2026_ppr.meta.json     blend weight, config and run time behind the csv
│   └── 2026_ppr.samples.npz   the posterior draws the quantiles were read from
├── weekly/<season>/week<NN>/  in-season projections, one folder per week projected
│   ├── start_sit.csv          that week's points: the lineup decision
│   └── rest_of_season.csv     that week to the end of the season: the waiver decision
└── overrides/                 hand-entered facts the data feeds have not caught up to
    ├── manual_status_2026.csv players out for a stretch or the season
    └── suspensions_2026.csv   announced bans, passed to project_season.py
```

## How they are made

| folder | command |
|---|---|
| `preseason/` | `python scripts/project_season.py --season 2026 --suspensions projections/overrides/suspensions_2026.csv` |
| `weekly/` | `python scripts/project_live.py --season 2026 --week 5` |

`week<NN>` is the week being **projected**, built from the weeks before it: `week04/` uses
weeks 1-3. Re-run late in the week. Game-status designations are filed Friday, and a
Wednesday run has none (see the caveats below).

## Reading the weekly files

`start_sit.csv` -- `projected_points` is the mean, `p10`/`p50`/`p90` the spread, and
`p_plays` the chance he is active. A low `p10` with a high `p_plays` is a bust risk; a low
`p_plays` is an availability risk. Two players with the same mean are not the same decision.

`rest_of_season.csv` -- every row is one calculation:
`rest_of_season_points = expected_games_played x points_per_active_game`.

| column | meaning |
|---|---|
| `rest_of_season_points` | expected total from this week to the end: games x rate |
| `expected_games_played` | of `games_left`, how many he is expected to be active for |
| `points_per_active_game` | his games-weighted rate **in the weeks he plays**; compare players on this |
| `points_per_scheduled_game` | the total over scheduled games, so it carries the absence; a cost, not a rate |
| `p10`, `p50`, `p90` | quantiles of the total, from measured residuals |
| `roster_status`, `out_through_week` | reserve-list players and overrides; the last week he is certain to miss |
| `on_bye_this_week` | his club has no game in the projected week. He has a rest of season but no row in `start_sit.csv` |

The identity holds to the CSV's rounding (about 0.02 points). See
`docs/ros-reconciliation-2026-10.md`.

Both files carry `overall_rank` and `pos_rank`. Raw PPR favours quarterbacks, so rank against
your own replacement level at each position rather than on `overall_rank`.

## Caveats that apply to every file

- **Returners are still under-projected.** Drafted players who just missed a game score
  about 8 points more than projected, top-50 players about 13. The model has no severity or
  timetable information, which the feeds do not carry.
- **Weeks 1-3 of a season are slightly less accurate** than the draft-board blend the file
  used before (1-2%); from week 4 the new total is at least as good.
- **A game cut short counts as a signal.** A player who played last week on far fewer
  snaps than usual (hurt at halftime, say) is projected as more likely to miss time, and
  his play probability and games drop accordingly. It is read from snap counts, so it
  cannot tell an injury from rest, and it does not exist for weeks 1-2. See
  `docs/partial-game-2026-10.md`.
- **Rates are unreliable when `expected_games_played` is small.** Rookies and long absences
  can read wildly high or low; filter on it before ranking on the rate.
- **The reserve-list floor is a floor, not a return date.** It is the fourth missed club game
  from placement and says when a player cannot be back, nothing about when he will be.
- **`overrides/manual_status_2026.csv` can only extend an absence**, never shorten one. It is
  read by `project_live.py` for every run.

## The weekly snapshots are what was sent, not regenerated

Each week's files are kept as produced; the week has passed, so they cannot be re-run and
the code has changed since.

| folder | what predates it |
|---|---|
| `week02/` | the active-game rate and the reserve-list handling; `points_per_game` assumes he plays every remaining game |
| `week03/` | the reserve-list handling and the direct rate; its `points_per_active_game` is the superseded ratio and overstates returning stars |
| `week04/` | the same schema and model as `week05/`. Built on a Wednesday: no game-status designations yet |
| `week05/` | current schema and model, built from weeks 1-4 (Dart and Achane out for the season via the override file); includes the clubs on a bye and the cut-short-game signal |

Do not use `week02/` or `week03/` for a decision in a later week.
