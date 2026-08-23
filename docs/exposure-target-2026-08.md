# The availability layer was fitting the wrong exposure (2026-08-23)

## What `games` actually counts

Roster-active weeks, capped at the team's slate. Not games played, not games
with a snap — weeks the player was on the active roster.

For a drafted player the distinction barely matters. For an undrafted one it is
the difference between employment and participation. Per player-team stint,
2022–2025:

| position | pool | roster | snap | stat line |
|---|---|---:|---:|---:|
| QB | drafted | 13.77 | 12.49 | 12.90 |
| QB | **undrafted** | **11.31** | **4.25** | 4.00 |
| RB | drafted | 13.26 | 12.43 | 12.38 |
| RB | undrafted | 9.54 | 7.47 | 5.11 |
| WR | drafted | 13.87 | 13.35 | 13.24 |
| WR | undrafted | 9.51 | 8.33 | 6.09 |
| TE | drafted | 14.39 | 13.82 | 13.40 |
| TE | undrafted | 11.15 | 10.43 | 6.23 |

A backup quarterback is on a roster 11.3 weeks and takes an offensive snap in
4.3. One regression is being fitted to a label that means two different things
for two halves of its population.

That connects directly to [the availability
diagnosis](availability-resolution-2026-08.md): the layer reproduces the
drafted/undrafted split *in sample*, which is what a model does when it cannot
tell two groups apart. Part of why it cannot is that the thing it is predicting
is not the same quantity for both.

## Why not the column already in the frame

`stat_activity_games` exists and is tempting — it counts weeks with a stat line,
which sounds like "games he played". It is worse than either alternative,
because a player who takes sixty snaps and is not thrown to records nothing.
Undrafted tight ends average **6.23** stat-line games against **10.87** with a
snap. Adopting it would trade a bias that inflates fringe exposure for one that
erases blockers.

Weeks with at least one offensive snap sits between the two and is the quantity
meant.

## The missing rows are not missing

14.5% of player-seasons have no row in the snap feed. They are filled with zero
rather than falling back to roster games, and that is exact rather than
convenient — across 2015–2025 those rows average:

- **1.70** roster games
- **0.01** games with a stat line
- **3.7%** drafted

A player absent from a covered team-season took no offensive snap. Zero is the
measurement, not a default. Filling from `games` would put the very bias this
column exists to remove back into a seventh of the frame.

The legacy snap source is a different case: it carries season totals with no
per-week rows, so it cannot count weeks at all. It leaves the column missing and
the pipeline rejects the target explicitly, rather than zero-filling into a
claim that nobody played.

## One decision, two models

The target cannot be changed in one place. `SeasonAvailabilityModel` fits a
count out of `team_games`; `SeasonSnapShareModel` divides the observed season
snap share by the matching fraction to recover a per-game rate; at prediction
time the pipeline multiplies that rate back by the availability draws.

Setting one without the other divides by one exposure and multiplies by
another, and **nothing downstream would raise**. The projection would simply be
wrong by the ratio between them — for undrafted quarterbacks, a factor of 2.6.
`SeasonAverageVolumePipeline.availability_target` owns the pairing so no caller
can set half of it.

## The part that argues against this change

The snap target **widens** the drafted/undrafted gap at every position:

| position | roster gap | snap gap |
|---|---:|---:|
| QB | 0.145 | **0.525** |
| RB | 0.211 | 0.278 |
| WR | 0.262 | 0.297 |
| TE | 0.193 | 0.201 |

That is the label doing its job — it separates starters from backups where the
roster label masked the difference. But a wider gap is *more* for the model to
resolve, not less. If the layer still cannot separate the groups, a more
meaningful target could leave a **larger** residual bias than the blurred one
did.

So this is not the safe cleanup it looks like. It makes the target more correct
and the estimation problem harder, and which effect dominates is an empirical
question this note does not answer.

## How it has to be judged

Not on availability metrics. Changing the target changes what `availability`
*means*, so its CRPS, coverage and bias are not comparable across the two arms —
they describe different quantities.

Total season points are the common currency: they are the same number however
the model decomposes them internally. The gate is the paired scoring
walk-forward on 2022–2024, both arms on
`.cache/ffmodel-wf-2025-snapexp`, which differs from the ADP cache in
`snap_games` and `snap_availability` and nothing else.

Off by default until that reports.
