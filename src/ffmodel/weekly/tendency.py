"""Pass rate over expected: what a coach *wants* to do, separated from what the
scoreboard made him do.

The panel already carries a team's recent pass attempts, and that column is a
mixture of two things a projection needs to keep apart. A team that threw 45
times last week either likes throwing or was down three scores, and those have
opposite implications for next Sunday: the tendency persists and the deficit does
not. nflverse prices every snap's pass probability from down, distance, field
position, score and clock, and reports the residual — `pass_oe`, in percentage
points. Averaged over a team-week it is that team's play-calling identity with
the game state divided out.

This is a different kind of column from expected fantasy points, which is why it
is worth building after that one came back a null. Expected points are a
restatement of a player's own opportunities, and the model already reads those
opportunities; there was nothing left once usage was partialled out. Pass rate
over expected is not a restatement of anything in the panel — it is the part of
play-calling the realized counts hide, and it is measurably more persistent than
the counts it corrects:

    week-to-week correlation, prior vs actual
        team pass attempts   0.244
        pass rate over expected   0.400

Two views are attached, both keyed on team-week and merged for the offence and
for the opponent's defence:

``proe``
    Mean pass rate over expected across the team's own dropback-eligible plays.
    Positive is a team that throws more than its situations call for.

``xpass``
    The mean *expected* pass rate itself — the situations, without the coach.
    Carried alongside so a fit can separate "passed more than expected" from
    "was in more passing situations", which are the two halves of a pass-heavy
    week and only one of them belongs to the team.

Both are lagged by the feature layer before anything sees them.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ffmodel.data import ingest

TENDENCY_COLUMNS = ("proe", "xpass")

# Plays the expected-pass model declines to price -- kneels, spikes, special
# teams -- carry no `xpass` and are dropped rather than counted as zero, which
# would read a victory formation as run-heavy identity.
REQUIRED = "xpass"


def load_team_tendency(seasons: Iterable[int]) -> pd.DataFrame:
    """Team-week pass rate over expected, one row per team-week.

    Play-by-play is loaded a season at a time and reduced immediately: the raw
    feed is 372 columns wide and ten seasons of it does not need to be resident
    at once to produce 5,522 team-weeks.
    """
    frames = []
    for season in sorted({int(s) for s in seasons}):
        try:
            plays = ingest.load_pbp([season])
        except Exception:
            continue
        wanted = [c for c in ("season", "week", "posteam", "xpass", "pass_oe") if c in plays.columns]
        if REQUIRED not in wanted or "pass_oe" not in wanted:
            continue
        rows = plays[wanted]
        rows = rows[rows[REQUIRED].notna() & rows["posteam"].notna()]
        frames.append(
            rows.groupby(["season", "week", "posteam"], as_index=False).agg(
                proe=("pass_oe", "mean"), xpass=("xpass", "mean")
            )
        )
        del plays, rows
    if not frames:
        return pd.DataFrame(columns=["season", "week", "team", *TENDENCY_COLUMNS])
    out = pd.concat(frames, ignore_index=True).rename(columns={"posteam": "team"})
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["team"] = out["team"].astype(str)
    return out


def attach_tendency(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach team-week play-calling tendency to a weekly panel.

    Unlike expected points this is a *team* quantity, so every player on a roster
    in a given week gets the same value and a missing week is missing for all of
    them rather than for one. No zero convention applies: a team-week with no
    priced plays is unknown tendency, not neutral tendency.
    """
    frame = panel.copy()
    tendency = load_team_tendency(sorted(frame["season"].unique().tolist()))
    if tendency.empty:
        for column in TENDENCY_COLUMNS:
            frame[column] = float("nan")
        return frame
    return frame.merge(tendency, on=["season", "week", "team"], how="left")
