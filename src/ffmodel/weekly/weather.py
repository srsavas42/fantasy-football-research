"""Roof, temperature and wind: the game's physical conditions.

Three columns of the nflverse schedule that nothing in this package has ever
read. They are attached here as a measurable rung rather than assumed to help,
because the folklore around weather in fantasy football is considerably louder
than the evidence for it.

**The three are not the same kind of column, and the difference is the point.**

``roof``
    Known when the schedule is published, months before kickoff. It is fully
    populated -- every regular-season game 2016-2025 carries one of ``outdoors``,
    ``dome``, ``closed`` or ``open`` -- and it is not a forecast of anything. A
    live projection can read it with no more entitlement than it reads the
    opponent.

``temp`` and ``wind``
    Recorded *at* the game. They are what the conditions turned out to be, not
    what anyone knew on Sunday morning. A model fitted on them and scored on a
    holdout is therefore measuring the **ceiling**: what perfect foreknowledge of
    the weather would have been worth. That is a deliberately generous test, and
    it is the right one to run first. If perfect weather knowledge does not pay,
    a forecast of it certainly does not, and the question is closed for the price
    of one ablation. If it does pay, the shippable version reads
    :mod:`ffmodel.data.weather` -- Open-Meteo's previous-run forecast archive,
    which is already wired up and returns what was forecast at a stated lead
    time rather than what happened.

So ``roof`` is shippable as it stands and the two readings are a ceiling probe.
The rung reports them separately for exactly this reason.

**The encoding.** Indoors, the conditions are not missing -- they are controlled,
and a stadium with a closed roof is 70 degrees and still by construction. Filling
them that way is a statement of fact rather than an imputation, and the
``roof_indoor`` indicator alongside lets the fit distinguish a dome from a calm
70-degree afternoon in September, which are the same numbers and not the same
game. Outdoor games missing a reading -- 9.1% of them overall, and 48.7% of 2022,
where the feed simply stopped recording -- are left as NaN for the design's
median fill, with ``wx_missing`` marking them so those rows get their own level
instead of being quietly placed at the median of a distribution they may not sit
in.

**The thresholds.** Wind and cold are not believed to act linearly: the received
version of the claim is that wind matters once it is strong enough to move a
ball in flight, and cold once it is at freezing, neither of which a slope through
the middle of the distribution can express. ``wx_wind_high`` and ``wx_freezing``
are entered alongside the continuous columns so the fit can find a threshold if
one is there. They are set at 15 mph and 32 degrees, fixed a priori rather than
tuned, so no holdout is spent selecting them.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ffmodel.data import ingest

# Indoors is a controlled climate, not an unknown one.
INDOOR_TEMP = 70.0
INDOOR_WIND = 0.0

# Fixed a priori. Wind strong enough to move a ball, and freezing.
WIND_HIGH = 15.0
FREEZING = 32.0

# A retractable roof left open is an outdoor game and is treated as one; the
# category exists to record that the stadium could have been closed, which is a
# fact about the building rather than about the weather the players faced.
INDOOR_ROOFS = ("dome", "closed")
OUTDOOR_ROOFS = ("outdoors", "open")

WEATHER_COLUMNS = (
    "roof_indoor",
    "wx_temp",
    "wx_wind",
    "wx_wind_high",
    "wx_freezing",
    "wx_missing",
)


def load_game_conditions(seasons: Iterable[int]) -> pd.DataFrame:
    """(season, week, team) -> roof and weather, one row per team-week.

    Both clubs in a game face the same conditions, so the schedule's one row per
    game becomes two rows here, exactly as the market lines do.
    """
    empty = pd.DataFrame(columns=["season", "week", "team", *WEATHER_COLUMNS])
    try:
        schedule = ingest.load_schedules(list(seasons))
    except Exception:
        return empty
    if schedule.empty:
        return empty
    needed = {"season", "week", "home_team", "away_team", "roof"}
    if not needed.issubset(schedule.columns):
        return empty
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"] == "REG"]

    keep = ["season", "week", "roof"]
    for column in ("temp", "wind"):
        if column in schedule.columns:
            keep.append(column)

    frames = []
    for side in ("home_team", "away_team"):
        block = schedule[[*keep, side]].copy()
        frames.append(block.rename(columns={side: "team"}))
    out = pd.concat(frames, ignore_index=True)

    roof = out.pop("roof").astype(str).str.strip().str.lower()
    out["roof_indoor"] = roof.isin(INDOOR_ROOFS).astype(float)
    outdoor = roof.isin(OUTDOOR_ROOFS)

    for column, indoor_value in (("temp", INDOOR_TEMP), ("wind", INDOOR_WIND)):
        reading = (
            pd.to_numeric(out.pop(column), errors="coerce")
            if column in out.columns
            else pd.Series(float("nan"), index=out.index)
        )
        # Indoors the reading is supplied by the building, not the feed, so a
        # blank there is not a gap. Outdoors a blank stays a gap.
        out[f"wx_{column}"] = reading.where(outdoor, indoor_value)

    # Missing means: this game was played outdoors and nobody wrote down what it
    # was like. Indoor rows are never missing, by the line above.
    out["wx_missing"] = (outdoor & out["wx_temp"].isna()).astype(float)

    out["wx_wind_high"] = (out["wx_wind"] >= WIND_HIGH).astype(float)
    out["wx_freezing"] = (out["wx_temp"] <= FREEZING).astype(float)
    # A threshold on an unknown reading is unknown, not false.
    unknown = out["wx_missing"] == 1.0
    out.loc[unknown, ["wx_wind_high", "wx_freezing"]] = float("nan")

    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    out["team"] = out["team"].astype(str)
    return (
        out.dropna(subset=["season", "week"])
        .drop_duplicates(subset=["season", "week", "team"])
        .reset_index(drop=True)
    )


def attach_weather(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach this week's conditions to a weekly panel.

    Unlike every usage column in the panel, these are **not** lagged. The
    conditions belong to the game being projected, in the same way the closing
    line does: a projection for Sunday is entitled to know the game is indoors,
    and the ceiling probe is entitled -- for the purpose of measuring a ceiling
    -- to know what the weather turned out to be. Nothing here describes a
    previous week, so there is no history to lag.
    """
    frame = panel.copy()
    conditions = load_game_conditions(sorted(frame["season"].unique().tolist()))
    if conditions.empty:
        for column in WEATHER_COLUMNS:
            frame[column] = float("nan")
        return frame
    return frame.merge(conditions, on=["season", "week", "team"], how="left")
