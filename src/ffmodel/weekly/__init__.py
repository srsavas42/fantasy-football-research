"""Weekly modelling: next week, and rest of season.

Deliberately independent of the season-average pipeline. The season layer
projects a full year from preseason information and is constrained by having one
observation per player per year; this layer has seventeen, and the questions it
answers -- start/sit, waiver, in-season value -- are not the season layer's
question restricted to a week.

Two responses, sharing a panel and a feature layer:

``next_week``
    Points in week ``w``, given every week before it. The start/sit decision.

``rest_of_season``
    Points from week ``w`` to the end of the regular season, given every week
    before it. At ``w = 1`` this is the draft question asked without a draft
    board; from ``w = 5`` on it is the waiver question.
"""

from pathlib import Path

from ffmodel.weekly.frame import (
    PANEL_POSITIONS,
    build_panel,
    load_panel,
)

# One canonical feature cache, named here so no script can quietly read a stale
# one. Two paths existed briefly -- a plain build and a "+news" build -- and the
# blend script kept reading the plain one for a full round of changes, returning
# byte-identical numbers that looked like a null result and were a stale file.
FEATURES_CACHE = Path(".cache/weekly_features_2016_2025.pkl")
PANEL_CACHE = Path(".cache/weekly_panel_2016_2025.pkl")

__all__ = [
    "FEATURES_CACHE",
    "PANEL_CACHE",
    "PANEL_POSITIONS",
    "build_panel",
    "load_panel",
]
