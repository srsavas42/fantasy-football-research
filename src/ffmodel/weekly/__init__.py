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

from ffmodel.weekly.frame import (
    PANEL_POSITIONS,
    build_panel,
    load_panel,
)

__all__ = ["PANEL_POSITIONS", "build_panel", "load_panel"]
