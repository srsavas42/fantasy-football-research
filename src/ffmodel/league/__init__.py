"""A fantasy league as an environment: draft, schedule, lineups, waivers.

The weekly layer answers "how many points will this player score". A manager
does not get paid for that answer -- he gets paid for the lineup he sets and the
players he picks up, and those are decisions made under a roster constraint,
against eleven other teams, with a win-loss record as the thing that actually
matters. This package is the environment where that decision can be scored.

Deliberately separate from the model. The environment replays a *historical*
season using the points players really scored, so it is a fixed, honest world
that any policy can be dropped into and compared in -- the shipped weekly model,
a naive exponentially-weighted average, an ADP ranking, or a learned agent. If a
projection is better, it should show up here as more wins, and if it does not,
that is worth knowing before anything is trained.

The one rule the whole package is built around: **a policy may only see what a
manager could have seen on the day.** The environment holds the future because
it has to score it, and every observation it hands a policy is filtered to weeks
that have already been played. See :mod:`ffmodel.league.env`.
"""

from ffmodel.league.config import LeagueConfig, RosterSlots
from ffmodel.league.pool import build_player_pool

__all__ = ["LeagueConfig", "RosterSlots", "build_player_pool"]
