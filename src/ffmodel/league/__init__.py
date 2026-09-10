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

That rule cuts both ways, and the second half matters as much as the first: a
bye is published in August and a game-status report lands the day before
kickoff, so *stating* them leaks nothing and withholding them models a manager
nobody is. :mod:`ffmodel.league.availability` draws that line, and
:mod:`ffmodel.league.roster` acts on it.

The one deliberate exception is :func:`ffmodel.league.credit.grade_claims`, which
grades a waiver claim against the season that followed it. That is a training
signal built from the future on purpose, it is computed only after an episode
ends, and it never touches an observation.
"""

from ffmodel.league.availability import Availability, build_availability
from ffmodel.league.config import LeagueConfig, RosterSlots
from ffmodel.league.credit import ClaimCredit, grade_claims, total_credit
from ffmodel.league.pool import build_player_pool
from ffmodel.league.roster import Transaction, manage_roster

__all__ = [
    "Availability",
    "ClaimCredit",
    "LeagueConfig",
    "RosterSlots",
    "Transaction",
    "build_availability",
    "build_player_pool",
    "grade_claims",
    "manage_roster",
    "total_credit",
]
