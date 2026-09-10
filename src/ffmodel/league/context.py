"""What a roster needs, and what the league is short of.

Every feature in :mod:`ffmodel.league.features` is a fact about a *player*: what
he has scored, what he is projected to score, what position he plays. That is
the right shape for a lineup, where the question really is "who is best" and the
roster is fixed. It is the wrong shape for an acquisition, where the question is
"is this player worth more *to me* than the man he would replace" -- and the
answer depends on a roster and a league rather than on the player.

So these are computed at decision time and cannot be cached per player-week. The
same free agent is a must-add for a team whose only tight end is hurt and a
waste of a roster spot for the team that already has two good ones, and nothing
about the player distinguishes those two cases.

Five features, each answering a question a manager actually asks:

``ctx_depth``
    How many bodies I already have at his position, against how many I must
    start. Negative means I am short.
``ctx_upgrade``
    What he adds over the man he would actually displace in my lineup, in
    rest-of-season points. This is the number the decision is really about, and
    it is unavailable to any per-player feature: it is a *difference* against a
    specific roster.
``ctx_wire_gap``
    What he adds over the next-best free agent at his position. A player who is
    barely better than the next man on the wire is not worth a roster spot; one
    with no equal behind him is.
``ctx_league_need``
    How many of the other eleven teams are short at his position. This is the
    only feature that looks outside the agent's own roster, and it prices
    scarcity: a replaceable tight end on a wire nobody else needs will still be
    there next week.
``ctx_starter``
    Whether he would walk straight into the starting lineup. A blunt version of
    ``ctx_upgrade`` that survives the cases where the upgrade is hard to compute.

**Everything here is public.** Rosters are visible in any league, the free-agent
pool is a page on the site, and what the other teams are short of is a matter of
reading their lineups. Nothing is drawn from a week that has not been played.
"""

from __future__ import annotations

import numpy as np

from ffmodel.league.config import FLEX_POSITIONS

CONTEXT_COLUMNS = (
    "ctx_depth",
    "ctx_upgrade",
    "ctx_wire_gap",
    "ctx_league_need",
    "ctx_starter",
)

# Rest-of-season points, roughly, for scaling the two difference features into
# the same range as a standardised per-player feature. A fixed divisor rather
# than a fitted one keeps the context block free of anything that would have to
# be re-estimated per season.
POINT_SCALE = 40.0


def _required(position: str, slots) -> float:
    """Starting bodies a roster needs at a position, flex included pro-rata."""
    dedicated = slots.dedicated().get(position, 0)
    if position in FLEX_POSITIONS and slots.flex:
        dedicated += slots.flex / len(FLEX_POSITIONS)
    return float(dedicated)


def _counts(keys, positions) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in keys:
        position = positions.get(key)
        if position is not None:
            out[position] = out.get(position, 0) + 1
    return out


def team_shortfalls(rosters, positions, slots, availability=None, week=None):
    """How many bodies each team lacks at each position, floored at zero."""
    out = {}
    for team, roster in rosters.items():
        if availability is not None and week is not None:
            roster = [k for k in roster if availability.is_available(k, week)]
        held = _counts(roster, positions)
        out[team] = {
            position: max(0.0, _required(position, slots) - held.get(position, 0))
            for position in slots.dedicated()
        }
    return out


def build_context(
    keys,
    *,
    values,
    roster,
    starters,
    positions,
    slots,
    free_agents,
    shortfalls,
    agent_team,
) -> np.ndarray:
    """One context row per candidate in ``keys``, in order.

    ``values`` maps a player key to his rest-of-season worth on whatever scale
    the caller is ranking by; ``starters`` is the set the agent would field this
    week. Both are supplied rather than recomputed so a caller that already has
    them does not pay for them twice.
    """
    keys = list(keys)
    if not keys:
        return np.zeros((0, len(CONTEXT_COLUMNS)))

    held = _counts(roster, positions)
    # The weakest man I would actually start at each position -- the one a new
    # arrival displaces, and so the one the upgrade is measured against.
    weakest: dict[str, float] = {}
    for key in starters:
        position = positions.get(key)
        if position is None:
            continue
        value = values.get(key, 0.0)
        if position not in weakest or value < weakest[position]:
            weakest[position] = value

    # The best free agent at each position other than the candidate himself,
    # which is what "is he replaceable" means on a wire.
    best_free: dict[str, list[float]] = {}
    for key in free_agents:
        position = positions.get(key)
        if position is not None:
            best_free.setdefault(position, []).append(values.get(key, 0.0))
    for position in best_free:
        best_free[position].sort(reverse=True)

    others = [team for team in shortfalls if team != agent_team]
    league = {
        position: float(
            np.mean([shortfalls[team].get(position, 0.0) for team in others])
        )
        if others
        else 0.0
        for position in slots.dedicated()
    }

    rows = np.zeros((len(keys), len(CONTEXT_COLUMNS)))
    starting = set(starters)
    for index, key in enumerate(keys):
        position = positions.get(key)
        value = values.get(key, 0.0)
        if position is None:
            continue
        rows[index, 0] = held.get(position, 0) - _required(position, slots)
        rows[index, 1] = (value - weakest.get(position, 0.0)) / POINT_SCALE
        alternatives = [v for v in best_free.get(position, []) if v != value]
        rows[index, 2] = (
            (value - alternatives[0]) / POINT_SCALE if alternatives else 0.0
        )
        rows[index, 3] = league.get(position, 0.0)
        rows[index, 4] = float(
            key in starting or value > weakest.get(position, float("inf"))
        )
    return rows
