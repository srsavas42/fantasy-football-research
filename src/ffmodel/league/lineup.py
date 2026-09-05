"""Turning a per-player score into a legal lineup, optimally.

Every policy in this package -- naive, model-driven or learned -- emits the same
thing: one number per rostered player, meaning "how much I want this player in
the lineup this week". This module is what converts that into a lineup card, and
keeping it separate from the policies is deliberate. It means a policy is only
ever judged on its *ranking* of its own players, not on whether it remembered
that a tight end can fill a flex, and it means two policies are compared on the
same assignment rule rather than on who implemented the roster constraint better.

The assignment is exact, not greedy-by-accident. Filling dedicated slots first
and then handing the flex whatever is left over is the obvious approach and it
is wrong: if the two best backs are also the two best flex candidates, greedy
seats them at RB and gives the flex a worse player than it could have had, when
seating one back and one receiver would have scored more. With one flex drawing
from three positions the exact answer is cheap -- try each flex-eligible
position, fill the rest greedily, keep the best total -- so there is no reason
to accept the approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ffmodel.league.config import FLEX_POSITIONS, RosterSlots


@dataclass(frozen=True)
class Lineup:
    """Who starts where, and what the card is worth under the scores given."""

    starters: dict[str, list[str]]
    bench: list[str]
    projected: float

    def starting_keys(self) -> list[str]:
        return [key for keys in self.starters.values() for key in keys]


def _take(pool: list[tuple[float, str]], count: int) -> tuple[list[str], list[tuple[float, str]]]:
    """The `count` best entries, and what is left."""
    chosen = [key for _, key in pool[:count]]
    return chosen, pool[count:]


def optimal_lineup(
    player_keys: list[str],
    positions: dict[str, str],
    scores: dict[str, float],
    slots: RosterSlots,
) -> Lineup:
    """Best legal lineup under ``scores``.

    ``scores`` is whatever the policy believes; this function does not care
    whether it is a projection, a random number, or a perfect oracle. That is
    the point -- swapping the score is how a policy is swapped.

    A player missing from ``scores`` is treated as a zero rather than dropped,
    so a policy that forgets somebody benches him instead of crashing.
    """
    by_position: dict[str, list[tuple[float, str]]] = {}
    for key in player_keys:
        position = positions.get(key)
        if position is None:
            continue
        by_position.setdefault(position, []).append((float(scores.get(key, 0.0)), key))
    for position in by_position:
        # Sort by score, breaking ties on the key so a lineup is reproducible.
        by_position[position].sort(key=lambda item: (-item[0], item[1]))

    dedicated = slots.dedicated()
    best: Lineup | None = None

    # The flex is the only choice worth searching over. Every other slot takes
    # the best remaining player of its own position, which is optimal given the
    # flex assignment is fixed.
    flex_options = [None] if slots.flex == 0 else list(FLEX_POSITIONS)
    for flex_position in flex_options:
        remaining = {position: list(entries) for position, entries in by_position.items()}
        starters: dict[str, list[str]] = {}
        total = 0.0

        # Reserve the flex first, from the *bottom* of its position's queue --
        # taking the best player for the flex and leaving a worse one for the
        # dedicated slot scores the same as the reverse, but reserving the
        # marginal player keeps the dedicated slots filled by the best players,
        # which is what makes the greedy fill below correct.
        flex_keys: list[str] = []
        if flex_position is not None and slots.flex:
            pool = remaining.get(flex_position, [])
            needed = dedicated.get(flex_position, 0)
            if len(pool) <= needed:
                # Not enough of this position to spare one for the flex.
                continue
            spare = pool[needed : needed + slots.flex]
            if len(spare) < slots.flex:
                continue
            flex_keys = [key for _, key in spare]
            total += sum(score for score, _ in spare)
            remaining[flex_position] = pool[:needed] + pool[needed + slots.flex :]

        for position, count in dedicated.items():
            if count == 0:
                continue
            pool = remaining.get(position, [])
            chosen, rest = _take(pool, count)
            # A roster too thin to fill a slot leaves it empty and scores zero
            # for it, which is the honest consequence of a bad roster rather
            # than an error. `_take` already returns only what exists, and the
            # total below sums only those, so nothing further is needed.
            total += sum(score for score, _ in pool[:count])
            starters[position] = chosen
            remaining[position] = rest
        if flex_keys:
            starters["FLEX"] = flex_keys

        bench = [key for entries in remaining.values() for _, key in entries]
        candidate = Lineup(starters=starters, bench=sorted(bench), projected=total)
        if best is None or candidate.projected > best.projected:
            best = candidate

    if best is None:
        # No flex assignment worked (a roster too thin to spare one). Fall back
        # to dedicated slots only.
        remaining = {position: list(entries) for position, entries in by_position.items()}
        starters = {}
        total = 0.0
        for position, count in dedicated.items():
            pool = remaining.get(position, [])
            chosen, rest = _take(pool, count)
            total += sum(score for score, _ in pool[:count])
            starters[position] = chosen
            remaining[position] = rest
        bench = [key for entries in remaining.values() for _, key in entries]
        best = Lineup(starters=starters, bench=sorted(bench), projected=total)
    return best


def score_lineup(lineup: Lineup, actual: dict[str, float]) -> float:
    """What the card really scored, under the points players actually put up."""
    return float(sum(actual.get(key, 0.0) for key in lineup.starting_keys()))


def round_robin(teams: int, weeks: int, seed: int = 0) -> list[list[tuple[int, int]]]:
    """A head-to-head schedule: one list of pairings per week.

    The circle method, which gives every team exactly one opponent a week and
    spreads rematches as evenly as the week count allows. With an odd number of
    teams one sits out each week, which is a bye.
    """
    rng = np.random.default_rng(seed)
    seats = list(range(teams))
    rng.shuffle(seats)
    if len(seats) % 2:
        seats.append(-1)  # a bye partner

    half = len(seats) // 2
    rotation = list(seats)
    schedule = []
    for _ in range(weeks):
        pairs = [
            (rotation[i], rotation[len(rotation) - 1 - i])
            for i in range(half)
            if rotation[i] != -1 and rotation[len(rotation) - 1 - i] != -1
        ]
        schedule.append(pairs)
        # Rotate all but the first seat.
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]
    return schedule
