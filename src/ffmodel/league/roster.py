"""The housekeeping a manager does before setting a lineup.

Setting a lineup is a ranking problem and the policies own it. Everything here
is the part that is not a ranking problem: a starter is ruled out, the roster
cannot cover him, and something has to move. Real managers do this every week
and mostly do it the same way, which is why it belongs in the environment as a
shared mechanic rather than in each policy as a strategy.

The cycle, in the order it has to happen:

1. **Bench whoever cannot play.** A player on bye or ruled out is worth zero with
   certainty, so he is scored below everybody and the assignment seats the next
   man up. This alone is most of the value -- with no roster move at all, depth a
   team already has covers the great majority of absences.
2. **Activate anyone whose injury has cleared.** He comes off IR onto the active
   roster; if it is full, the lowest-valued player on the bench is cut to make
   room. That cut is the real cost of stashing somebody, and it is why IR is not
   free.
3. **Cover a hole that is left.** If, after benching the absent, a starting slot
   has nobody to fill it, the week scores a zero for that slot. That is when a
   move is worth making: park the injured player on IR -- which is what IR is
   for, and only injuries qualify, never a bye -- and claim the best free agent
   who can fill the slot.

**A bye never sends anyone to IR.** It is one week and it resolves itself, and
real leagues would not allow it. A bye that leaves a hole is covered by a waiver
claim if the roster has room and eaten if it does not, which is exactly the
squeeze a thin roster feels in the middle of the season.

Every decision here needs to value players the team does not roster, so the
caller passes a ``valuation`` callable rather than a score dict. That keeps the
housekeeping using the same opinion as the lineup -- an EWMA team streams by
EWMA, a model-driven team streams by the model -- instead of a second, hidden
heuristic that would make the comparison between policies dishonest.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ffmodel.league.availability import Availability
from ffmodel.league.config import FLEX_POSITIONS, RosterSlots
from ffmodel.league.lineup import lineup_holes, optimal_lineup

# A score low enough that no assignment will ever seat the player, without being
# infinite: an infinity propagates into the lineup's projected total and turns a
# comparable number into a NaN.
BENCHED = -1e9

Valuation = Callable[[Iterable[str]], dict[str, float]]


@dataclass(frozen=True)
class Transaction:
    """One roster move, recorded so a season can be audited after the fact."""

    week: int
    kind: str  # bench-out | ir-place | ir-activate | waiver-add | drop
    player_key: str
    counterpart: str | None = None
    # Hours from the season's start. A week has two transaction phases and the
    # waiver period is measured against this, so "which week" is not enough to
    # say whether a move was legal.
    hour: int = 0

    def __str__(self) -> str:
        tail = f" <- {self.counterpart}" if self.counterpart else ""
        return f"w{self.week}h{self.hour} {self.kind} {self.player_key}{tail}"


def availability_scores(
    scores: dict[str, float],
    keys: Iterable[str],
    availability: Availability,
    week: int,
) -> dict[str, float]:
    """``scores`` with everyone known not to be playing pushed to the bottom.

    Applied to the scores rather than inside the assignment so that a policy
    which *wants* to start an absent player still can -- the environment states
    the fact, it does not overrule the decision. Nothing sensible wants to, but
    the distinction keeps the lineup module free of football rules.
    """
    out = dict(scores)
    for key in keys:
        if not availability.is_available(key, week):
            out[key] = BENCHED
    return out


def _bench_order(
    roster: list[str],
    positions: dict[str, str],
    values: dict[str, float],
    slots: RosterSlots,
    protected: set[str] | None = None,
) -> list[str]:
    """Who is on the bench, worst first -- the order they would be cut in.

    Computed from the lineup the team *would* set on merit, ignoring who is
    available this week: cutting a player because he happens to be on bye is how
    a roster loses its best back in week nine.

    ``protected`` is whoever arrived this week. Without it a manager covering two
    holes in the same week cuts the player it claimed for the first one to pay
    for the second, over and over -- the roster churns, nothing improves, and
    the waiver wire looks busy for no reason.
    """
    protected = protected or set()
    lineup = optimal_lineup(roster, positions, values, slots)
    starting = set(lineup.starting_keys())
    bench = [
        key for key in roster if key not in starting and key not in protected
    ]
    return sorted(bench, key=lambda key: (values.get(key, 0.0), key))


def _fill_positions(hole: str) -> tuple[str, ...]:
    """Which positions can fill a given empty slot."""
    return FLEX_POSITIONS if hole == "FLEX" else (hole,)


def manage_roster(
    *,
    roster: list[str],
    ir: list[str],
    free_agents: list[str],
    positions: dict[str, str],
    availability: Availability,
    week: int,
    slots: RosterSlots,
    valuation: Valuation,
    allow_waivers: bool = True,
    protected: set[str] | None = None,
    wire=None,
    hour: int = 0,
) -> list[Transaction]:
    """Run the cycle for one team in one week. Mutates the lists it is given.

    Returns the moves it made. An empty list is the common case and the one to
    expect: a roster with ordinary depth absorbs most absences by benching.
    """
    moves: list[Transaction] = []
    values = valuation(list(roster) + list(ir))
    # Nobody who arrives during this run may be cut during the same run, and
    # neither may anyone the caller has already committed to -- the agent's own
    # waiver claim, which housekeeping has no business reversing.
    protected = set(protected or ())

    # 1. Anyone whose injury has cleared comes back, making room if needed.
    returning = [key for key in ir if not availability.is_out(key, week)]
    for key in returning:
        if len(roster) >= slots.size:
            bench = _bench_order(roster, positions, values, slots, protected)
            if not bench:
                # Nothing to cut: every player is a starter. He stays parked
                # rather than displacing somebody who is playing this week.
                continue
            cut = bench[0]
            roster.remove(cut)
            free_agents.append(cut)
            if wire is not None:
                wire.dropped(cut, hour)
            moves.append(Transaction(week, "drop", cut, counterpart=key, hour=hour))
        ir.remove(key)
        roster.append(key)
        protected.add(key)
        moves.append(Transaction(week, "ir-activate", key, hour=hour))

    # 2. What the card looks like once the absent are benched.
    #
    # Built from the players who can actually play, rather than from the whole
    # roster with the absent scored low. The distinction is the entire point: a
    # team whose only kicker is on bye still *fills* its kicker slot when the
    # assignment is allowed to seat him, so scoring him at the bottom hides the
    # hole instead of revealing it. Dropping him from the candidate list is what
    # makes the empty slot visible, and the empty slot is the whole signal.
    playing = [key for key in roster if availability.is_available(key, week)]
    lineup = optimal_lineup(playing, positions, valuation(playing), slots)
    holes = lineup_holes(lineup, slots)
    if not holes or not allow_waivers:
        return moves

    # 3. A hole is worth a move. Park an injured player to free the spot where
    #    one is available -- that is what IR is for -- and claim a replacement.
    for hole, count in sorted(holes.items()):
        eligible = _fill_positions(hole)
        for _ in range(count):
            candidate = _best_free_agent(
                free_agents, positions, availability, week, eligible, valuation,
                wire, hour,
            )
            if candidate is None:
                break
            if len(roster) >= slots.size and not _make_room(
                roster=roster,
                ir=ir,
                free_agents=free_agents,
                positions=positions,
                availability=availability,
                week=week,
                slots=slots,
                values=values,
                moves=moves,
                protected=protected,
                wire=wire,
                hour=hour,
            ):
                break
            roster.append(candidate)
            free_agents.remove(candidate)
            if wire is not None:
                wire.added(candidate, hour)
            protected.add(candidate)
            # A new arrival with no entry here reads as worthless and would be
            # the first player cut to cover the next hole.
            values.update(valuation([candidate]))
            moves.append(Transaction(week, "waiver-add", candidate, counterpart=hole, hour=hour))
    return moves


def _make_room(
    *,
    roster: list[str],
    ir: list[str],
    free_agents: list[str],
    positions: dict[str, str],
    availability: Availability,
    week: int,
    slots: RosterSlots,
    values: dict[str, float],
    moves: list[Transaction],
    protected: set[str] | None = None,
    wire=None,
    hour: int = 0,
) -> bool:
    """Free one active roster spot. IR first, a cut second.

    Preferring IR is not just tidiness: a player ruled out is usually the most
    valuable player who is currently worth nothing, so cutting him to cover his
    own absence is the move a manager most regrets. Parking him keeps him.
    """
    parkable = [
        key
        for key in roster
        if availability.is_out(key, week) and len(ir) < slots.ir
    ]
    if parkable:
        # The best of them, because IR holds one and should hold the player
        # worth keeping.
        key = max(parkable, key=lambda k: (values.get(k, 0.0), k))
        roster.remove(key)
        ir.append(key)
        moves.append(Transaction(week, "ir-place", key, hour=hour))
        return True

    bench = _bench_order(roster, positions, values, slots, protected)
    if not bench:
        return False
    cut = bench[0]
    roster.remove(cut)
    free_agents.append(cut)
    if wire is not None:
        wire.dropped(cut, hour)
    moves.append(Transaction(week, "drop", cut, hour=hour))
    return True


def _best_free_agent(
    free_agents: list[str],
    positions: dict[str, str],
    availability: Availability,
    week: int,
    eligible: tuple[str, ...],
    valuation: Valuation,
    wire=None,
    hour: int = 0,
) -> str | None:
    """The most valuable *addable* free agent who can fill the slot this week.

    Filtered on two things. Availability, because a replacement who is himself
    on bye leaves the slot exactly as empty as it was. And the waiver wire,
    because a player cut an hour ago is not available to anybody yet -- which is
    what stops a team cutting a player and re-claiming him in the same breath.
    """
    candidates = [
        key
        for key in free_agents
        if positions.get(key) in eligible
        and availability.is_available(key, week)
        and (wire is None or wire.is_free(key, hour))
    ]
    if not candidates:
        return None
    values = valuation(candidates)
    return max(candidates, key=lambda key: (values.get(key, 0.0), key))
