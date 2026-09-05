"""A naive ADP snake draft, which is the point rather than a shortcut.

Every team in the environment -- including the one a policy will control -- is
seated by the same mechanical rule: take the best player still on the board that
your roster can still use. No reaching, no runs, no positional value beyond what
the consensus board already prices. That makes the draft a *fixed, neutral*
starting condition rather than a second thing being optimised, so a difference
in final record is attributable to in-season decisions.

Two rules keep it from producing rosters no human would field. A cap stops a
team taking a third quarterback, kicker or defense, none of which can be started
in the same week as the first two. And a roster that owes as many players as it
has picks left is forced to take only what it still needs, which is what
guarantees every team ends able to field a legal lineup -- including the flex,
whose requirement is aggregate rather than per-position and is the one a
naive "fill each slot" rule gets wrong.

Note what the second rule does *not* do: it does not forbid taking a kicker
early. The board is free to hand one over in the tenth round if that is genuinely
the best player left under the caps. It only guarantees the roster cannot finish
short.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ffmodel.league.config import FLEX_POSITIONS, POSITIONS, LeagueConfig

# How many of each position one team may hold. A position absent from this map
# is uncapped: backs, receivers and tight ends can be stockpiled without limit,
# because depth at the flex-eligible positions is the whole point of a bench and
# capping it would be the environment making a roster-construction decision that
# a policy should be free to make for itself.
#
# The three that are capped are the ones where hoarding is not a strategy but a
# mistake: a third quarterback, kicker or defense cannot be started in any week,
# so the cap of two allows a bye-week handcuff and nothing beyond it.
DEFAULT_CAPS = {"QB": 2, "K": 2, "DST": 2}

# Positions a roster must end the draft holding, or it cannot field a lineup.
REQUIRED = ("QB", "RB", "WR", "TE", "K", "DST")

# Unlimited, spelled out rather than left as a magic absence.
UNCAPPED = float("inf")


@dataclass
class DraftResult:
    """Who ended up where, and the board that was drafted from."""

    rosters: dict[int, list[str]]
    picks: pd.DataFrame
    undrafted: pd.DataFrame

    def roster_frame(self) -> pd.DataFrame:
        rows = [
            {"team_id": team, "player_key": key}
            for team, keys in self.rosters.items()
            for key in keys
        ]
        return pd.DataFrame(rows)


def snake_order(teams: int, rounds: int) -> list[int]:
    """Pick order, reversing every other round."""
    order: list[int] = []
    for round_index in range(rounds):
        seats = range(teams) if round_index % 2 == 0 else range(teams - 1, -1, -1)
        order.extend(seats)
    return order


def _board(pool: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per draftable player for a season, ordered by consensus rank.

    A player is draftable if the board ranked him. Everyone else starts the
    season as a free agent, which is what the waiver wire is.
    """
    block = pool[pool["season"] == season]
    board = (
        block.groupby("player_key", as_index=False)
        .agg(
            player_name=("player_name", "first"),
            position=("position", "first"),
            adp_rank=("adp_rank", "first"),
            team=("team", "first"),
        )
        .dropna(subset=["adp_rank"])
    )
    return board.sort_values("adp_rank", kind="mergesort").reset_index(drop=True)


def _still_needed(counts: dict[str, int], slots) -> tuple[int, set[str]]:
    """How many more players this roster *must* take, and which positions help.

    "Fill the starting roster" is not just the dedicated slots. The flex needs a
    body too, and it can come from any of three positions, so the requirement is
    partly aggregate: a roster owes two backs, two receivers and a tight end for
    the dedicated slots *and* a sixth flex-eligible player on top, but that sixth
    can be any of the three. Counting only per-position shortfalls would let a
    team finish the draft one player short of a legal lineup, with every
    individual minimum satisfied.

    So the flex-eligible requirement binds on whichever is larger: the sum of the
    individual shortfalls, or the aggregate headcount. Returns the total still
    owed and the set of positions that would reduce it.
    """
    dedicated = slots.dedicated()

    total = 0
    wanted: set[str] = set()

    # Positions that can only fill their own slot.
    for position in ("QB", "K", "DST"):
        short = dedicated[position] - counts.get(position, 0)
        if short > 0:
            total += short
            wanted.add(position)

    # The flex-eligible group, where the aggregate can bind beyond the parts.
    individual = {
        position: max(0, dedicated[position] - counts.get(position, 0))
        for position in FLEX_POSITIONS
    }
    held = sum(counts.get(position, 0) for position in FLEX_POSITIONS)
    required = sum(dedicated[position] for position in FLEX_POSITIONS) + slots.flex
    aggregate = max(0, required - held)

    total += max(sum(individual.values()), aggregate)
    wanted.update(position for position, short in individual.items() if short > 0)
    if aggregate > sum(individual.values()):
        # Every dedicated minimum is met and the flex is what is still owed, so
        # any of the three closes it.
        wanted.update(FLEX_POSITIONS)
    return total, wanted


def run_draft(
    pool: pd.DataFrame,
    season: int,
    config: LeagueConfig,
    *,
    caps: dict[str, int] | None = None,
    seed: int = 0,
) -> DraftResult:
    """Seat every team from the consensus board.

    The seed only shuffles the draft *order*, not the picks: with a fixed board
    and a deterministic rule the picks follow from the seats, and randomising
    which team drafts first is the only variation that makes sense to average
    over across episodes.
    """
    caps = dict(DEFAULT_CAPS if caps is None else caps)
    slots = config.slots
    board = _board(pool, season)
    if len(board) < config.roster_spots:
        raise ValueError(
            f"{season} board has {len(board)} ranked players, short of the "
            f"{config.roster_spots} the league must seat"
        )

    rng = np.random.default_rng(seed)
    seats = list(range(config.teams))
    rng.shuffle(seats)
    order = [seats[pick] for pick in snake_order(config.teams, slots.size)]

    rosters: dict[int, list[str]] = {team: [] for team in range(config.teams)}
    counts: dict[int, dict[str, int]] = {team: {} for team in range(config.teams)}
    taken: set[str] = set()
    rows = []

    available = board.to_dict("records")
    for pick_number, team in enumerate(order, start=1):
        held = counts[team]
        spots_left = slots.size - len(rosters[team])
        owed, needed = _still_needed(held, slots)

        # Once the roster owes as many players as it has spots left, every
        # remaining pick is forced. This is the guarantee that no team finishes
        # the draft unable to field a legal lineup -- and the only mechanism
        # that puts a kicker and a defense on every roster, since the board
        # ranks both low enough that "best available" would otherwise skip them.
        forced = owed >= spots_left
        wanted = needed if forced else {
            position
            for position in POSITIONS
            if held.get(position, 0) < caps.get(position, UNCAPPED)
        }

        choice = None
        for candidate in available:
            if candidate["player_key"] in taken:
                continue
            if candidate["position"] not in wanted:
                continue
            choice = candidate
            break
        if choice is None:
            # No ranked player fits. Rather than leave a hole, relax the cap and
            # take the best remaining -- a roster with an empty slot cannot
            # field a lineup, and a silent hole is worse than a redundant back.
            for candidate in available:
                if candidate["player_key"] not in taken:
                    choice = candidate
                    break
        if choice is None:
            raise ValueError(f"board exhausted at pick {pick_number}")

        taken.add(choice["player_key"])
        rosters[team].append(choice["player_key"])
        held[choice["position"]] = held.get(choice["position"], 0) + 1
        rows.append(
            {
                "pick": pick_number,
                "team_id": team,
                "player_key": choice["player_key"],
                "player_name": choice["player_name"],
                "position": choice["position"],
                "adp_rank": choice["adp_rank"],
            }
        )

    picks = pd.DataFrame(rows)
    undrafted = board[~board["player_key"].isin(taken)].reset_index(drop=True)
    return DraftResult(rosters=rosters, picks=picks, undrafted=undrafted)
