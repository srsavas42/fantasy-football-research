"""A naive ADP snake draft, which is the point rather than a shortcut.

Every team in the environment -- including the one a policy will control -- is
seated by the same mechanical rule: take the best player still on the board that
your roster can still use. No reaching, no runs, no positional value beyond what
the consensus board already prices. That makes the draft a *fixed, neutral*
starting condition rather than a second thing being optimised, so a difference
in final record is attributable to in-season decisions.

Two rules keep it from producing rosters no human would field. A cap per
position stops a team taking a sixth quarterback because the board happened to
rank him next, and kickers and defenses are deferred until the roster would
otherwise not have room for them -- which is what real drafters do, and what
"best available" alone gets badly wrong, since a board that ranks the top kicker
around the tenth round would otherwise have every team taking one there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ffmodel.league.config import LeagueConfig

# How many of each position one team may hold. Starters plus a sane bench
# allowance: enough depth to cover a bye or an injury, not enough to hoard.
# Kickers and defenses are capped at their starting requirement because a
# second of either is dead weight no naive manager carries.
DEFAULT_CAPS = {"QB": 2, "RB": 6, "WR": 6, "TE": 2, "K": 1, "DST": 1}

# Positions a roster must end the draft holding, or it cannot field a lineup.
REQUIRED = ("QB", "RB", "WR", "TE", "K", "DST")


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


def _still_needed(counts: dict[str, int], slots) -> dict[str, int]:
    """Mandatory positions this roster has not yet filled."""
    dedicated = slots.dedicated()
    return {
        position: dedicated[position] - counts.get(position, 0)
        for position in REQUIRED
        if dedicated[position] - counts.get(position, 0) > 0
    }


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
        needed = _still_needed(held, slots)

        # Once the roster has exactly as many spots left as mandatory positions
        # unfilled, every remaining pick is forced. This is what keeps a kicker
        # and a defense on every team without letting either be taken early.
        forced = sum(needed.values()) >= spots_left
        wanted = set(needed) if forced else {
            position for position in caps if held.get(position, 0) < caps[position]
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
