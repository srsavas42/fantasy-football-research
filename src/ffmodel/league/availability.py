"""Who a manager knows, before kickoff, cannot score.

The environment's opponents used to start whoever their average liked best,
which meant they started players on bye. Nobody does that. A bye is published
before the season begins and an "Out" designation lands a median 28 hours before
kickoff (measured in :mod:`ffmodel.weekly.news`), so both are facts a manager has
in hand when the lineup locks. An environment whose opponents ignore them is not
a hard environment, it is a strawman, and any agent measured against it collects
an edge that would evaporate against a real league.

Two sources, and the line between them is what is *knowable*, not what is true:

``BYE``
    The player's club does not play. Certain, known in August, and the larger of
    the two effects -- every player has one, and it costs a starting slot every
    time it is missed.

``OUT``
    The club's own game-status report ruled him out. Precise where it fires: of
    the rows it flags across 2023-2025, exactly zero recorded a stat line.

Everything else -- a healthy scratch, a first-quarter hamstring, a starter who
simply gets no targets -- stays unknowable and stays the manager's risk. That
boundary is the point. Modelling those too would hand every policy information
no real manager has, and the environment would stop measuring anything.

**Byes are inferred from the panel rather than fetched.** A club's bye is the one
week in a season it contributes no rows at all, which the stacked pool already
answers offline for all thirty-two. :func:`bye_weeks` refuses to guess when a
season does not produce exactly one such week per club, because a silently wrong
bye is a permanently benched player.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# A club plays every week of the regular season but one.
SEASON_WEEKS = tuple(range(1, 19))

ACTIVE, OUT, BYE = "ACTIVE", "OUT", "BYE"


def bye_weeks(pool: pd.DataFrame) -> dict[tuple[int, str], int]:
    """``(season, club) -> bye week``, inferred from the week a club is absent.

    A club contributes rows every week it plays, so the one week it contributes
    none is its bye. Two cases are not byes and are handled rather than guessed:

    *No missing week* means no bye is observable -- a pool truncated before the
    club's bye, or a synthetic one. The club is simply left out of the map, and
    nobody on it is ever marked absent. Silence is the safe failure here.

    *More than one missing week* is genuinely ambiguous, and there is no way to
    tell a bye from a gap in the panel. That raises, because picking one would
    bench a player for a week his club really played, every season, invisibly.
    """
    out: dict[tuple[int, str], int] = {}
    for (season, club), block in pool.groupby(["season", "team"], sort=False):
        weeks = set(pd.to_numeric(block["week"], errors="coerce").dropna().astype(int))
        if not weeks:
            continue
        # Only inside the span the club is actually observed over, so a pool cut
        # at week 14 does not read weeks 15-18 as four more byes.
        missing = sorted(set(range(min(SEASON_WEEKS), max(weeks) + 1)) - weeks)
        if len(missing) > 1:
            raise ValueError(
                f"{club} in {season} is absent for {len(missing)} weeks "
                f"{missing}; a bye is exactly one. The panel has a gap, and "
                "guessing which is the bye would bench a player who played."
            )
        if missing:
            out[(int(season), str(club))] = int(missing[0])
    return out


@dataclass
class Availability:
    """What is known before kickoff about who can play, for one season.

    ``status(key, week)`` is the whole interface. It answers for every player in
    the pool, including those nobody has drafted, because the waiver wire needs
    the same answer: picking up a player to cover a bye, only to find he is on
    bye himself, is a mistake the environment should let a policy avoid.
    """

    season: int
    _bye: set[tuple[str, int]] = field(default_factory=set, repr=False)
    _out: set[tuple[str, int]] = field(default_factory=set, repr=False)

    def status(self, key: str, week: int) -> str:
        pair = (key, int(week))
        if pair in self._out:
            return OUT
        if pair in self._bye:
            return BYE
        return ACTIVE

    def is_available(self, key: str, week: int) -> bool:
        return self.status(key, week) == ACTIVE

    def is_out(self, key: str, week: int) -> bool:
        """Ruled out by the injury report, as distinct from idle on a bye.

        The distinction matters to the roster rules and nowhere else: a bye is
        one week and resolves itself, so it is a lineup problem. An injury is
        open-ended, which is what the IR slot exists for.
        """
        return (key, int(week)) in self._out

    def unavailable(self, keys, week: int) -> set[str]:
        """Which of ``keys`` are known not to be playing in ``week``."""
        return {key for key in keys if not self.is_available(key, week)}


def build_availability(pool: pd.DataFrame, season: int) -> Availability:
    """Read one season's known-unavailability out of the stacked pool.

    ``is_out`` is expected on the pool and is only ever populated for the skill
    positions: the injury report covers the players it covers, kickers are
    almost never ruled out, and a defense is a club rather than a person and can
    never be. Those are honest zeros, not gaps that need filling.
    """
    block = pool[pool["season"] == int(season)]
    if block.empty:
        raise ValueError(f"no pool rows for season {season}")

    byes = {club: week for (_, club), week in bye_weeks(block).items()}

    # Resolved per player-week rather than per player, because a player who
    # changes clubs mid-season sits out whichever bye his club at the time had
    # -- possibly both, possibly neither. Attributing him to one club for the
    # whole season gets that wrong in the weeks it matters, and gets it wrong
    # silently, which is how a permanently benched player happens.
    bye: set[tuple[str, int]] = set()
    ordered = block[["player_key", "week", "team"]].sort_values(["player_key", "week"])
    for key, rows in ordered.groupby("player_key", sort=False):
        weeks = rows["week"].astype(int).to_numpy()
        clubs = rows["team"].astype(str).to_numpy()
        observed = set(weeks.tolist())
        # Only inside the span he is observed over: before his first week and
        # after his last he is not on a bye, he is not in the league.
        for week in range(int(weeks.min()), int(weeks.max()) + 1):
            if week in observed:
                continue
            # The club he was last seen with going into that week, which is the
            # one whose bye he would have taken.
            before = weeks < week
            club = clubs[before][-1] if before.any() else clubs[0]
            if byes.get(str(club)) == week:
                bye.add((str(key), week))

    out: set[tuple[str, int]] = set()
    if "is_out" in block.columns:
        flagged = block[pd.to_numeric(block["is_out"], errors="coerce").fillna(0) == 1]
        out = set(zip(flagged["player_key"], flagged["week"].astype(int)))

    return Availability(season=int(season), _bye=bye, _out=out)
