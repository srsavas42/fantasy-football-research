"""League shape: how many teams, which slots, what a win is worth.

Every number here is a default rather than a fact about football, so every one
is tunable. The defaults describe the common twelve-team league: one quarterback,
two backs, two receivers, a tight end, a flex, a kicker, a defense, and six on
the bench.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# What the flex will accept. Kickers and defenses are deliberately absent: no
# mainstream league lets a defense fill a flex, and allowing it would let a
# policy exploit the environment in a way that would not transfer.
FLEX_POSITIONS = ("RB", "WR", "TE")

# Every position the league drafts. Ordered as a lineup card reads.
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")


@dataclass(frozen=True)
class RosterSlots:
    """Starting slots, plus the bench.

    ``flex`` draws from :data:`FLEX_POSITIONS`. The bench has no position
    requirement -- it is depth, and what a manager keeps there is one of the
    decisions being scored.
    """

    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 1
    k: int = 1
    dst: int = 1
    bench: int = 6

    @property
    def starters(self) -> int:
        return self.qb + self.rb + self.wr + self.te + self.flex + self.k + self.dst

    @property
    def size(self) -> int:
        """Total roster spots, starters plus bench."""
        return self.starters + self.bench

    def dedicated(self) -> dict[str, int]:
        """Position-locked starting slots, excluding the flex."""
        return {
            "QB": self.qb,
            "RB": self.rb,
            "WR": self.wr,
            "TE": self.te,
            "K": self.k,
            "DST": self.dst,
        }


@dataclass(frozen=True)
class LeagueConfig:
    """One league's rules.

    ``win_bonus`` is the whole reason this is a league and not a projection
    scoreboard. Rewarding a point per fantasy point alone would make the
    environment a points-maximisation problem, which is *nearly* the right
    objective and not the same one: a manager who wins 100-40 wasted sixty
    points that a manager who wins 70-69 did not, and both took the same thing
    from the season. The bonus is what makes the opponent's score matter. It is
    tunable precisely because the balance between the two is a modelling choice
    rather than a known constant, and a bonus large enough to dominate turns the
    signal sparse and hard to learn from.
    """

    teams: int = 12
    slots: RosterSlots = field(default_factory=RosterSlots)
    seasons: tuple[int, ...] = ()
    # Regular-season fantasy weeks. Most leagues stop before the NFL does,
    # because week 18 rests starters and is not worth simulating as though it
    # were a normal week.
    first_week: int = 1
    last_week: int = 14

    # Reward shaping.
    points_weight: float = 1.0
    win_bonus: float = 50.0
    tie_bonus: float = 25.0

    # Waivers. A cap rather than a bidding market: FAAB is a second learning
    # problem stacked on the first, and the point here is start/sit and add/drop.
    waiver_adds_per_week: int = 1
    # How many free agents a policy is shown. The pool is hundreds of players
    # deep and almost all of it is noise; a policy that must rank every one is
    # solving a harder problem than a manager does with a sorted waiver page.
    waiver_shortlist: int = 25

    def __post_init__(self) -> None:
        if self.teams < 2:
            raise ValueError("a league needs at least two teams")
        if self.last_week < self.first_week:
            raise ValueError("last_week cannot precede first_week")
        # A draft that cannot fill every roster is a silent source of empty
        # lineup slots later, so it is caught here instead.
        if self.teams * self.slots.size > 400:
            raise ValueError(
                f"{self.teams} teams x {self.slots.size} spots exceeds the "
                "drafted population any season provides"
            )

    @property
    def weeks(self) -> tuple[int, ...]:
        return tuple(range(self.first_week, self.last_week + 1))

    @property
    def roster_spots(self) -> int:
        return self.teams * self.slots.size
