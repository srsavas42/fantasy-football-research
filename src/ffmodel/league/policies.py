"""How a team decides who to start, and who to pick up.

Every policy has the same shape: given what has already happened, score each
player. :mod:`ffmodel.league.lineup` turns those scores into a legal lineup, and
the environment turns the lineup into points. A policy never sees a future week
-- the environment hands it a history frame that has already been truncated, and
the policies here only ever read from what they are given.

The two that matter most are the opponents, because they set the bar the whole
environment is measured against:

:class:`AdpPolicy`
    Start the players the preseason board liked. What a manager does in week 1
    because there is nothing else to go on.

:class:`EwmaPolicy`
    Start the players who have been scoring. An exponentially weighted average
    of what each rostered player has done so far.

:class:`SeasonPolicy`
    The two spliced: the board early, the average once there is enough season to
    average. This is the standard opponent, and the splice week is tunable
    because "how long before recent form beats the draft board" is exactly the
    kind of thing the weekly layer has an opinion about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Half-life on a player's own scoring history, in games. One game matches the
# weekly feature layer's selected decay, so the naive opponent here is the same
# heuristic the model's own `recency-mean` rung uses -- which makes "does the
# model beat an EWMA" a question this environment can actually answer.
EWMA_HALFLIFE = 1.0

# A rank that means "the board did not have him". Large enough to sort below
# every ranked player without being infinite, which would poison an average.
UNRANKED = 999.0


class Policy:
    """Score each of a team's players for one week.

    ``history`` contains only weeks strictly before the one being decided; the
    environment guarantees that, and a policy that reaches around it is a bug
    rather than a clever feature.
    """

    name: str = "policy"

    def score(
        self,
        player_keys: list[str],
        history: pd.DataFrame,
        week: int,
        board: pd.DataFrame,
    ) -> dict[str, float]:
        raise NotImplementedError


@dataclass
class AdpPolicy(Policy):
    """Start by preseason consensus. Never learns anything."""

    name: str = "adp"

    def score(self, player_keys, history, week, board) -> dict[str, float]:
        ranks = board.set_index("player_key")["adp_rank"] if len(board) else pd.Series(dtype=float)
        out = {}
        for key in player_keys:
            rank = ranks.get(key, np.nan)
            rank = UNRANKED if not np.isfinite(rank) else float(rank)
            # Invert: a low pick is a high score. Reciprocal rather than
            # negation so the gap between the 1st and 10th pick is larger than
            # between the 101st and 110th, which is how draft value behaves.
            out[key] = 1.0 / rank
        return out


@dataclass
class EwmaPolicy(Policy):
    """Start whoever has been scoring, exponentially weighted.

    Weeks a player did not play are included as zeros, because that is what a
    manager's own memory of "he's been quiet" contains, and excluding them would
    make an injured player look like a must-start the week he returns.
    """

    halflife: float = EWMA_HALFLIFE
    name: str = "ewma"
    fallback_to_board: bool = True

    def score(self, player_keys, history, week, board) -> dict[str, float]:
        alpha = 1.0 - 0.5 ** (1.0 / self.halflife)
        out: dict[str, float] = {}
        if len(history):
            played = history.sort_values("week")
            grouped = played.groupby("player_key")["points"]
            averages = grouped.apply(
                lambda s: s.ewm(alpha=alpha, adjust=True).mean().iloc[-1]
            )
        else:
            averages = pd.Series(dtype=float)

        board_scores = (
            AdpPolicy().score(player_keys, history, week, board)
            if self.fallback_to_board
            else {}
        )
        for key in player_keys:
            value = averages.get(key, np.nan)
            if np.isfinite(value):
                out[key] = float(value)
            else:
                # No history at all -- a rookie, or somebody just picked up.
                # The board is the only thing left to rank him by, scaled down
                # so a ranked-but-unseen player does not outrank a producing one.
                out[key] = board_scores.get(key, 0.0)
        return out


@dataclass
class SeasonPolicy(Policy):
    """The board early, recent form later. The standard opponent.

    ``switch_week`` is the first week decided by the average rather than the
    board. The default of 4 means weeks 1-3 are drafted-team autopilot, which is
    both what the environment was specified to do and roughly where the weekly
    layer's own measurements put the crossover: the draft board is genuinely
    good in September and decays from there.
    """

    switch_week: int = 4
    halflife: float = EWMA_HALFLIFE
    name: str = "adp-then-ewma"

    board_policy: AdpPolicy = field(default_factory=AdpPolicy)
    form_policy: EwmaPolicy | None = None

    def __post_init__(self) -> None:
        if self.form_policy is None:
            self.form_policy = EwmaPolicy(halflife=self.halflife)

    def score(self, player_keys, history, week, board) -> dict[str, float]:
        if week < self.switch_week or history.empty:
            return self.board_policy.score(player_keys, history, week, board)
        return self.form_policy.score(player_keys, history, week, board)


@dataclass
class ProjectionPolicy(Policy):
    """Start by a supplied projection: one row per player-week.

    This is how the shipped weekly model enters the environment. The frame is
    indexed on ``(player_key, week)`` and is expected to have been produced by a
    walk-forward fit, so the projection for week `w` was made without week `w`.
    Nothing here can verify that -- it is a property of how the frame was built
    -- so the caller owns it, and :mod:`ffmodel.league.env` says so where the
    projection is passed in.
    """

    projections: pd.DataFrame
    name: str = "projection"
    fallback: Policy | None = None

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = EwmaPolicy()
        frame = self.projections
        needed = {"player_key", "week", "projection"}
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"projection frame missing {sorted(missing)}")
        self._lookup = frame.set_index(["player_key", "week"])["projection"]

    def score(self, player_keys, history, week, board) -> dict[str, float]:
        backup = self.fallback.score(player_keys, history, week, board)
        out = {}
        for key in player_keys:
            try:
                value = self._lookup.get((key, week), np.nan)
            except (KeyError, TypeError):
                value = np.nan
            out[key] = float(value) if np.isfinite(value) else backup.get(key, 0.0)
        return out


@dataclass
class PerfectPolicy(Policy):
    """Starts the players who actually scored. The ceiling, not a competitor.

    Deliberately provided: an environment where nobody knows how much headroom
    exists is one where a small win is indistinguishable from a large one. This
    is the only policy allowed to read the week being decided, and it exists to
    put a number on what a perfect start/sit would have been worth.
    """

    truth: pd.DataFrame
    name: str = "oracle"

    def __post_init__(self) -> None:
        self._lookup = self.truth.set_index(["player_key", "week"])["points"]

    def score(self, player_keys, history, week, board) -> dict[str, float]:
        out = {}
        for key in player_keys:
            try:
                value = self._lookup.get((key, week), 0.0)
            except (KeyError, TypeError):
                value = 0.0
            out[key] = float(value) if np.isfinite(value) else 0.0
        return out
