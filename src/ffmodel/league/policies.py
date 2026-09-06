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

    # The board is fixed for a season but `score` is called thousands of times a
    # season, and re-indexing a frame on every call was the single largest cost
    # left once the averages were precomputed. Keyed on the board's identity so
    # a different board is not silently answered from the last one's cache.
    _ranks: dict = field(default_factory=dict, repr=False, compare=False)

    def _lookup(self, board) -> dict[str, float]:
        cached = self._ranks.get(id(board))
        if cached is None:
            cached = (
                dict(zip(board["player_key"], board["adp_rank"]))
                if len(board)
                else {}
            )
            self._ranks.clear()
            self._ranks[id(board)] = cached
        return cached

    def score(self, player_keys, history, week, board) -> dict[str, float]:
        ranks = self._lookup(board)
        out = {}
        for key in player_keys:
            rank = ranks.get(key, np.nan)
            rank = UNRANKED if rank is None or not np.isfinite(rank) else float(rank)
            # Invert: a low pick is a high score. Reciprocal rather than
            # negation so the gap between the 1st and 10th pick is larger than
            # between the 101st and 110th, which is how draft value behaves.
            out[key] = 1.0 / rank
        return out


# Which weeks of a player's history the average is taken over.
#
# "all"        every week he was on an NFL roster, zeros included.
# "active"     only weeks he recorded a stat line.
# "available"  every week except the ones the environment already handles --
#              the game-status report ruled him out, or his club was on bye.
HISTORY_MODES = ("all", "active", "available")


@dataclass
class EwmaPolicy(Policy):
    """Start whoever has been scoring, exponentially weighted.

    ``history_mode`` decides which weeks the average sees, and the choice is not
    cosmetic -- it is worth roughly a quarter of a win a season. The argument for
    narrowing it: absence is no longer the average's job, because the environment
    states who is on bye and who has been ruled out and
    :mod:`ffmodel.league.roster` benches them before the card is set. Counting
    those weeks again inside the average is double-counting, and it is what makes
    a returning starter look unstartable in the week he comes back.

    That argument is right about the weeks it names and wrong about how many
    weeks those are. ``"active"`` drops every week without a stat line, and among
    draftable players only about a quarter of those were ever flagged Out. The
    rest are healthy scratches, in-game injuries, and starters who simply drew no
    targets -- absence and failure nobody could see coming, which is real risk
    and belongs in the projection. Dropping it measures "what is he worth when he
    produces", which is a different and more flattering question than the one a
    lineup decision asks. Measured in the league, ``"active"`` costs 0.32 wins
    and 23 points a season against ``"all"``.

    ``"available"`` is the version that survives the objection: it removes
    exactly the weeks the environment already accounts for -- ruled out, or on
    bye -- and keeps every week the player was there and did nothing. That is the
    double-counting argument applied only where it holds.
    """

    halflife: float = EWMA_HALFLIFE
    name: str = "ewma"
    fallback_to_board: bool = True
    history_mode: str = "all"
    _board: AdpPolicy = field(default_factory=AdpPolicy, repr=False, compare=False)
    # A precomputed table from :mod:`ffmodel.league.features`, which holds the
    # same averages this would derive from the history frame and is what makes
    # an episode fast enough to train against. Optional and equivalent: with it
    # or without it the policy returns the same numbers, which
    # `verify_against_history` checks to the float.
    table: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    # The average depends on the week, not on whose roster is being scored, but
    # the environment calls this once per team -- twelve times a week for the
    # same numbers. Caching on the week turns the dominant cost of a season
    # into one computation instead of twelve.
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.history_mode not in HISTORY_MODES:
            raise ValueError(
                f"history_mode {self.history_mode!r} is not one of {HISTORY_MODES}"
            )

    def _weeks_counted(self, history: pd.DataFrame) -> pd.DataFrame:
        """The rows the average is taken over, under the configured mode."""
        if self.history_mode == "active" and "played" in history.columns:
            return history[history["played"] == 1]
        if self.history_mode == "available" and "is_out" in history.columns:
            # A bye contributes no row at all, so it is already excluded and only
            # the game-status report needs filtering here.
            return history[history["is_out"] != 1]
        return history

    def _from_table(self, week: int) -> pd.Series:
        """The week's averages, read rather than derived."""
        column = f"ewma{self.halflife:g}"
        try:
            block = self.table.xs(int(week), level="week")
        except KeyError:
            return pd.Series(dtype=float)
        # A player in his first week of the season has no past, and the frame
        # version says so by having nothing to average. The table fills those
        # rows with zeros for the learner's benefit, so they are masked back to
        # missing here -- otherwise everyone would score zero in week 1 and the
        # board fallback, which is the only thing with an opinion that early,
        # would never fire.
        return block[column].where(block["experience"] > 0.0).to_dict()

    def _averages(self, history: pd.DataFrame, week: int) -> pd.Series:
        cached = self._cache.get(week)
        if cached is not None:
            return cached
        if self.table is not None:
            if self.history_mode != "all":
                raise ValueError(
                    "the precomputed table holds the 'all' history mode only; "
                    f"this policy is set to {self.history_mode!r}"
                )
            averages = self._from_table(week)
            self._cache.clear()
            self._cache[week] = averages
            return averages
        history = self._weeks_counted(history)
        if not len(history):
            averages = {}
        else:
            alpha = 1.0 - 0.5 ** (1.0 / self.halflife)
            played = history.sort_values("week")
            averages = (
                played.groupby("player_key")["points"]
                .apply(lambda s: s.ewm(alpha=alpha, adjust=True).mean().iloc[-1])
                .to_dict()
            )
        # Keyed on the week alone, so a fresh episode must not inherit the last
        # one's numbers: the environment builds a new policy per episode, and
        # this guards the case where it does not.
        self._cache.clear()
        self._cache[week] = averages
        return averages

    def score(self, player_keys, history, week, board) -> dict[str, float]:
        out: dict[str, float] = {}
        averages = self._averages(history, week)

        board_scores = (
            self._board.score(player_keys, history, week, board)
            if self.fallback_to_board
            else {}
        )
        for key in player_keys:
            value = averages.get(key, np.nan)
            if value is not None and np.isfinite(value):
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
    history_mode: str = "all"
    table: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    board_policy: AdpPolicy = field(default_factory=AdpPolicy)
    form_policy: EwmaPolicy | None = None

    def __post_init__(self) -> None:
        if self.form_policy is None:
            self.form_policy = EwmaPolicy(
                halflife=self.halflife,
                history_mode=self.history_mode,
                table=self.table,
            )

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
