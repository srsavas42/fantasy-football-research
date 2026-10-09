"""Rest-of-season points as expected games times points per game played.

Every row of the rest-of-season output is one calculation, not three::

    rest_of_season_points  =  expected_games_played  x  points_per_active_game

``points_per_scheduled_game`` is the same total over the games on the schedule,
and p10/p50/p90 are quantiles of the same total. Each column is a different way of
breaking down one number, so none of them can disagree with another.

That is a change from the first version, whose three numbers came from three
separately fitted models and did not multiply out. The total was blended toward a
draft-board curve that knows nothing about a player's absences; the games count was
discounted for exactly those absences; the rate was a third model. They disagreed,
and they disagreed most for the players the decision is about -- a star who has just
missed time. ``docs/ros-reconciliation-2026-10.md`` has the measurements.

**Why the rate is weighted by games played.** Fitting the plain average of points
per game played, and multiplying by expected games, leaves a large gap, and a
correction factor cannot honestly close it (one fitted on returners came out at
1.27-1.36, and folding it into the games made "expected games" 0.55 of the games
left against an actual 0.45). The cause is that a total is a sum over games, so the
rate that multiplies out exactly is the *games-weighted* one, E[games x rate] /
E[games]. A player who plays more games is usually one whose role held up, so the
two differ. Fitting the rate by weighted least squares, with each row weighted by
the games he actually played, removes the overall bias with no correction at all.

**Expected games is fitted separately for each way of having just missed a game.**
"Missed last game" lumps together an injured-reserve placement, a designated Out, a
Questionable who sat, a healthy scratch and a player on the field with no stat line,
and their futures have nothing in common. Reserve-list players play 18% of what
remains, players who were Out last week 63%. One linear fit can only shift an
intercept between them; separate fits let each have its own slopes. The groups are
crossed with how many games he has missed in a row, and each carries the injury
mechanism on the report (:mod:`ffmodel.weekly.injury_type`).

**An era term.** Return-to-play rates stepped up around 2021: players who had missed
three to five games in a row played 0.21-0.30 of the remaining games in 2016-2020
and 0.33-0.37 from 2021. A fit over all years averages the two and under-projects
returners in every recent season. ``era21`` lets the level differ. Every walk-forward
holdout is in the newer era, which flatters the fix; for a live season it matters
because half the training years are in the older one. Why the step happened is not
known.

**Intervals** are empirical. A composed variance (a Beta for availability, a level
SD for scoring, a noise scale backed out of the rest) came out too narrow and the
wrong shape, and no single width corrects that -- see
``docs/hierarchical-calibration-2026-09.md``. Here the residuals of the *total* are
measured on the two most recent complete seasons, each predicted by a model fitted
before it, and binned by the size of the projection and by whether he just missed a
game and was drafted. p10/p50/p90 are the projection plus those residual quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.weekly.availability_rate import (
    GAMES_PLAYED_REST,
    RATE_FEATURES,
    RATE_TARGET,
    TARGET as SHARE,
    add_played_rate_target,
    add_points_per_active_game_target,
)
from ffmodel.weekly.features import relevant_population
from ffmodel.weekly.fitting import Ridge
from ffmodel.weekly.injury_type import mechanism_columns
from ffmodel.weekly.partial_game import add_partial_game
from ffmodel.weekly.restofseason import OFFSET, RIDGE_PENALTY, TARGET, DirectTotal

#: First season of the newer return-to-play regime.
ERA_START = 2021

#: Fewest training rows a group needs before it gets its own fit.
MIN_GROUP_ROWS = 150

#: Fewest residuals a bin needs before it is used on its own.
MIN_POOL_ROWS = 120

QUANTILE_LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)
_P10, _P50, _P90 = 9, 49, 89

_EXTRA = ("missed_last", "adp_log_rank", "adp_drafted", "prior_points_given_played", "depth_rank")
_INTERACTIONS = ("adp_log_rank", "prior_points_given_played", "prior_snap_share_recent")
_STREAK = ("missed_n", "log_missed")

#: Whether the previous game was cut short (he played, on far fewer snaps than usual),
#: and how many of the last three were. See :mod:`ffmodel.weekly.partial_game`. The
#: weeks 1-2 depth-chart label (``partial_prev_early``) is left out: it predicts the next
#: game and not the rest of the season (``scripts/validate_partial_game.py``).
PARTIAL_FEATURES = ("partial_prev", "partial_recent")

#: Fit "left early last game" players as a group of their own, as the reserve list and
#: designated Out already are, rather than only shifting the intercept of the group
#: that played last. Measured best of four arms; see ``scripts/validate_partial_game.py``.
PARTIAL_GROUP = True


def _lagged(frame: pd.DataFrame, column: str) -> pd.Series:
    """The previous game's value, per player and season, on the frame's own index."""
    order = frame.sort_values(["player_key", "season", "week"], kind="mergesort").index
    lag = frame.loc[order].groupby(["player_key", "season"], sort=False)[column].shift(1)
    return lag.reindex(frame.index)


def add_absence_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Why and for how long he has been out, from what the rows already carry.

    ``missed_last`` is whether the previous game was a zero. The four ``*_prev``
    flags say *why*: on the reserve list, designated Out, Questionable or
    Doubtful and did not play, or inactive with no designation. ``missed_n`` is
    games missed in a row, counted in club games and carried across seasons.
    """
    out = frame.copy()
    played_prev = _lagged(out, "played")
    out["missed_last"] = (1.0 - pd.to_numeric(played_prev, errors="coerce")).fillna(0.0)
    status_prev = _lagged(out, "status")
    injury_prev = pd.to_numeric(
        out["inj_status_lagged"] if "inj_status_lagged" in out else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0)
    out["res_prev"] = status_prev.eq("RES").astype(float)
    out["out_prev"] = (injury_prev >= 3).astype(float)
    out["qd_prev"] = ((injury_prev >= 1) & (injury_prev < 3)).astype(float)
    out["ina_prev"] = status_prev.eq("INA").astype(float)
    streak = pd.to_numeric(out["weeks_since_played"], errors="coerce") - 1.0
    out["missed_n"] = streak.clip(lower=0.0).fillna(0.0)
    out["log_missed"] = np.log1p(out["missed_n"])
    out["era21"] = (pd.to_numeric(out["season"], errors="coerce") >= ERA_START).astype(float)
    return add_partial_game(out)


def _games_design(frame: pd.DataFrame, medians: pd.Series | None = None):
    extra = _EXTRA + mechanism_columns() + _STREAK + ("era21",) + PARTIAL_FEATURES
    columns = [c for c in RATE_FEATURES if c in frame.columns] + [c for c in extra if c in frame.columns]
    block = frame[columns].apply(pd.to_numeric, errors="coerce")
    for name in _INTERACTIONS:
        if name in block:
            block[f"missed_x_{name}"] = block["missed_last"] * block[name]
    if medians is None:
        medians = block.median()
    return block.fillna(medians), medians


def _absence_groups(frame: pd.DataFrame) -> np.ndarray:
    """Why he is out, and for players who just missed a game, for how long."""
    reason = np.zeros(len(frame), dtype=int)
    missed = frame["missed_last"].to_numpy() == 1
    reason[missed] = 5
    reason[missed & (frame["ina_prev"].to_numpy() == 1)] = 4
    reason[missed & (frame["qd_prev"].to_numpy() == 1)] = 3
    reason[missed & (frame["out_prev"].to_numpy() == 1)] = 2
    reason[missed & (frame["res_prev"].to_numpy() == 1)] = 1
    n = frame["missed_n"].to_numpy()
    length = np.select([n <= 1, n == 2, n <= 4], [0, 1, 2], default=3)
    # Reserve-list and inactive players are one group each: the reason is most of
    # the story. The rest split by how long the streak has run.
    groups = np.where(np.isin(reason, (2, 3, 5)), 10 * reason + length, reason)
    if PARTIAL_GROUP and "partial_prev" in frame.columns:
        # He played last game but was cut short: not absent, and not a normal game.
        groups = np.where((reason == 0) & (frame["partial_prev"].to_numpy() == 1), 6, groups)
    return groups


@dataclass
class ExpectedGames:
    """Share of his club's remaining games he plays, fitted per absence group."""

    models: dict = field(default_factory=dict)
    fallback: Ridge | None = None
    columns: list = field(default_factory=list)
    medians: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "ExpectedGames":
        usable = frame[frame[SHARE].notna()]
        design, self.medians = _games_design(usable)
        self.columns = list(design.columns)
        x, y = design.to_numpy(float), usable[SHARE].to_numpy(float)
        groups = _absence_groups(usable)
        self.models = {
            int(g): Ridge.fit(x[groups == g], y[groups == g], penalty=1.0)
            for g in np.unique(groups)
            if (groups == g).sum() >= MIN_GROUP_ROWS
        }
        self.fallback = Ridge.fit(x, y, penalty=1.0)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        design, _ = _games_design(frame, self.medians)
        x = design[self.columns].to_numpy(float)
        groups = _absence_groups(frame)
        out = self.fallback.predict(x)
        for g, model in self.models.items():
            hit = groups == g
            if hit.any():
                out[hit] = model.predict(x[hit])
        # A floor, not zero: a player nobody expects back still gets a rate that
        # multiplies to something small rather than to nothing.
        return np.clip(out, 0.05, 1.0)


@dataclass
class WeightedRate:
    """Points per game played, weighted by the games he played.

    Same design as the direct rest-of-season regression. The weights are what make
    the rate multiply out to the total; see the module docstring.
    """

    base: DirectTotal = field(
        default_factory=lambda: DirectTotal(use_team=True, use_phase=True, use_adp=True, use_role=True)
    )
    centre: np.ndarray | None = None
    scale: np.ndarray | None = None
    beta: np.ndarray | None = None
    intercept: float = 0.0

    def fit(self, frame: pd.DataFrame) -> "WeightedRate":
        self.base.medians = None
        x = self.base._design(frame)
        y = frame[RATE_TARGET].to_numpy(float)
        w = frame[GAMES_PLAYED_REST].to_numpy(float)
        ok = np.isfinite(y) & (w > 0)
        x, y, w = x[ok], y[ok], w[ok] / w[ok].mean()  # mean one keeps the ridge penalty's scale
        self.centre = np.average(x, axis=0, weights=w)
        self.scale = np.sqrt(np.average((x - self.centre) ** 2, axis=0, weights=w))
        self.scale[self.scale == 0] = 1.0
        z = (x - self.centre) / self.scale
        self.intercept = float(np.average(y, weights=w))
        gram = z.T @ (w[:, None] * z) + RIDGE_PENALTY * np.eye(z.shape[1])
        self.beta = np.linalg.solve(gram, z.T @ (w * (y - self.intercept)))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.base._design(frame)
        return ((x - self.centre) / self.scale) @ self.beta + self.intercept


@dataclass
class ResidualQuantiles:
    """Quantiles of (actual - projected total), by projection size and absence state."""

    edges: np.ndarray
    table: dict

    @staticmethod
    def _group(missed_last, drafted) -> np.ndarray:
        return (np.asarray(missed_last) == 1).astype(int) * 2 + (np.asarray(drafted) == 1).astype(int)

    @classmethod
    def build(cls, total, actual, missed_last, drafted) -> "ResidualQuantiles":
        total, actual = np.asarray(total, float), np.asarray(actual, float)
        edges = np.unique(np.quantile(total, np.linspace(0.0, 1.0, 11)))
        bins = np.clip(np.searchsorted(edges, total, side="right") - 1, 0, len(edges) - 2)
        groups = cls._group(missed_last, drafted)
        resid = actual - total
        table = {}
        for b in range(len(edges) - 1):
            for g in range(4):
                pool = resid[(bins == b) & (groups == g)]
                if len(pool) < MIN_POOL_ROWS:  # same returner flag, any draft status
                    pool = resid[(bins == b) & (groups // 2 == g // 2)]
                if len(pool) < MIN_POOL_ROWS:
                    pool = resid[bins == b]
                table[(b, g)] = np.quantile(pool, QUANTILE_LEVELS)
        return cls(edges=edges, table=table)

    def quantiles(self, total, missed_last, drafted) -> np.ndarray:
        """(rows, 99) grid of the total's quantiles, sorted and never below zero."""
        total = np.asarray(total, float)
        bins = np.clip(np.searchsorted(self.edges, total, side="right") - 1, 0, len(self.edges) - 2)
        groups = self._group(missed_last, drafted)
        out = np.empty((len(total), len(QUANTILE_LEVELS)))
        for i in range(len(total)):
            out[i] = total[i] + self.table[(int(bins[i]), int(groups[i]))]
        return np.sort(np.maximum(out, 0.0), axis=1)


def summarise(grid: np.ndarray) -> dict[str, np.ndarray]:
    """p10, p50 and p90 from a quantile grid."""
    return {"p10": grid[:, _P10], "p50": grid[:, _P50], "p90": grid[:, _P90]}


@dataclass
class ReconciledROS:
    """Expected games x points per game played, with empirical intervals."""

    games: ExpectedGames = field(default_factory=ExpectedGames)
    rate: WeightedRate = field(default_factory=WeightedRate)
    intervals: ResidualQuantiles | None = None

    @staticmethod
    def _targets(frame: pd.DataFrame) -> pd.DataFrame:
        return add_points_per_active_game_target(add_played_rate_target(frame))

    def fit(self, frame: pd.DataFrame) -> "ReconciledROS":
        """Fit on rows that carry the rest-of-season target and the absence state.

        ``frame`` must already have :func:`add_absence_state` applied and the
        injury type attached; both need history that only the whole frame has.
        """
        frame = self._targets(frame)
        self.games.fit(frame)
        self.rate.fit(frame)
        self.intervals = None
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Expected games, rate and their product, on the frame's own index."""
        left = pd.to_numeric(frame[OFFSET], errors="coerce").fillna(0.0).to_numpy(float)
        expected = left * self.games.predict(frame)
        rate = self.rate.predict(frame)
        return pd.DataFrame(
            {"expected_games": expected, "rate": rate, "total": expected * rate}, index=frame.index
        )

    def fit_intervals(self, frame: pd.DataFrame, *, inner: int = 2) -> "ReconciledROS":
        """Measure the total's residuals on the last ``inner`` complete seasons.

        Each is predicted by a model fitted strictly before it, so the residuals
        are out of sample and the season being forecast never informs its own
        intervals.
        """
        seasons = sorted(int(s) for s in frame["season"].unique())
        pooled = []
        for season in seasons[-inner:]:
            earlier = frame[frame["season"] < season]
            block = frame[frame["season"] == season]
            block = block[relevant_population(block).to_numpy(bool)]
            model = ReconciledROS().fit(earlier)
            predicted = model.predict(block)
            pooled.append(
                pd.DataFrame(
                    {
                        "total": predicted["total"].to_numpy(),
                        "actual": pd.to_numeric(block[TARGET], errors="coerce").to_numpy(),
                        "missed_last": block["missed_last"].to_numpy(),
                        "drafted": pd.to_numeric(block["adp_drafted"], errors="coerce").fillna(0).to_numpy(),
                    }
                )
            )
        both = pd.concat(pooled, ignore_index=True).dropna()
        self.intervals = ResidualQuantiles.build(
            both["total"], both["actual"], both["missed_last"], both["drafted"]
        )
        return self

    def quantile_grid(self, frame: pd.DataFrame, total) -> np.ndarray:
        if self.intervals is None:
            raise RuntimeError("fit_intervals first")
        return self.intervals.quantiles(
            total,
            frame["missed_last"].to_numpy(),
            pd.to_numeric(frame["adp_drafted"], errors="coerce").fillna(0).to_numpy(),
        )
