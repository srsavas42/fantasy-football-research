"""A learned policy: one small parameter vector for lineups and waivers.

The environment asks a policy for one number per rostered player and turns those
numbers into a lineup itself. That shape decides what a learned policy has to be:
not a lineup generator, but a **ranking function**. So the agent here is a linear
score over the per-player features in :mod:`ffmodel.league.features`, and
learning is finding the weights.

Small on purpose. The entire prize is 2.58 wins of 14 against a seed noise of
plus or minus three wins a season, so the signal-to-noise ratio is brutal and the
number of parameters that can be fit before the search is just memorising seeds
is small. Fifteen weights and a threshold is a budget the measurement can
actually support; a network would fit the draft order.

**The waiver decision shares the weights.** A claim is worth making when the best
free agent is worth more than the worst player the roster can spare, and "worth
more" is the same judgement the lineup makes -- so it is the same function, with
one extra parameter for how big the gap has to be before acting. Learning them
jointly is also what stops the agent claiming a player it would never start.

The features are standardised with statistics frozen from the training seasons,
which is what keeps a weight comparable between them and the holdout. The
constant and the position indicators are left alone: they are already on a fixed
scale, and standardising a column that never varies produces a division by zero
rather than a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.league.context import CONTEXT_COLUMNS, build_context, team_shortfalls
from ffmodel.league.features import FEATURE_COLUMNS, as_matrix
from ffmodel.league.lineup import optimal_lineup
from ffmodel.league.policies import Policy

# Columns already on a fixed, interpretable scale, which are passed through.
UNSCALED = ("bias",) + tuple(c for c in FEATURE_COLUMNS if c.startswith("is_"))

# Weights, plus the waiver threshold. A split adds one delta per named feature.
PARAMETER_COUNT = len(FEATURE_COLUMNS) + 1


def parameter_count(split: tuple[str, ...] = (), context: bool = False) -> int:
    return PARAMETER_COUNT + len(split) + (len(CONTEXT_COLUMNS) if context else 0)


def _split_indices(split: tuple[str, ...]) -> list[int]:
    for name in split:
        if name not in FEATURE_COLUMNS:
            raise ValueError(f"unknown feature {name!r}")
    return [FEATURE_COLUMNS.index(name) for name in split]


@dataclass
class Scaler:
    """Per-feature centring and scaling, frozen from the training seasons."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, tables: dict) -> "Scaler":
        stacked = np.concatenate(
            [table[list(FEATURE_COLUMNS)].to_numpy(float) for table in tables.values()]
        )
        mean = stacked.mean(axis=0)
        scale = stacked.std(axis=0)
        for index, column in enumerate(FEATURE_COLUMNS):
            if column in UNSCALED:
                mean[index], scale[index] = 0.0, 1.0
        # A column that never varies would divide by zero; leaving it alone is
        # the only sane answer and costs nothing, since it carries no signal.
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Scaler":
        return cls(mean=np.asarray(data["mean"], float), scale=np.asarray(data["scale"], float))


@dataclass
class LinearAgent(Policy):
    """Rank players by a linear score, and claim when the gap is worth it."""

    weights: np.ndarray
    claim_threshold: float
    table: pd.DataFrame = field(repr=False)
    scaler: Scaler = field(repr=False)
    name: str = "learned"
    # Features the add/drop decision is allowed to weigh differently from the
    # lineup. Empty means one ranking for both, which is the default and the
    # reason the agent cannot claim a player it would never start.
    #
    # The case for allowing a difference is that the two decisions have
    # different horizons: a lineup is a question about Sunday and a claim is a
    # question about the rest of the season. Shared weights must compromise
    # between them, and the compromise is optimal for neither. Carried as a
    # *delta* on the shared weight rather than a second vector, so zero recovers
    # the shared model exactly and the search only has to find departures from
    # it.
    split: tuple[str, ...] = ()
    waiver_delta: np.ndarray | None = field(default=None, repr=False)
    # Weights on the decision-time context block in :mod:`ffmodel.league.context`
    # -- roster depth, the upgrade over the man actually displaced, how
    # replaceable the player is on the wire, and what the rest of the league is
    # short of. They apply to the acquisition decision only, because they are
    # answers to "is he worth more to me than the alternative" and a lineup has
    # no alternative to weigh: the roster is already what it is.
    context_weights: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_parameters(
        cls,
        theta: np.ndarray,
        table: pd.DataFrame,
        scaler: Scaler,
        split: tuple[str, ...] = (),
        context: bool = False,
        **kwargs,
    ) -> "LinearAgent":
        theta = np.asarray(theta, float)
        expected = parameter_count(split, context)
        if theta.shape != (expected,):
            raise ValueError(f"expected {expected} parameters, got {theta.shape}")
        cut = len(FEATURE_COLUMNS)
        rest = theta[cut + 1 :]
        deltas = rest[: len(split)]
        weights = rest[len(split) :] if context else None
        return cls(
            weights=theta[:cut],
            claim_threshold=float(theta[cut]),
            table=table,
            scaler=scaler,
            split=tuple(split),
            waiver_delta=deltas,
            context_weights=weights,
            **kwargs,
        )

    def parameters(self) -> np.ndarray:
        parts = [self.weights, [self.claim_threshold]]
        if self.waiver_delta is not None:
            parts.append(self.waiver_delta)
        if self.context_weights is not None:
            parts.append(self.context_weights)
        return np.concatenate([np.asarray(part, float) for part in parts])

    @property
    def waiver_weights(self) -> np.ndarray:
        """The weights the add/drop decision ranks by."""
        if not self.split or self.waiver_delta is None or not len(self.waiver_delta):
            return self.weights
        weights = self.weights.copy()
        for index, delta in zip(_split_indices(self.split), self.waiver_delta):
            weights[index] += delta
        return weights

    def values(self, keys, week: int, waiver: bool = False) -> dict[str, float]:
        keys = list(keys)
        if not keys:
            return {}
        rows = self.scaler.apply(as_matrix(self.table, keys, week))
        weights = self.waiver_weights if waiver else self.weights
        return dict(zip(keys, rows @ weights))

    # ------------------------------------------------------------- lineups

    def score(self, player_keys, history, week, board, state=None) -> dict[str, float]:
        return self.values(player_keys, week)

    # ------------------------------------------------------------- waivers

    def claim(self, env, observation):
        """Swap the worst spare player for the best free agent, if it is worth it.

        A starter is never offered up. Cutting somebody the agent itself would
        put in this week's lineup to make room for a free agent is not a trade a
        threshold should be able to authorise, and letting the search discover
        that costs generations it does not have.
        """
        from ffmodel.league.env import WaiverClaim

        shortlist = list(observation["free_agents"])
        if not shortlist:
            return None
        week = observation["week"]
        roster = list(observation["roster"])

        # Who is spare is a lineup question -- it is this week's card that says
        # who is not needed. Which of the spares to cut, and who to claim, are
        # rest-of-season questions, so both sides of the comparison are valued
        # on the waiver weights.
        lineup = optimal_lineup(
            roster, env.positions, self.values(roster, week), env.config.slots
        )
        spare = [key for key in roster if key not in set(lineup.starting_keys())]
        if not spare:
            return None
        roster_values = self.values(roster, week, waiver=True)

        available = [
            key for key in shortlist if env.availability.is_available(key, week)
        ]
        if not available:
            return None
        free_values = self.values(available, week, waiver=True)

        # Context is what turns a ranking into an acquisition decision: the same
        # free agent is a must-add for a team short at his position and a wasted
        # roster spot for one already deep there, and nothing about the player
        # tells those apart.
        scores = {**roster_values, **free_values}
        if self.context_weights is not None:
            starting = set(lineup.starting_keys())
            shortfalls = team_shortfalls(
                env.rosters, env.positions, env.config.slots,
                env.availability, week,
            )
            shared = dict(
                values=scores,
                roster=roster,
                starters=starting,
                positions=env.positions,
                slots=env.config.slots,
                free_agents=available,
                shortfalls=shortfalls,
                agent_team=env.agent_team,
            )
            for keys, target in ((available, free_values), (spare, roster_values)):
                rows = build_context(keys, **shared)
                for key, bonus in zip(keys, rows @ self.context_weights):
                    target[key] = target.get(key, 0.0) + bonus

        drop = min(spare, key=lambda key: (roster_values.get(key, 0.0), key))
        add = max(available, key=lambda key: (free_values.get(key, 0.0), key))

        gain = free_values[add] - roster_values.get(drop, 0.0)
        if gain <= self.claim_threshold:
            return None
        return WaiverClaim(add_key=add, drop_key=drop)


def as_waiver_policy(agent: LinearAgent):
    """Adapt the agent's claim rule to what :func:`run_episode` expects."""

    def choose(env, observation):
        return agent.claim(env, observation)

    return choose
