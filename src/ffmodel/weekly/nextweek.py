"""Model 1: points in week ``w``, given every week before it.

The start/sit response. Each estimator here is fitted on strictly earlier
seasons and returns predictive draws with shape ``(rows, draws)``, so the whole
ladder is scored by the same CRPS and coverage the season layer uses.

It is written as a ladder because the season layer's most expensive lesson was
that a four-parameter curve beat a hierarchical Bayesian pipeline carrying
fifteen features, and nobody found that out until they built the curve. Each
rung adds exactly one idea, so a rung that does not pay for itself is visible
rather than absorbed:

1. :class:`PositionClimatology` -- the position's weekly distribution. Knows
   nothing about the player, and is the score to beat before claiming any skill.
2. :class:`PlayerMean` -- his career average, zeros included.
3. :class:`RecencyMean` -- the same average, exponentially weighted.
4. :class:`Hurdle` -- availability and magnitude modelled separately.
5. :class:`HurdleTeam` -- the hurdle, plus the offence he plays in.

The structural claim being tested is rung 4. Roughly 43% of rostered
player-weeks record no stat line at all, and a model that predicts the mean of a
distribution that is half point-mass will be wrong about both halves: too low
for the weeks he plays, too high for the weeks he does not. Splitting the two
means the magnitude model is fitted only on weeks a player was actually on the
field, which is the population the question "what is he worth if I start him"
is about.

Negative outcomes are left alone. A quarterback with two interceptions and no
touchdown scores below zero in PPR, and clipping the draws at zero would misstate
a real tail. The point mass at exactly zero is reserved for weeks he did not play.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.weekly.fitting import LocalResiduals, Logistic, Ridge

POSITIONS = ("QB", "RB", "WR", "TE")

# Whether he suits up: history of doing so, and how recently.
AVAILABILITY_FEATURES = (
    "prior_play_rate",
    "recent_play_rate",
    "weeks_since_played",
    "prior_games",
)

# What he is worth when he does: scoring history and the usage behind it.
MAGNITUDE_FEATURES = (
    "prior_points_given_played",
    "prior_points_recent_given_played",
    "prior_points_recent",
    "prior_targets_recent",
    "prior_rush_att_recent",
    "prior_pass_att_recent",
    "prior_target_share_recent",
    "prior_rush_share_recent",
)

TEAM_FEATURES = (
    "team_plays_recent",
    "team_points_recent",
    "team_pass_att_recent",
    "team_rush_att_recent",
)

# The matchup, kept separate from team context so the two can be ruled in or out
# independently. Folk wisdom rates this the most important weekly input there is.
MATCHUP_FEATURES = ("defense_points_allowed_recent",)

RIDGE_PENALTY = 10.0
LOGISTIC_PENALTY = 5.0


def _design(
    frame: pd.DataFrame, columns: tuple[str, ...], medians: pd.Series | None = None
) -> tuple[np.ndarray, pd.Series]:
    """Feature matrix with position dummies, medians filled in for missing history.

    A player with no history at all is not an error to be dropped -- he is a
    rookie in week 1, and the panel is full of him. He gets the training median
    and an explicit indicator saying so, which lets the fit give those rows their
    own level instead of pretending they sit at the middle of the distribution.
    """
    block = frame.reindex(columns=list(columns)).apply(
        pd.to_numeric, errors="coerce"
    )
    if medians is None:
        medians = block.median()
    medians = medians.fillna(0.0)
    missing = block.isna().all(axis=1).astype(float)
    filled = block.fillna(medians).fillna(0.0)
    position = frame["position"].astype(str)
    dummies = np.column_stack([(position == name).astype(float) for name in POSITIONS])
    design = np.column_stack([filled.to_numpy(float), dummies, missing.to_numpy()])
    return design, medians


@dataclass
class PositionClimatology:
    """Resample the position's weekly outcomes. The floor every rung must clear."""

    name: str = "climatology"
    pools: dict[str, np.ndarray] = field(default_factory=dict)
    overall: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "PositionClimatology":
        target = np.asarray(target, float)
        position = frame["position"].astype(str).to_numpy()
        self.overall = target[np.isfinite(target)]
        for name in POSITIONS:
            values = target[(position == name) & np.isfinite(target)]
            if len(values) >= 100:
                self.pools[name] = values
        return self

    def predict_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        position = frame["position"].astype(str).to_numpy()
        out = np.empty((len(frame), draws), dtype=float)
        for name in POSITIONS:
            want = position == name
            if not want.any():
                continue
            pool = self.pools.get(name, self.overall)
            out[want] = rng.choice(pool, size=(int(want.sum()), draws), replace=True)
        unknown = ~np.isin(position, POSITIONS)
        if unknown.any():
            out[unknown] = rng.choice(
                self.overall, size=(int(unknown.sum()), draws), replace=True
            )
        return out


@dataclass
class HistoryMean:
    """A single lagged average, with residuals drawn from nearby fitted values.

    ``column`` selects which average: the career mean over every rostered week,
    or its exponentially weighted counterpart. Both include the zeros, so this is
    the honest form of "he averages twelve points" -- the heuristic a manager
    actually uses, made into a distribution so it can be scored against the rest.
    """

    column: str
    name: str = "history-mean"
    residuals: LocalResiduals | None = None
    fallback: float = 0.0

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "HistoryMean":
        fitted = self._fitted(frame)
        target = np.asarray(target, float)
        self.fallback = float(np.nanmedian(target))
        keep = np.isfinite(fitted) & np.isfinite(target)
        self.residuals = LocalResiduals.fit(fitted[keep], target[keep])
        return self

    def _fitted(self, frame: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(frame[self.column], errors="coerce").to_numpy(float)
        return np.where(np.isfinite(values), values, self.fallback)

    def predict_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        if self.residuals is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        fitted = self._fitted(frame)
        return fitted[:, None] + self.residuals.draw(fitted, draws, rng)


@dataclass
class Hurdle:
    """Availability and magnitude, fitted separately and recombined.

    ``P(plays)`` comes from a penalised logistic on his appearance history;
    ``points | plays`` from a ridge fitted **only on weeks he played**, with the
    spread resampled from training rows whose projection was similar. A draw
    picks one branch or the other, so the predictive distribution has the point
    mass at zero the response actually has.
    """

    name: str = "hurdle"
    use_team: bool = False
    use_matchup: bool = False
    availability: Logistic | None = None
    magnitude: Ridge | None = None
    residuals: LocalResiduals | None = None
    availability_medians: pd.Series | None = None
    magnitude_medians: pd.Series | None = None

    @property
    def magnitude_features(self) -> tuple[str, ...]:
        return (
            MAGNITUDE_FEATURES
            + (TEAM_FEATURES if self.use_team else ())
            + (MATCHUP_FEATURES if self.use_matchup else ())
        )

    @property
    def availability_features(self) -> tuple[str, ...]:
        return AVAILABILITY_FEATURES

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "Hurdle":
        target = np.asarray(target, float)
        played = pd.to_numeric(frame["played"], errors="coerce").fillna(0).to_numpy(int)

        design, self.availability_medians = _design(frame, self.availability_features)
        self.availability = Logistic.fit(design, played, penalty=LOGISTIC_PENALTY)

        on_field = (played == 1) & np.isfinite(target)
        if on_field.sum() < 200:
            raise ValueError("the magnitude model needs at least 200 played weeks")
        played_frame = frame.loc[on_field]
        magnitude_design, self.magnitude_medians = _design(
            played_frame, self.magnitude_features
        )
        self.magnitude = Ridge.fit(
            magnitude_design, target[on_field], penalty=RIDGE_PENALTY
        )
        fitted = self.magnitude.predict(magnitude_design)
        self.residuals = LocalResiduals.fit(fitted, target[on_field])
        return self

    def _magnitude_mean(self, frame: pd.DataFrame) -> np.ndarray:
        design, _ = _design(frame, self.magnitude_features, self.magnitude_medians)
        return self.magnitude.predict(design)

    def play_probability(self, frame: pd.DataFrame) -> np.ndarray:
        design, _ = _design(frame, self.availability_features, self.availability_medians)
        return self.availability.predict_proba(design)

    def magnitude_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        """``points | he plays``, without the zero atom.

        The only honest way to ask whether the magnitude half is calibrated. The
        full predictive cannot be scored against played rows: conditioning the
        *outcomes* on having played while leaving the *forecast* unconditional
        selects exactly the rows where the atom was wrong, and reports a
        correctly-sized zero mass as a downward bias.
        """
        if self.magnitude is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        mean = self._magnitude_mean(frame)
        return mean[:, None] + self.residuals.draw(mean, draws, rng)

    def predict_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        if self.availability is None or self.magnitude is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        probability = self.play_probability(frame)
        mean = self._magnitude_mean(frame)
        spread = self.residuals.draw(mean, draws, rng)
        plays = rng.random((len(frame), draws)) < probability[:, None]
        return np.where(plays, mean[:, None] + spread, 0.0)


def next_week_ladder() -> list:
    """The ladder, in the order the document reports it."""
    return [
        PositionClimatology(),
        HistoryMean(column="prior_points_mean", name="career-mean"),
        HistoryMean(column="prior_points_recent", name="recency-mean"),
        Hurdle(name="hurdle", use_team=False),
        Hurdle(name="hurdle+team", use_team=True),
        Hurdle(name="hurdle+team+matchup", use_team=True, use_matchup=True),
    ]
