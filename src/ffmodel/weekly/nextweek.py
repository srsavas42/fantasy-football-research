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

# The same matchup question asked properly: run and pass defence apart, and
# volume conceded apart from efficiency conceded.
PHASE_FEATURES = (
    "def_rush_att_allowed",
    "def_rush_yds_allowed",
    "def_rush_ypc_allowed",
    "def_rush_epa_allowed",
    "def_targets_allowed",
    "def_rec_yds_allowed",
    "def_rec_epa_allowed",
)

# Game script. The spread says who is expected to lead, which decides whether a
# team runs out a win or throws to catch up; the implied totals say how much
# scoring is expected of this offence and how much it will have to answer.
SCRIPT_FEATURES = (
    "spread",
    "game_total",
    "implied_team_total",
    "implied_opponent_total",
    "own_def_rush_epa_allowed",
    "own_def_rec_epa_allowed",
    "own_def_rec_yds_allowed",
)

# The draft board, as an input rather than a rival. It carries the one thing the
# history cannot: what happened over the offseason -- a trade, a rookie, a
# vacated backfield -- none of which is in a box score until it is too late to
# help. That is worth most in week 1 and decays to nothing, so the level is
# interacted with an early-season indicator rather than entered flat, letting the
# fit lean on the board while the history is thin and discount it afterwards.
ADP_FEATURES = (
    "adp_log_rank",
    "adp_drafted",
    "adp_log_rank_early",
    "adp_drafted_early",
)

# Published before the game and not yet reflected in any average. The split
# between the two halves is the point: the injury report is overwhelmingly a
# statement about whether a player takes the field, while the depth chart is a
# statement about how much work he gets once he does.
NEWS_AVAILABILITY_FEATURES = (
    "inj_status",
    "inj_practice",
    "inj_out",
)

NEWS_MAGNITUDE_FEATURES = (
    "inj_status",
    "inj_practice",
    "depth_rank",
    "depth_promoted",
    "ahead_out",
    "position_group_out",
)

# Snap share: how much field time he got, as opposed to what he did with it.
# Kept separate so it can be ruled in or out on its own.
SNAP_FEATURES = (
    "prior_snap_share_recent",
    "prior_snap_share_step",
)

# The most recent observation, carried alongside the smoothed one. See the
# feature layer: a single decay cannot both react to a role change and average
# away a noisy week, and the fit does better given both than given either.
RECENT_FEATURES = (
    "prior_points_last",
    "prior_target_share_last",
    "prior_rush_share_last",
    "prior_snap_share_last",
)

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
    use_phase: bool = False
    use_script: bool = False
    use_adp: bool = False
    use_news: bool = False
    use_snaps: bool = False
    use_recent: bool = False
    by_position: bool = False
    availability: Logistic | None = None
    magnitude: Ridge | None = None
    residuals: LocalResiduals | None = None
    availability_medians: pd.Series | None = None
    magnitude_medians: pd.Series | None = None
    # Per-position fits, when ``by_position``. A position without enough played
    # weeks of its own falls back to the pooled fit above rather than to a
    # slope estimated from a handful of rows.
    parts: dict = field(default_factory=dict)

    @property
    def magnitude_features(self) -> tuple[str, ...]:
        return (
            MAGNITUDE_FEATURES
            + (TEAM_FEATURES if self.use_team else ())
            + (MATCHUP_FEATURES if self.use_matchup else ())
            + (PHASE_FEATURES if self.use_phase else ())
            + (SCRIPT_FEATURES if self.use_script else ())
            + (ADP_FEATURES if self.use_adp else ())
            + (NEWS_MAGNITUDE_FEATURES if self.use_news else ())
            + (SNAP_FEATURES if self.use_snaps else ())
            + (RECENT_FEATURES if self.use_recent else ())
        )

    @property
    def availability_features(self) -> tuple[str, ...]:
        return (
            AVAILABILITY_FEATURES
            + (NEWS_AVAILABILITY_FEATURES if self.use_news else ())
            + (SNAP_FEATURES if self.use_snaps else ())
            + (RECENT_FEATURES if self.use_recent else ())
        )

    def _fit_magnitude(
        self, frame: pd.DataFrame, target: np.ndarray
    ) -> tuple[Ridge, pd.Series, LocalResiduals]:
        design, medians = _design(frame, self.magnitude_features)
        model = Ridge.fit(design, target, penalty=RIDGE_PENALTY)
        residuals = LocalResiduals.fit(model.predict(design), target)
        return model, medians, residuals

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "Hurdle":
        target = np.asarray(target, float)
        played = pd.to_numeric(frame["played"], errors="coerce").fillna(0).to_numpy(int)

        design, self.availability_medians = _design(frame, self.availability_features)
        self.availability = Logistic.fit(design, played, penalty=LOGISTIC_PENALTY)

        on_field = (played == 1) & np.isfinite(target)
        if on_field.sum() < 200:
            raise ValueError("the magnitude model needs at least 200 played weeks")
        played_frame = frame.loc[on_field]
        played_target = target[on_field]
        self.magnitude, self.magnitude_medians, self.residuals = self._fit_magnitude(
            played_frame, played_target
        )

        # Several of the game-script terms point in opposite directions by
        # position -- a favourite's running back gets the fourth quarter and a
        # favourite's receivers do not -- so a single pooled slope averages them
        # towards zero and reports a real effect as a null. Fitting each position
        # its own slopes is the encoding that lets those terms be seen at all,
        # and it avoids the collinear-interaction pathology that sank the season
        # layer's ADP interactions: these are four separate designs, not one
        # design carrying a level plus three deviations under a shared prior.
        self.parts = {}
        if self.by_position:
            position = played_frame["position"].astype(str).to_numpy()
            for name in POSITIONS:
                want = position == name
                if want.sum() < 1000:
                    continue
                self.parts[name] = self._fit_magnitude(
                    played_frame.loc[want], played_target[want]
                )
        return self

    def _magnitude_mean(self, frame: pd.DataFrame) -> np.ndarray:
        design, _ = _design(frame, self.magnitude_features, self.magnitude_medians)
        mean = self.magnitude.predict(design)
        if not self.parts:
            return mean
        position = frame["position"].astype(str).to_numpy()
        for name, (model, medians, _) in self.parts.items():
            want = position == name
            if not want.any():
                continue
            block, _ = _design(frame.loc[want], self.magnitude_features, medians)
            mean[want] = model.predict(block)
        return mean

    def _draw_residuals(
        self, frame: pd.DataFrame, mean: np.ndarray, draws: int, rng
    ) -> np.ndarray:
        """Spread from each position's own residual pool when fitted that way."""
        if not self.parts:
            return self.residuals.draw(mean, draws, rng)
        position = frame["position"].astype(str).to_numpy()
        out = np.empty((len(frame), draws), dtype=float)
        covered = np.zeros(len(frame), dtype=bool)
        for name, (_, _, residuals) in self.parts.items():
            want = position == name
            if not want.any():
                continue
            out[want] = residuals.draw(mean[want], draws, rng)
            covered |= want
        if (~covered).any():
            out[~covered] = self.residuals.draw(mean[~covered], draws, rng)
        return out

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
        return mean[:, None] + self._draw_residuals(frame, mean, draws, rng)

    def predict_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        if self.availability is None or self.magnitude is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        probability = self.play_probability(frame)
        mean = self._magnitude_mean(frame)
        spread = self._draw_residuals(frame, mean, draws, rng)
        plays = rng.random((len(frame), draws)) < probability[:, None]
        return np.where(plays, mean[:, None] + spread, 0.0)


def next_week_ladder() -> list:
    """The ladder, in the order the document reports it.

    The market curve is first because it is the baseline everything else has to
    beat, not because it is the weakest.
    """
    from ffmodel.weekly.market import WeeklyRankCurve

    return [
        WeeklyRankCurve(name="adp-curve", per_game=True),
        PositionClimatology(),
        HistoryMean(column="prior_points_mean", name="career-mean"),
        HistoryMean(column="prior_points_recent", name="recency-mean"),
        Hurdle(name="hurdle", use_team=False),
        Hurdle(name="hurdle+team+matchup", use_team=True, use_matchup=True),
        Hurdle(
            name="hurdle+context",
            use_team=True,
            use_matchup=True,
            use_phase=True,
            use_script=True,
        ),
        Hurdle(
            name="hurdle+context/position",
            use_team=True,
            use_matchup=True,
            use_phase=True,
            use_script=True,
            by_position=True,
        ),
        Hurdle(
            name="hurdle+everything+recent/position",
            use_team=True, use_matchup=True, use_phase=True, use_script=True,
            use_adp=True, use_news=True, use_snaps=True, use_recent=True,
            by_position=True,
        ),
        Hurdle(
            name="hurdle+context+adp+news+snaps/position",
            use_team=True,
            use_matchup=True,
            use_phase=True,
            use_script=True,
            use_adp=True,
            use_news=True,
            use_snaps=True,
            by_position=True,
        ),
        Hurdle(
            name="hurdle+context+adp+news/position",
            use_team=True,
            use_matchup=True,
            use_phase=True,
            use_script=True,
            use_adp=True,
            use_news=True,
            by_position=True,
        ),
        Hurdle(
            name="hurdle+context+adp/position",
            use_team=True,
            use_matchup=True,
            use_phase=True,
            use_script=True,
            use_adp=True,
            by_position=True,
        ),
    ]
