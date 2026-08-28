"""Model 2: points from week ``w`` to the end of the regular season.

At ``w = 1`` this is the draft question answered without a draft board. From
about week 5 it is the waiver question: this player is available, what is he
worth for the rest of the year. The response is a sum over a known number of
remaining games, so the schedule is an offset the model is entitled to read --
how many games are left is public information, not a forecast.

The hard part is not the mean. It is the spread, and there is one way to get it
badly wrong that looks entirely reasonable.

**Why independent weeks are not enough.** The obvious construction simulates the
remaining games one at a time from the next-week model and adds them up. Drawing
those weeks independently makes the total's variance ``G`` times a single week's,
which is the variance of a player whose true ability is *known* and whose weeks
differ only by luck. That is not this problem. Most of the uncertainty about a
player's rest of season is uncertainty about him -- whether the role is real,
whether he stays healthy -- and it does not average out over ten games, because
it is the same unknown in every one of them. A model that ignores it produces
intervals far too narrow on exactly the decision where being wrong is expensive.

So each draw fixes the player first and then plays the games:

1. Draw a latent availability rate ``pi`` from a Beta whose mean is the fitted
   play probability and whose concentration is estimated from how much the
   realized play counts over-disperse relative to Binomial. This is the season
   layer's Beta-Binomial, applied where the exposure is large enough to identify
   it.
2. Draw a latent per-game level ``lambda`` around the fitted magnitude, with a
   standard deviation estimated by variance components: the covariance between
   two different weeks of the same player is the persistent part, and what is
   left over is week-to-week noise.
3. Play the remaining games against that fixed ``pi`` and ``lambda`` and sum.

Both latents are drawn once per draw and shared by every game in it, which is
what puts the correlation between weeks back.

:class:`DirectTotal` is the control. It regresses the total on the same features
with the games-remaining offset and resamples residuals locally -- no simulation,
no latent structure. If the hierarchy is not earning its place, the control will
say so, which is the comparison ``test_composition_cost.py`` made at season level
and the reason that document could rule an explanation out instead of asserting
one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ffmodel.weekly.features import HISTORY_ALPHA
from ffmodel.weekly.fitting import LocalResiduals, Logistic, Ridge
from ffmodel.weekly.nextweek import (
    ADP_FEATURES,
    AVAILABILITY_FEATURES,
    NEWS_AVAILABILITY_FEATURES,
    RECENT_FEATURES,
    SNAP_FEATURES,
    LOGISTIC_PENALTY,
    MAGNITUDE_FEATURES,
    PHASE_FEATURES,
    RIDGE_PENALTY,
    TEAM_FEATURES,
    _design,
)

TARGET = "ros_points"
OFFSET = "games_remaining"

# A Beta concentration this large is indistinguishable from no player-level
# dispersion at all, and one this small would let a single draw put a starter at
# a 5% play rate. Both ends are guards on an estimate from a variance remainder,
# which can come out negative on a well-behaved population.
MIN_CONCENTRATION = 1.0
MAX_CONCENTRATION = 400.0


def add_rest_of_season_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the remaining-points response and the games-remaining offset.

    Both are built so a player who leaves the league mid-season is handled
    honestly. His points after the exit are zero and are summed as zero; the
    games remaining are his *club's*, taken from the schedule at week ``w``, so
    the offset cannot quietly encode the fact that he was about to be cut.
    """
    out = frame.sort_values(["player_key", "season", "week"], kind="mergesort").copy()
    points = pd.to_numeric(out["points"], errors="coerce").fillna(0.0)
    grouped = points.groupby([out["player_key"], out["season"]], sort=False)
    # Reverse cumulative sum: every row gets its own week plus all later ones.
    out[TARGET] = grouped.transform(lambda s: s[::-1].cumsum()[::-1])

    schedule = out[["season", "week", "team"]].drop_duplicates()
    total = schedule.groupby(["season", "team"])["week"].transform("size")
    order = schedule.groupby(["season", "team"])["week"].rank(method="first")
    schedule = schedule.assign(games_remaining=(total - order + 1).astype(int))
    out = out.merge(schedule, on=["season", "week", "team"], how="left")
    out[OFFSET] = pd.to_numeric(out["games_remaining"], errors="coerce").fillna(1.0)
    return out


def _beta_concentration(
    played: np.ndarray, probability: np.ndarray, groups: np.ndarray
) -> float:
    """Concentration implied by how much realized play counts over-disperse.

    Exactly the decomposition ``scripts/measure_dispersion_link.py`` runs on the
    season layer's proportions: the observed variance of the realized rate minus
    the part independent Bernoulli trials at the fitted probability would
    produce, with the remainder read back as a Beta concentration. A remainder at
    or below zero means no durable player-to-player difference survives beyond
    sampling noise, and the caller is handed the ceiling rather than a negative
    concentration.
    """
    frame = pd.DataFrame(
        {"played": played, "probability": probability, "group": groups}
    )
    block = frame.groupby("group").agg(
        games=("played", "size"), rate=("played", "mean"), mu=("probability", "mean")
    )
    block = block[block["games"] >= 4]
    if len(block) < 50:
        return MAX_CONCENTRATION
    mu = block["mu"].to_numpy(float)
    rate = block["rate"].to_numpy(float)
    games = block["games"].to_numpy(float)
    total = float(np.mean((rate - mu) ** 2))
    binomial = float(np.mean(mu * (1.0 - mu) / games))
    latent = total - binomial
    if latent <= 0:
        return MAX_CONCENTRATION
    average = float(np.mean(mu * (1.0 - mu)))
    return float(np.clip(average / latent - 1.0, MIN_CONCENTRATION, MAX_CONCENTRATION))


def _persistent_sd(residual: np.ndarray, groups: np.ndarray) -> float:
    """Player-level standard deviation, from the covariance between his weeks.

    Two weeks of the same player share whatever the fitted value got wrong about
    *him* and not the luck in either week, so the mean cross-product of distinct
    weeks' residuals estimates that shared variance directly. Weeks are not
    paired with themselves, which is what keeps week-to-week noise out of it.
    """
    frame = pd.DataFrame({"residual": residual, "group": groups}).dropna()
    block = frame.groupby("group")["residual"].agg(["sum", "count", lambda s: (s**2).sum()])
    block.columns = ["total", "count", "square"]
    block = block[block["count"] >= 2]
    if block.empty:
        return 0.0
    cross = (block["total"] ** 2 - block["square"]).to_numpy(float)
    pairs = (block["count"] * (block["count"] - 1)).to_numpy(float)
    covariance = float(cross.sum() / pairs.sum())
    return float(np.sqrt(covariance)) if covariance > 0 else 0.0


@dataclass
class DirectTotal:
    """Regress the remaining total on the offset and the same lagged features.

    The control for the hierarchy. No latent player draw, no simulation: one
    ridge on the total, with the spread resampled from training rows whose
    projection was similar.
    """

    name: str = "direct-total"
    use_team: bool = True
    use_phase: bool = False
    use_adp: bool = False
    use_role: bool = False
    model: Ridge | None = None
    residuals: LocalResiduals | None = None
    medians: pd.Series | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return (
            MAGNITUDE_FEATURES
            + AVAILABILITY_FEATURES
            + (TEAM_FEATURES if self.use_team else ())
            + (PHASE_FEATURES if self.use_phase else ())
            + (ADP_FEATURES if self.use_adp else ())
            # Snap share, the last observation, and the injury report. All three
            # describe a role or a body rather than a single game, so they carry
            # over a multi-week horizon in a way a spread does not.
            + (
                SNAP_FEATURES + RECENT_FEATURES + NEWS_AVAILABILITY_FEATURES
                if self.use_role
                else ()
            )
            # Game script is deliberately absent even when requested. A spread is
            # published for one game; the rest-of-season response spans up to
            # seventeen, and this week's line says nothing about week twelve's.
            # Only the season-long part of the context travels.
        )

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        design, medians = _design(frame, self.features, self.medians)
        if self.medians is None:
            self.medians = medians
        offset = pd.to_numeric(frame[OFFSET], errors="coerce").fillna(1.0).to_numpy(float)
        # The offset enters directly and interacted with the per-game level, so
        # the fit can express "points per game times games left" rather than
        # having to approximate it additively.
        level = (
            pd.to_numeric(frame["prior_points_recent"], errors="coerce")
            .fillna(0.0)
            .to_numpy(float)
        )
        return np.column_stack([design, offset, offset * level])

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "DirectTotal":
        self.medians = None
        design = self._design(frame)
        target = np.asarray(target, float)
        keep = np.isfinite(target)
        self.model = Ridge.fit(design[keep], target[keep], penalty=RIDGE_PENALTY)
        fitted = self.model.predict(design[keep])
        self.residuals = LocalResiduals.fit(fitted, target[keep])
        return self

    def predict_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        fitted = self.model.predict(self._design(frame))
        drawn = fitted[:, None] + self.residuals.draw(fitted, draws, rng)
        return np.maximum(drawn, 0.0)


@dataclass
class HierarchicalSeason:
    """Draw the player once, then play out his remaining games against him.

    ``persistent`` switches the latent structure off, which turns the estimator
    into the independent-weeks construction the module docstring argues against.
    It exists so that argument can be measured instead of asserted.
    """

    name: str = "hierarchical"
    use_team: bool = True
    use_phase: bool = False
    use_role: bool = False
    use_adp: bool = False
    persistent: bool = True
    availability: Logistic | None = None
    magnitude: Ridge | None = None
    residuals: LocalResiduals | None = None
    availability_medians: pd.Series | None = None
    magnitude_medians: pd.Series | None = None
    concentration: float = MAX_CONCENTRATION
    level_sd: float = 0.0
    # The local residual pool carries a single week's whole spread, persistent
    # part included. Once the persistent part is added back as its own draw,
    # the week-level noise has to be shrunk by this factor or the total is
    # dispersed twice over.
    week_sd_scale: float = 1.0

    @property
    def magnitude_features(self) -> tuple[str, ...]:
        return (
            MAGNITUDE_FEATURES
            + (TEAM_FEATURES if self.use_team else ())
            + (PHASE_FEATURES if self.use_phase else ())
            + (ADP_FEATURES if self.use_adp else ())
            + (SNAP_FEATURES + RECENT_FEATURES if self.use_role else ())
        )

    @property
    def availability_features(self) -> tuple[str, ...]:
        return AVAILABILITY_FEATURES + (
            NEWS_AVAILABILITY_FEATURES if self.use_role else ()
        )

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "HierarchicalSeason":
        """Fitted on weekly outcomes, not on the totals it predicts.

        The components are a per-game play probability and a per-game scoring
        level, so the rows that identify them are player-weeks. The rest-of-season
        target never enters a fit -- it is only ever scored.
        """
        played = pd.to_numeric(frame["played"], errors="coerce").fillna(0).to_numpy(int)
        weekly = pd.to_numeric(frame["points"], errors="coerce").to_numpy(float)

        design, self.availability_medians = _design(frame, self.availability_features)
        self.availability = Logistic.fit(design, played, penalty=LOGISTIC_PENALTY)
        probability = self.availability.predict_proba(design)
        season_key = (
            frame["player_key"].astype(str) + "|" + frame["season"].astype(str)
        ).to_numpy()
        self.concentration = _beta_concentration(played, probability, season_key)

        on_field = (played == 1) & np.isfinite(weekly)
        played_frame = frame.loc[on_field]
        magnitude_design, self.magnitude_medians = _design(
            played_frame, self.magnitude_features
        )
        self.magnitude = Ridge.fit(
            magnitude_design, weekly[on_field], penalty=RIDGE_PENALTY
        )
        fitted = self.magnitude.predict(magnitude_design)
        residual = weekly[on_field] - fitted
        self.residuals = LocalResiduals.fit(fitted, weekly[on_field])
        self.level_sd = _persistent_sd(residual, season_key[on_field])
        # The local pool already carries the total spread of a single week. The
        # persistent part is about to be added back explicitly, so remove it here
        # or it is counted twice.
        self.week_sd_scale = 1.0
        total_variance = float(np.var(residual))
        remaining = total_variance - self.level_sd**2
        if total_variance > 0 and remaining > 0:
            self.week_sd_scale = float(np.sqrt(remaining / total_variance))
        return self

    def predict_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        if self.availability is None or self.magnitude is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        design, _ = _design(frame, self.availability_features, self.availability_medians)
        probability = np.clip(self.availability.predict_proba(design), 1e-4, 1 - 1e-4)
        magnitude_design, _ = _design(
            frame, self.magnitude_features, self.magnitude_medians
        )
        level = self.magnitude.predict(magnitude_design)
        games = (
            pd.to_numeric(frame[OFFSET], errors="coerce")
            .fillna(1.0)
            .to_numpy(float)
            .astype(int)
        )
        games = np.clip(games, 0, None)

        rows = len(frame)
        if self.persistent:
            alpha = probability * self.concentration
            beta = (1.0 - probability) * self.concentration
            rate = rng.beta(alpha[:, None], beta[:, None], size=(rows, draws))
            shift = rng.normal(0.0, self.level_sd, size=(rows, draws))
        else:
            rate = np.repeat(probability[:, None], draws, axis=1)
            shift = np.zeros((rows, draws))

        scale = self.week_sd_scale if self.persistent else 1.0
        totals = np.zeros((rows, draws), dtype=float)
        # Games are played against a player who was fixed before the first of
        # them, which is what correlates the weeks inside a draw. Only the rows
        # that still have a game left are simulated at each step -- a week-15 row
        # has three games and must not pay for a week-1 row's eighteen.
        for step in range(int(games.max()) if len(games) else 0):
            live = np.flatnonzero(games > step)
            if not live.size:
                continue
            centre = level[live]
            noise = self.residuals.draw(centre, draws, rng) * scale
            weekly = centre[:, None] + shift[live] + noise
            plays = rng.random((len(live), draws)) < rate[live]
            totals[live] += np.where(plays, weekly, 0.0)
        return np.maximum(totals, 0.0)


def rest_of_season_ladder() -> list:
    """The ladder, in the order the document reports it."""
    from ffmodel.weekly.market import WeeklyRankCurve

    return [
        WeeklyRankCurve(name="adp-curve", per_game=False, offset=OFFSET),
        DirectTotal(name="direct-total", use_team=True),
        DirectTotal(name="direct-total+phase", use_team=True, use_phase=True),
        DirectTotal(
            name="direct-total+phase+adp", use_team=True, use_phase=True, use_adp=True
        ),
        DirectTotal(
            name="direct-total+everything",
            use_team=True, use_phase=True, use_adp=True, use_role=True,
        ),
        RecursiveSeason(
            name="recursive-weekly",
            use_team=True, use_phase=True, use_adp=True, use_role=True,
            calibrate=False,
        ),
        RecursiveSeason(
            name="recursive+drift",
            use_team=True, use_phase=True, use_adp=True, use_role=True,
            calibrate=True,
        ),
        HierarchicalSeason(
            name="aggregated-weekly",
            # Same feature surface as the shipped weekly model, minus the
            # per-position fits and the single-game script terms. The first is
            # unimplemented here rather than judged unhelpful; the second is
            # deliberate, since a spread describes one game and this response
            # spans up to seventeen.
            persistent=True, use_team=True, use_phase=True, use_adp=True,
            use_role=True,
        ),
        HierarchicalSeason(name="independent-weeks", persistent=False),
        HierarchicalSeason(name="hierarchical", persistent=True),
    ]


# Features whose value is a function of realised outcomes, and so can be carried
# forward through a simulation. Everything else -- usage shares, snap share, team
# and defensive context, ADP, the injury report -- is held at its week-``w``
# value, because simulating it would mean simulating the offence around the
# player as well.
RECURSIVE_STATE = (
    "prior_points_recent",
    "prior_points_last",
    "prior_points_given_played",
    "prior_points_recent_given_played",
    "prior_play_rate",
    "recent_play_rate",
    "prior_games",
    "weeks_since_played",
)


def _column_index(columns: tuple[str, ...], name: str) -> int | None:
    """Where a feature sits in a design built by ``_design``.

    Features come first in declaration order, then the position dummies, then the
    missing-history indicator, so a feature's design index is just its position
    in the tuple.
    """
    try:
        return list(columns).index(name)
    except ValueError:
        return None


def _linear_deltas(model, columns: tuple[str, ...]) -> dict[str, float]:
    """Change in the linear predictor per unit change in each state feature.

    Both halves of the hurdle are linear in the standardised design, so a
    prediction can be updated without rebuilding it: moving feature ``j`` by
    ``d`` moves the linear predictor by ``coefficient_j * d / scale_j``. That is
    what makes a per-draw, per-week simulation affordable -- rebuilding the
    feature frame for 800 draws across seventeen weeks is not.
    """
    out = {}
    for name in RECURSIVE_STATE:
        index = _column_index(columns, name)
        if index is None:
            continue
        out[name] = float(
            model.coefficients[index] / model.standardizer.scale[index]
        )
    return out


@dataclass
class RecursiveSeason:
    """Simulate the season forward, feeding each week's draw into the next.

    The hierarchical simulator fixes a player once -- a latent availability rate
    and a latent scoring level -- and plays every remaining game against that
    frozen description. It is under-dispersed, covering 0.575 against a nominal
    0.80, and adding better weekly features did not move that at all.

    This is the construction that addresses the defect rather than the mean. Each
    simulated week updates the player's own history exactly as a real week would:
    a drawn outcome moves his recency-weighted average, his last-observation
    column, his play rate and his games-played count, and week ``g + 1`` is then
    predicted from the updated state. A hot streak inside a draw raises the
    projection that generates the next week of that same draw, so trajectories
    fan out the way careers actually do, and the spread of season totals is
    produced by the process rather than asserted by a variance component.

    Three honest limitations, none of them hidden by the result:

    **Only outcome-derived state is carried forward.** Usage shares, snap share,
    team and defensive context, ADP and the injury report stay at their week-``w``
    values. Simulating those means simulating the whole offence, which is a
    different and much larger model.

    **The model is fitted on real histories and fed its own.** By week ten of a
    draw the features it reads were generated by itself, not by the league. That
    is the standard exposure problem in recursive multi-step forecasting and it
    is precisely why this has to be measured rather than assumed better.

    **The exponential updates use the recursive form** (``alpha * y + (1 - alpha)
    * previous``) while the feature layer builds its averages with pandas'
    adjusted weighting. The two agree closely once a player has several
    observations behind him, which he does by construction here, since the
    simulation starts from his real history.
    """

    name: str = "recursive"
    use_team: bool = True
    use_phase: bool = False
    use_adp: bool = False
    use_role: bool = False
    # Per-game level shift, drawn once per draw and held for every remaining
    # game. Zero disables it; the default is estimated in ``fit``. See
    # ``_calibrate_drift``.
    drift_sd: float = 0.0
    calibrate: bool = True
    availability: Logistic | None = None
    magnitude: Ridge | None = None
    residuals: LocalResiduals | None = None
    availability_medians: pd.Series | None = None
    magnitude_medians: pd.Series | None = None
    chunk: int = 2000
    calibration_rows: int = 1500
    calibration_draws: int = 200

    @property
    def magnitude_features(self) -> tuple[str, ...]:
        return (
            MAGNITUDE_FEATURES
            + (TEAM_FEATURES if self.use_team else ())
            + (PHASE_FEATURES if self.use_phase else ())
            + (ADP_FEATURES if self.use_adp else ())
            + (SNAP_FEATURES + RECENT_FEATURES if self.use_role else ())
        )

    @property
    def availability_features(self) -> tuple[str, ...]:
        return AVAILABILITY_FEATURES + (
            NEWS_AVAILABILITY_FEATURES if self.use_role else ()
        )

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "RecursiveSeason":
        """Fitted on weekly rows, exactly as the next-week model is."""
        played = pd.to_numeric(frame["played"], errors="coerce").fillna(0).to_numpy(int)
        weekly = pd.to_numeric(frame["points"], errors="coerce").to_numpy(float)

        design, self.availability_medians = _design(frame, self.availability_features)
        self.availability = Logistic.fit(design, played, penalty=LOGISTIC_PENALTY)

        on_field = (played == 1) & np.isfinite(weekly)
        played_frame = frame.loc[on_field]
        magnitude_design, self.magnitude_medians = _design(
            played_frame, self.magnitude_features
        )
        self.magnitude = Ridge.fit(
            magnitude_design, weekly[on_field], penalty=RIDGE_PENALTY
        )
        self.residuals = LocalResiduals.fit(
            self.magnitude.predict(magnitude_design), weekly[on_field]
        )
        if self.calibrate:
            self.drift_sd = self._calibrate_drift(frame)
        return self

    def _calibrate_drift(self, frame: pd.DataFrame) -> float:
        """Size the variance the recursion cannot generate, from training data.

        The recursion propagates uncertainty about *scoring*, because points are
        what it simulates. It cannot propagate uncertainty about *role*: usage
        shares, snap share and the offence around the player stay frozen at week
        ``w``, and simulating them means simulating the whole team. Role change is
        the largest single source of weekly error and is mostly unannounced, so
        the missing spread is not a modelling detail -- it is most of what is
        unknown about a season.

        What cannot be generated can still be measured. Simulate the most recent
        training season with the drift switched off, and compare the spread the
        simulator produced against the spread the outcomes actually had. A level
        shift of ``d`` per game moves a total over ``G`` games by ``G * d``, so

            Var(observed - predicted) = Var_simulated + G^2 * drift^2

        and the remainder identifies ``drift`` directly. The estimate is a method
        of moments on the simulator's own shortfall, in the same spirit as the
        Beta concentration and the persistent-SD estimators above -- but measured
        against this construction rather than assumed from weekly residuals,
        which is the whole point: those weekly residuals are exactly what the
        recursion already reproduces.

        Fitted on the last training season only, never on a holdout, so the
        confirmation stays out of sample. A negative remainder means the
        simulator was already wide enough and returns zero rather than a
        nonsensical negative standard deviation.
        """
        if TARGET not in frame.columns:
            return 0.0
        seasons = sorted(frame["season"].unique().tolist())
        if len(seasons) < 2:
            return 0.0
        inner = frame[frame["season"] == seasons[-1]]
        inner = inner[np.isfinite(pd.to_numeric(inner[TARGET], errors="coerce"))]
        if len(inner) < 200:
            return 0.0
        # A subsample keeps the cost of fitting bounded; the quantity being
        # estimated is a single scalar and does not need every row.
        rng = np.random.default_rng(0)
        if len(inner) > self.calibration_rows:
            inner = inner.iloc[
                rng.choice(len(inner), self.calibration_rows, replace=False)
            ]

        previous, self.drift_sd = self.drift_sd, 0.0
        try:
            samples = self.predict_samples(
                inner, draws=self.calibration_draws, seed=917
            )
        finally:
            self.drift_sd = previous

        observed = pd.to_numeric(inner[TARGET], errors="coerce").to_numpy(float)
        games = (
            pd.to_numeric(inner[OFFSET], errors="coerce")
            .fillna(1.0)
            .to_numpy(float)
        )
        residual = observed - samples.mean(axis=1)
        simulated = samples.var(axis=1)
        usable = games > 0
        remainder = (residual[usable] ** 2 - simulated[usable]) / games[usable] ** 2
        drift = float(np.mean(remainder))
        return float(np.sqrt(drift)) if drift > 0 else 0.0

    def predict_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        if self.availability is None or self.magnitude is None:
            raise RuntimeError("fit before predicting")
        rng = np.random.default_rng(seed)
        games = (
            pd.to_numeric(frame[OFFSET], errors="coerce")
            .fillna(1.0)
            .to_numpy(float)
            .astype(int)
        )
        games = np.clip(games, 0, None)
        totals = np.zeros((len(frame), draws), dtype=np.float64)

        for start in range(0, len(frame), self.chunk):
            stop = min(start + self.chunk, len(frame))
            block = frame.iloc[start:stop]
            totals[start:stop] = self._simulate(block, games[start:stop], draws, rng)
        return np.maximum(totals, 0.0)

    def _simulate(
        self, block: pd.DataFrame, games: np.ndarray, draws: int, rng
    ) -> np.ndarray:
        rows = len(block)
        availability_design, _ = _design(
            block, self.availability_features, self.availability_medians
        )
        magnitude_design, _ = _design(
            block, self.magnitude_features, self.magnitude_medians
        )
        base_eta = (
            self.availability.standardizer.apply(availability_design)
            @ self.availability.coefficients
            + self.availability.intercept
        )
        base_mu = self.magnitude.predict(magnitude_design)
        eta_delta = _linear_deltas(self.availability, self.availability_features)
        mu_delta = _linear_deltas(self.magnitude, self.magnitude_features)

        # Starting state, taken from the player's real history at week w.
        def start(name: str, fallback: float) -> np.ndarray:
            values = pd.to_numeric(block.get(name), errors="coerce")
            if values is None:
                return np.full((rows, draws), fallback, dtype=np.float32)
            filled = values.fillna(fallback).to_numpy(np.float32)
            return np.repeat(filled[:, None], draws, axis=1)

        state = {name: start(name, 0.0) for name in RECURSIVE_STATE}
        # One shift per draw, fixed before the first game and shared by all of
        # them: role uncertainty is persistent, not week-to-week noise, so it
        # must not average out across the season.
        drift = (
            rng.normal(0.0, self.drift_sd, size=(rows, draws)).astype(np.float32)
            if self.drift_sd > 0
            else np.zeros((rows, draws), dtype=np.float32)
        )
        base = {name: state[name][:, :1].copy() for name in RECURSIVE_STATE}
        # Counters the running averages need, as floats so the updates vectorise.
        play_count = start("prior_games", 0.0)
        week_count = np.maximum(
            start("prior_weeks", 1.0), np.float32(1.0)
        )
        given_count = np.maximum(play_count.copy(), np.float32(1.0))

        alpha = np.float32(HISTORY_ALPHA)
        totals = np.zeros((rows, draws), dtype=np.float64)

        for step in range(int(games.max()) if len(games) else 0):
            live = games > step
            if not live.any():
                continue
            eta = base_eta[:, None] + sum(
                coefficient * (state[name] - base[name])
                for name, coefficient in eta_delta.items()
            )
            probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
            mu = base_mu[:, None] + sum(
                coefficient * (state[name] - base[name])
                for name, coefficient in mu_delta.items()
            )
            noise = self.residuals.draw(base_mu, draws, rng)
            plays = rng.random((rows, draws)) < probability
            drawn = np.where(plays, mu + noise + drift, 0.0)
            totals[live] += drawn[live]

            # Update the state exactly as a real week would have.
            played = plays.astype(np.float32)
            outcome = drawn.astype(np.float32)
            week_count = week_count + 1.0
            play_count = play_count + played
            state["prior_games"] = play_count
            state["prior_play_rate"] = play_count / week_count
            state["recent_play_rate"] = (
                alpha * played + (1.0 - alpha) * state["recent_play_rate"]
            )
            state["weeks_since_played"] = np.where(
                plays, 1.0, state["weeks_since_played"] + 1.0
            ).astype(np.float32)
            state["prior_points_recent"] = (
                alpha * outcome + (1.0 - alpha) * state["prior_points_recent"]
            )
            # The conditional averages move only on weeks he actually played.
            state["prior_points_last"] = np.where(
                plays, outcome, state["prior_points_last"]
            ).astype(np.float32)
            state["prior_points_recent_given_played"] = np.where(
                plays,
                alpha * outcome
                + (1.0 - alpha) * state["prior_points_recent_given_played"],
                state["prior_points_recent_given_played"],
            ).astype(np.float32)
            given_count = given_count + played
            state["prior_points_given_played"] = np.where(
                plays,
                state["prior_points_given_played"]
                + (outcome - state["prior_points_given_played"]) / given_count,
                state["prior_points_given_played"],
            ).astype(np.float32)
        return totals
