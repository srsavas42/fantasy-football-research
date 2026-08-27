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

from ffmodel.weekly.fitting import LocalResiduals, Logistic, Ridge
from ffmodel.weekly.nextweek import (
    AVAILABILITY_FEATURES,
    LOGISTIC_PENALTY,
    MAGNITUDE_FEATURES,
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
    model: Ridge | None = None
    residuals: LocalResiduals | None = None
    medians: pd.Series | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return MAGNITUDE_FEATURES + AVAILABILITY_FEATURES + (
            TEAM_FEATURES if self.use_team else ()
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
        return MAGNITUDE_FEATURES + (TEAM_FEATURES if self.use_team else ())

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "HierarchicalSeason":
        """Fitted on weekly outcomes, not on the totals it predicts.

        The components are a per-game play probability and a per-game scoring
        level, so the rows that identify them are player-weeks. The rest-of-season
        target never enters a fit -- it is only ever scored.
        """
        played = pd.to_numeric(frame["played"], errors="coerce").fillna(0).to_numpy(int)
        weekly = pd.to_numeric(frame["points"], errors="coerce").to_numpy(float)

        design, self.availability_medians = _design(frame, AVAILABILITY_FEATURES)
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
        design, _ = _design(frame, AVAILABILITY_FEATURES, self.availability_medians)
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
    return [
        DirectTotal(name="direct-total", use_team=True),
        HierarchicalSeason(name="independent-weeks", persistent=False),
        HierarchicalSeason(name="hierarchical", persistent=True),
    ]
