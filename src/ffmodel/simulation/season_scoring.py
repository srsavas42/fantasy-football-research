"""Coherent season stat lines from volume and efficiency posteriors.

Every output draw uses one aligned volume draw and one aligned efficiency draw.
Categorical constructions enforce football accounting identities: passing
touchdowns are completions, receiving touchdowns are receptions, completions
plus interceptions cannot exceed attempts, and receptions cannot exceed
targets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

from ffmodel.config import SCORING_FORMATS, ScoringRules
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    SeasonAverageEfficiencyPrediction,
    SeasonAveragePosteriorEfficiencyPipeline,
)
from ffmodel.models.volume_season_average import SeasonAveragePrediction


PLAYER_ID_COLUMNS = ("season", "team", "player_key")
REQUIRED_EFFICIENCY_TARGETS = (
    "pass_completion_rate",
    "pass_yards_per_attempt",
    "pass_td_rate",
    "pass_int_rate",
    "rec_catch_rate",
    "rec_yards_per_target",
    "rec_td_rate",
    "rush_yards_per_carry",
    "rush_td_rate",
    "fumble_lost_rate",
)

EFFICIENCY_COPULA_STREAMS = {
    "pass": (
        "pass_completion_rate",
        "pass_yards_per_attempt",
        "pass_td_rate",
        "pass_int_rate",
    ),
    "rec": ("rec_catch_rate", "rec_yards_per_target", "rec_td_rate"),
    "rush": ("rush_yards_per_carry", "rush_td_rate"),
}

_COPULA_EXPOSURES = {
    "pass": ("pass_att", 50),
    "rec": ("targets", 20),
    "rush": ("rush_att", 20),
}


@dataclass
class SeasonScoringPrediction:
    """Aligned posterior stat lines and fantasy-point distributions."""

    player_rows: pd.DataFrame
    volume: SeasonAveragePrediction
    efficiency: SeasonAverageEfficiencyPrediction
    pass_cmp: np.ndarray
    pass_yds: np.ndarray
    pass_td: np.ndarray
    pass_int: np.ndarray
    receptions: np.ndarray
    rec_yds: np.ndarray
    rec_td: np.ndarray
    rush_yds: np.ndarray
    rush_td: np.ndarray
    fumbles_lost: np.ndarray
    fantasy_points: dict[str, np.ndarray]

    @property
    def draws(self) -> int:
        return self.pass_cmp.shape[1]

    def summary(
        self,
        scoring: str = "ppr",
        *,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> pd.DataFrame:
        """Return player-level posterior means and requested point quantiles."""
        if scoring not in self.fantasy_points:
            raise ValueError(f"unknown scoring format: {scoring}")
        points = self.fantasy_points[scoring]
        out = self.player_rows.copy().reset_index(drop=True)
        out[f"{scoring}_points_mean"] = points.mean(axis=1)
        for quantile in quantiles:
            label = int(round(quantile * 100))
            out[f"{scoring}_points_p{label:02d}"] = np.quantile(
                points, quantile, axis=1
            )
        return out


def volume_efficiency_rows(volume: SeasonAveragePrediction) -> pd.DataFrame:
    """Attach prediction-time volume summaries to efficiency feature rows."""
    rows = volume.player_rows.copy().reset_index(drop=True)
    rows["oof_pass_attempts_per_team_game"] = np.asarray(
        volume.pass_attempts_per_team_game, dtype=float
    ).mean(axis=1)
    rows["oof_targets_per_team_game"] = np.asarray(
        volume.targets_per_team_game, dtype=float
    ).mean(axis=1)
    rows["oof_carries_per_team_game"] = np.asarray(
        volume.carries_per_team_game, dtype=float
    ).mean(axis=1)
    rows["oof_fumble_opportunities_per_team_game"] = (
        rows["oof_pass_attempts_per_team_game"]
        + rows["oof_targets_per_team_game"]
        + rows["oof_carries_per_team_game"]
    )
    return rows


def volume_efficiency_exposures(
    volume: SeasonAveragePrediction,
) -> dict[str, np.ndarray]:
    """Map every efficiency response to its aligned future opportunity draws."""
    passes = np.asarray(volume.pass_attempts, dtype=int)
    targets = np.asarray(volume.targets, dtype=int)
    carries = np.asarray(volume.carries, dtype=int)
    fumble_opportunities = passes + targets + carries
    return {
        "pass_completion_rate": passes,
        "pass_yards_per_attempt": passes,
        "pass_td_rate": passes,
        "pass_int_rate": passes,
        "rec_catch_rate": targets,
        "rec_yards_per_target": targets,
        "rec_td_rate": targets,
        "rush_yards_per_carry": carries,
        "rush_td_rate": carries,
        "fumble_lost_rate": fumble_opportunities,
    }


def volume_efficiency_feature_samples(
    volume: SeasonAveragePrediction,
) -> dict[str, np.ndarray]:
    """Map fitted per-team-game volume covariates to aligned draw samples.

    These are distinct from integer opportunity exposures: the feature values
    are the simulated per-team-game role intensities used by the efficiency
    regressions. Passing them downstream makes a volume-to-efficiency link
    explicit in each scoring draw while retaining the independently fitted
    marginal model as the baseline option.
    """
    passes = np.asarray(volume.pass_attempts_per_team_game, dtype=float)
    targets = np.asarray(volume.targets_per_team_game, dtype=float)
    carries = np.asarray(volume.carries_per_team_game, dtype=float)
    return {
        "oof_pass_attempts_per_team_game": passes,
        "oof_targets_per_team_game": targets,
        "oof_carries_per_team_game": carries,
        "oof_fumble_opportunities_per_team_game": passes + targets + carries,
    }


def scale_efficiency_dispersion(
    prediction: SeasonAverageEfficiencyPrediction,
    scale: float,
) -> SeasonAverageEfficiencyPrediction:
    """Scale rate deviations around their conditional means.

    This is a prediction-time calibration layer: it cannot change a model's
    conditional mean and clips each response back to its physical bounds.
    ``scale=0`` removes latent season-to-season efficiency variation while
    retaining downstream Binomial event noise; ``scale=1`` is the fitted
    posterior.
    """
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("efficiency dispersion scale must be finite and nonnegative")
    rates = {}
    for target, values in prediction.rates.items():
        if target not in prediction.means:
            raise ValueError(f"efficiency means are missing {target}")
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        mean = np.asarray(prediction.means[target], dtype=float)
        values = np.asarray(values, dtype=float)
        if values.shape != mean.shape:
            raise ValueError(f"efficiency mean/rate draws are misaligned for {target}")
        rates[target] = np.clip(
            mean + scale * (values - mean), spec.lower, spec.upper
        )
    return SeasonAverageEfficiencyPrediction(
        player_rows=prediction.player_rows,
        means=prediction.means,
        rates=rates,
    )


def _nearest_correlation(matrix: np.ndarray, shrinkage: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num((matrix + matrix.T) / 2.0)
    np.fill_diagonal(matrix, 1.0)
    matrix = (1.0 - shrinkage) * matrix + shrinkage * np.eye(len(matrix))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    matrix = (eigenvectors * np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(matrix), 1e-12, None))
    matrix = matrix / np.outer(scale, scale)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def estimate_efficiency_copulas(
    rows: pd.DataFrame,
    *,
    shrinkage: float = 0.15,
) -> dict[str, np.ndarray]:
    """Estimate leakage-safe residual rank correlations by efficiency stream.

    Residuals use the lagged, partially pooled prior as their forecast. The
    Spearman matrix therefore measures future-season co-movement without an
    in-sample fitted residual leaking current outcomes into another row.
    Shrinkage toward independence stabilizes the smaller quarterback sample.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("copula shrinkage must be between zero and one")
    replacement = pd.to_numeric(
        rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
        errors="coerce",
    ).fillna(0)
    result = {}
    for stream, targets in EFFICIENCY_COPULA_STREAMS.items():
        exposure_name, minimum = _COPULA_EXPOSURES[stream]
        exposure = pd.to_numeric(
            rows.get(exposure_name, pd.Series(0, index=rows.index)),
            errors="coerce",
        )
        valid = exposure.ge(minimum) & replacement.ne(1)
        residuals = pd.DataFrame(index=rows.index)
        for target in targets:
            observed = pd.to_numeric(rows[target], errors="coerce")
            prior = pd.to_numeric(rows[f"prior_{target}"], errors="coerce")
            residuals[target] = observed - prior
        residuals = residuals[valid].dropna()
        if len(residuals) < max(30, 5 * len(targets)):
            result[stream] = np.eye(len(targets))
            continue
        correlation = residuals.corr(method="spearman").to_numpy(dtype=float)
        result[stream] = _nearest_correlation(correlation, shrinkage)
    return result


def apply_efficiency_copulas(
    prediction: SeasonAverageEfficiencyPrediction,
    correlations: Mapping[str, np.ndarray],
    *,
    seed: int = 0,
) -> SeasonAverageEfficiencyPrediction:
    """Reorder marginal draws to impose residual rank dependence.

    Iman-Conover-style rank reordering preserves every player's fitted
    marginal samples exactly. Only their joint pairing changes, so all
    per-response calibration and point forecasts remain untouched.
    """
    rates = {
        target: np.asarray(values, dtype=float).copy()
        for target, values in prediction.rates.items()
    }
    if not rates:
        return prediction
    draws = next(iter(rates.values())).shape[1]
    rng = np.random.default_rng(seed)
    for stream, targets in EFFICIENCY_COPULA_STREAMS.items():
        if not set(targets) <= set(rates):
            continue
        correlation = np.asarray(
            correlations.get(stream, np.eye(len(targets))), dtype=float
        )
        if correlation.shape != (len(targets), len(targets)):
            raise ValueError(f"{stream} copula has the wrong shape")
        correlation = _nearest_correlation(correlation, 0.0)
        cholesky = np.linalg.cholesky(correlation)
        supported = np.logical_and.reduce(
            [np.isfinite(rates[target]).all(axis=1) for target in targets]
        )
        for row in np.flatnonzero(supported):
            scores = rng.normal(size=(draws, len(targets))) @ cholesky.T
            for column, target in enumerate(targets):
                ordered = np.sort(rates[target][row])
                rank_order = np.argsort(scores[:, column], kind="mergesort")
                reordered = np.empty(draws, dtype=float)
                reordered[rank_order] = ordered
                rates[target][row] = reordered
    return SeasonAverageEfficiencyPrediction(
        player_rows=prediction.player_rows,
        means=prediction.means,
        rates=rates,
    )


def _conditional_probability(probability: np.ndarray, used: np.ndarray) -> np.ndarray:
    remaining = np.clip(1.0 - used, 0.0, 1.0)
    return np.divide(
        probability,
        remaining,
        out=np.zeros_like(probability, dtype=float),
        where=remaining > 1e-12,
    ).clip(0.0, 1.0)


def _three_outcomes(
    exposure: np.ndarray,
    first_probability: np.ndarray,
    second_probability: np.ndarray,
    third_probability: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized multinomial allocation through conditional Binomials."""
    first_probability = np.clip(first_probability, 0.0, 1.0)
    second_probability = np.minimum(
        np.clip(second_probability, 0.0, 1.0), 1.0 - first_probability
    )
    third_probability = np.minimum(
        np.clip(third_probability, 0.0, 1.0),
        1.0 - first_probability - second_probability,
    )
    first = rng.binomial(exposure, first_probability)
    remaining = exposure - first
    second = rng.binomial(
        remaining,
        _conditional_probability(second_probability, first_probability),
    )
    remaining = remaining - second
    third = rng.binomial(
        remaining,
        _conditional_probability(
            third_probability, first_probability + second_probability
        ),
    )
    return first, second, third


def fantasy_points_samples(
    statistics: Mapping[str, np.ndarray],
    rules: ScoringRules | str = "ppr",
) -> np.ndarray:
    """Apply one scoring system to aligned posterior stat arrays."""
    if isinstance(rules, str):
        rules = SCORING_FORMATS[rules]
    return (
        statistics["pass_yds"] * rules.pass_yd
        + statistics["pass_td"] * rules.pass_td
        + statistics["pass_int"] * rules.interception
        + statistics["rush_yds"] * rules.rush_yd
        + statistics["rush_td"] * rules.rush_td
        + statistics["rec_yds"] * rules.rec_yd
        + statistics["rec_td"] * rules.rec_td
        + statistics["receptions"] * rules.reception
        + statistics["fumbles_lost"] * rules.fumble_lost
    )


def scale_fantasy_point_dispersion(
    prediction: SeasonScoringPrediction,
    scale: float,
) -> SeasonScoringPrediction:
    """Calibrate joint scoring tails without changing player point means.

    The coherent stat arrays remain untouched. Only the derived fantasy-point
    distributions are scaled around each player's posterior mean, representing
    residual cross-stat dependence that independent efficiency components do
    not capture.
    """
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("fantasy-point dispersion scale must be finite and nonnegative")
    points = {}
    for name, samples in prediction.fantasy_points.items():
        samples = np.asarray(samples, dtype=float)
        mean = samples.mean(axis=1, keepdims=True)
        points[name] = mean + scale * (samples - mean)
    return replace(prediction, fantasy_points=points)


def simulate_season_scoring(
    volume: SeasonAveragePrediction,
    efficiency: SeasonAverageEfficiencyPrediction,
    *,
    scoring_formats: Mapping[str, ScoringRules] | None = None,
    seed: int = 0,
) -> SeasonScoringPrediction:
    """Combine aligned volume and efficiency draws into total season scoring."""
    volume_rows = volume.player_rows.reset_index(drop=True)
    efficiency_rows = efficiency.player_rows.reset_index(drop=True)
    missing_keys = set(PLAYER_ID_COLUMNS) - set(volume_rows)
    if missing_keys:
        raise ValueError(f"volume rows are missing identifiers: {sorted(missing_keys)}")
    if not volume_rows[list(PLAYER_ID_COLUMNS)].equals(
        efficiency_rows[list(PLAYER_ID_COLUMNS)]
    ):
        raise ValueError("volume and efficiency player rows are not aligned")
    missing_targets = set(REQUIRED_EFFICIENCY_TARGETS) - set(efficiency.rates)
    if missing_targets:
        raise ValueError(
            f"scoring requires fitted efficiency targets: {sorted(missing_targets)}"
        )

    pass_attempts = np.asarray(volume.pass_attempts, dtype=int)
    targets = np.asarray(volume.targets, dtype=int)
    carries = np.asarray(volume.carries, dtype=int)
    shape = pass_attempts.shape
    if targets.shape != shape or carries.shape != shape:
        raise ValueError("volume opportunity arrays must share one row/draw shape")
    if efficiency.draws != shape[1]:
        raise ValueError("volume and efficiency predictions must have the same draw count")
    if len(volume_rows) != shape[0]:
        raise ValueError("volume opportunity arrays must align to player rows")

    def rate(name: str) -> np.ndarray:
        values = np.asarray(efficiency.rates[name], dtype=float)
        if values.shape != shape:
            raise ValueError(f"efficiency samples are misaligned for {name}")
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)

    rng = np.random.default_rng(seed)

    completion_probability = np.clip(rate("pass_completion_rate"), 0.0, 1.0)
    pass_td_probability = np.minimum(
        np.clip(rate("pass_td_rate"), 0.0, 1.0), completion_probability
    )
    non_td_completion_probability = completion_probability - pass_td_probability
    pass_int_probability = np.minimum(
        np.clip(rate("pass_int_rate"), 0.0, 1.0),
        1.0 - completion_probability,
    )
    pass_td, non_td_completions, pass_int = _three_outcomes(
        pass_attempts,
        pass_td_probability,
        non_td_completion_probability,
        pass_int_probability,
        rng=rng,
    )
    pass_cmp = pass_td + non_td_completions
    pass_yards_per_completion = np.divide(
        rate("pass_yards_per_attempt"),
        np.clip(completion_probability, 0.05, None),
    ).clip(0.0, 40.0)
    pass_yds = np.rint(pass_cmp * pass_yards_per_completion).astype(int)

    catch_probability = np.clip(rate("rec_catch_rate"), 0.0, 1.0)
    rec_td_probability = np.minimum(
        np.clip(rate("rec_td_rate"), 0.0, 1.0), catch_probability
    )
    non_td_catch_probability = catch_probability - rec_td_probability
    rec_td, non_td_receptions, _ = _three_outcomes(
        targets,
        rec_td_probability,
        non_td_catch_probability,
        np.zeros(shape, dtype=float),
        rng=rng,
    )
    receptions = rec_td + non_td_receptions
    rec_yards_per_reception = np.divide(
        rate("rec_yards_per_target"),
        np.clip(catch_probability, 0.03, None),
    ).clip(0.0, 40.0)
    rec_yds = np.rint(receptions * rec_yards_per_reception).astype(int)

    rush_td_probability = np.clip(rate("rush_td_rate"), 0.0, 1.0)
    rush_td = rng.binomial(carries, rush_td_probability)
    rush_yds = np.rint(carries * rate("rush_yards_per_carry")).astype(int)

    fumble_opportunities = pass_attempts + targets + carries
    fumbles_lost = rng.binomial(
        fumble_opportunities,
        np.clip(rate("fumble_lost_rate"), 0.0, 1.0),
    )

    statistics = {
        "pass_yds": pass_yds,
        "pass_td": pass_td,
        "pass_int": pass_int,
        "rush_yds": rush_yds,
        "rush_td": rush_td,
        "rec_yds": rec_yds,
        "rec_td": rec_td,
        "receptions": receptions,
        "fumbles_lost": fumbles_lost,
    }
    scoring_formats = SCORING_FORMATS if scoring_formats is None else scoring_formats
    points = {
        name: fantasy_points_samples(statistics, rules)
        for name, rules in scoring_formats.items()
    }
    return SeasonScoringPrediction(
        player_rows=volume_rows,
        volume=volume,
        efficiency=efficiency,
        pass_cmp=pass_cmp,
        pass_yds=pass_yds,
        pass_td=pass_td,
        pass_int=pass_int,
        receptions=receptions,
        rec_yds=rec_yds,
        rec_td=rec_td,
        rush_yds=rush_yds,
        rush_td=rush_td,
        fumbles_lost=fumbles_lost,
        fantasy_points=points,
    )


def score_volume_prediction(
    volume: SeasonAveragePrediction,
    efficiency_model: SeasonAveragePosteriorEfficiencyPipeline,
    *,
    scoring_formats: Mapping[str, ScoringRules] | None = None,
    efficiency_dispersion_scale: float = 1.0,
    draw_conditioned_efficiency: bool = False,
    seed: int = 0,
) -> SeasonScoringPrediction:
    """Run the prediction-time volume-to-efficiency scoring handoff.

    The default preserves the accepted independent-marginal scoring path. The
    optional draw-conditioned candidate evaluates fitted efficiency means at
    each simulated per-team-game volume draw, which creates a directed shared
    role state without changing the volume generator or using future outcomes.
    """
    rows = volume_efficiency_rows(volume)
    draws = np.asarray(volume.pass_attempts).shape[1]
    efficiency = efficiency_model.predict_samples(
        rows,
        draws=draws,
        exposure_samples=volume_efficiency_exposures(volume),
        volume_feature_samples=(
            volume_efficiency_feature_samples(volume)
            if draw_conditioned_efficiency
            else None
        ),
        seed=seed,
    )
    efficiency = scale_efficiency_dispersion(
        efficiency, efficiency_dispersion_scale
    )
    return simulate_season_scoring(
        volume,
        efficiency,
        scoring_formats=scoring_formats,
        seed=seed + 1_000,
    )
