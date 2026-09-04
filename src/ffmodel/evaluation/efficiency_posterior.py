"""Distributional validation for season efficiency and total scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import (
    crps_decomposition,
    empirical_crps,
    interval_coverage,
    ordering_metrics,
    pit_calibration,
)

from ffmodel.models.efficiency_season_average import (
    EfficiencyModelSpec,
    ExposureWeightedEfficiencyModel,
    PosteriorSeasonEfficiencyModel,
)
from ffmodel.simulation.scoring import fantasy_points
from ffmodel.simulation.season_scoring import SeasonScoringPrediction

# Twelve is one starter per team in a standard league, so the projected top
# twelve at a position is exactly the set a drafter is trying to identify.
TOP_K = 12


# Efficiency-v1 promotion decisions. The posterior candidate must first beat
# whichever point baseline survived that screen, not merely the weaker of the
# two alternatives.
ACCEPTED_POINT_BASELINE = {
    "pass_completion_rate": "ridge",
    "pass_yards_per_attempt": "ridge",
    "pass_td_rate": "ridge",
    "pass_int_rate": "ridge",
    "rec_catch_rate": "prior",
    "rec_yards_per_target": "ridge",
    "rec_td_rate": "prior",
    "rush_yards_per_carry": "ridge",
    "rush_td_rate": "prior",
    "fumble_rate": "prior",
}


def efficiency_validation_mask(
    rows: pd.DataFrame, spec: EfficiencyModelSpec
) -> np.ndarray:
    observed = pd.to_numeric(
        rows.get(spec.target, pd.Series(np.nan, index=rows.index)), errors="coerce"
    )
    exposure = pd.to_numeric(
        rows.get(spec.exposure, pd.Series(np.nan, index=rows.index)), errors="coerce"
    )
    replacement = pd.to_numeric(
        rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
        errors="coerce",
    ).fillna(0)
    return (
        rows["position"].astype(str).isin(spec.positions)
        & exposure.ge(spec.min_exposure)
        & observed.notna()
        & np.isfinite(observed)
        & replacement.ne(1)
    ).to_numpy()


def score_efficiency_posterior(
    model: PosteriorSeasonEfficiencyModel,
    rows: pd.DataFrame,
    *,
    draws: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Score one holdout against ridge and pooled-prior point benchmarks."""
    out = rows.copy().reset_index(drop=True)
    spec = model.spec
    exposure_values = pd.to_numeric(
        out.get(spec.exposure, pd.Series(np.nan, index=out.index)), errors="coerce"
    ).fillna(0).to_numpy(dtype=float)
    latent = model.predict_samples(
        out,
        draws=draws,
        exposure_samples=exposure_values,
        seed=seed,
    )
    samples = model.predict_observed_samples(out, draws=draws, seed=seed)
    valid = efficiency_validation_mask(out, spec)
    valid = valid & np.isfinite(samples).all(axis=1)
    if not valid.any():
        raise ValueError(f"no valid holdout rows for {spec.target}")

    observed = pd.to_numeric(out[spec.target], errors="coerce").to_numpy(dtype=float)
    exposure = pd.to_numeric(out[spec.exposure], errors="coerce").to_numpy(dtype=float)
    # Point accuracy belongs to the latent conditional mean. Averaging noisy
    # posterior-predictive counts would add avoidable Monte Carlo error and
    # unfairly penalize a well-calibrated distribution against deterministic
    # ridge/prior baselines.
    posterior_mean = latent.mean.mean(axis=1)
    posterior_error = posterior_mean[valid] - observed[valid]
    weights = exposure[valid]
    crps = empirical_crps(observed[valid], samples[valid])
    coverage80 = interval_coverage(observed[valid], samples[valid], 0.80)
    coverage95 = interval_coverage(observed[valid], samples[valid], 0.95)

    # The fitted ridge prediction is supplied by callers that have access to
    # the training fold. Keep this function focused on posterior calibration;
    # ``point_baseline_metrics`` adds fold-trained comparisons below.
    # Compare on the same holdout rows. A raw lagged prior is absent for
    # rookies and other cold starts; the deployable prior baseline uses the
    # training-fold position fallback persisted by the posterior model.
    prior = model._prior_mean(out)
    prior_valid = valid & np.isfinite(prior)
    prior_error = prior[prior_valid] - observed[prior_valid]
    return {
        "target": spec.target,
        "n": int(valid.sum()),
        "opportunities": float(weights.sum()),
        "posterior_mae": float(np.abs(posterior_error).mean()),
        "posterior_rmse": float(np.sqrt(np.mean(posterior_error**2))),
        "posterior_weighted_mae": float(
            np.average(np.abs(posterior_error), weights=weights)
        ),
        "posterior_crps": float(crps.mean()),
        "posterior_weighted_crps": float(np.average(crps, weights=weights)),
        "coverage_80": float(coverage80["coverage"]),
        "coverage_95": float(coverage95["coverage"]),
        "prior_n": int(prior_valid.sum()),
        "prior_weighted_mae": (
            float(np.average(np.abs(prior_error), weights=exposure[prior_valid]))
            if prior_valid.any()
            else float("nan")
        ),
    }


def add_point_baseline_metrics(
    record: dict[str, object],
    *,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    model: PosteriorSeasonEfficiencyModel,
    ridge_alpha: float = 500.0,
) -> dict[str, object]:
    """Fit the leakage-safe ridge fold and add the accepted v1 benchmark."""
    spec = model.spec
    ridge = ExposureWeightedEfficiencyModel(
        spec,
        alpha=ridge_alpha,
        use_volume=model.use_volume,
        use_advanced=model.use_advanced,
    ).fit(train_rows)
    estimate = ridge.predict(test_rows)
    observed = pd.to_numeric(test_rows[spec.target], errors="coerce").to_numpy(
        dtype=float
    )
    exposure = pd.to_numeric(test_rows[spec.exposure], errors="coerce").to_numpy(
        dtype=float
    )
    valid = efficiency_validation_mask(test_rows, spec) & np.isfinite(estimate)
    error = estimate[valid] - observed[valid]
    ridge_weighted_mae = float(
        np.average(np.abs(error), weights=exposure[valid])
    )
    result = dict(record)
    result["ridge_weighted_mae"] = ridge_weighted_mae
    baseline = ACCEPTED_POINT_BASELINE[spec.target]
    result["accepted_point_baseline"] = baseline
    result["accepted_point_weighted_mae"] = float(
        ridge_weighted_mae
        if baseline == "ridge"
        else result["prior_weighted_mae"]
    )
    benchmark = result["accepted_point_weighted_mae"]
    result["posterior_point_relative_improvement"] = float(
        (benchmark - result["posterior_weighted_mae"]) / benchmark
        if benchmark and np.isfinite(benchmark)
        else float("nan")
    )
    return result


def observed_scoring_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Build canonical observed stat lines from prefixed efficiency outcomes."""
    mapping = {
        "pass_yds": "eff_pass_yds",
        "pass_td": "eff_pass_td",
        "pass_int": "eff_pass_int",
        "rush_yds": "eff_rush_yds",
        "rush_td": "eff_rush_td",
        "rec_yds": "eff_rec_yds",
        "rec_td": "eff_rec_td",
        "receptions": "eff_receptions",
        "fumbles_lost": "eff_fumbles_lost",
        "fumbles": "eff_fumbles",
    }
    out = pd.DataFrame(index=rows.index)
    for target, source in mapping.items():
        out[target] = pd.to_numeric(
            rows.get(source, pd.Series(np.nan, index=rows.index)), errors="coerce"
        )
    return out


def score_fantasy_points_posterior(
    prediction: SeasonScoringPrediction,
    *,
    scoring: str = "ppr",
    subset: np.ndarray | None = None,
) -> dict[str, object]:
    """Score total-season fantasy-point samples against observed stat lines.

    ``subset`` restricts the population without changing anything else, for
    asking whether a change that does nothing to the pooled number does
    something to a group that matters -- most of the pooled rows are fringe
    players whose season is a handful of points.
    """
    rows = prediction.player_rows.reset_index(drop=True)
    observed_stats = observed_scoring_rows(rows)
    observed = fantasy_points(observed_stats, scoring).to_numpy(dtype=float)
    samples = np.asarray(prediction.fantasy_points[scoring], dtype=float)
    replacement = pd.to_numeric(
        rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
        errors="coerce",
    ).fillna(0)
    valid = (
        np.isfinite(observed)
        & np.isfinite(samples).all(axis=1)
        & replacement.ne(1).to_numpy()
    )
    if subset is not None:
        subset = np.asarray(subset, dtype=bool)
        if subset.shape != valid.shape:
            raise ValueError(
                f"subset has {subset.shape} entries for {valid.shape} rows"
            )
        valid = valid & subset
    if not valid.any():
        raise ValueError("no rows left to score")
    mean = samples.mean(axis=1)
    error = mean[valid] - observed[valid]
    crps = empirical_crps(observed[valid], samples[valid])
    coverage80 = interval_coverage(observed[valid], samples[valid], 0.80)
    coverage95 = interval_coverage(observed[valid], samples[valid], 0.95)
    # MAE, RMSE and CRPS are one quantity -- distance from truth in points --
    # read at the centre, in the tails, and over the whole distribution. Two
    # things they cannot see are added here.
    #
    # The decomposition separates *calibration* from *information*. A change
    # that widens the posterior moves reliability; a change that tells players
    # apart better moves resolution. CRPS alone mixes them, so a feature that
    # adds real signal and a prior that was merely too tight look the same.
    #
    # Ordering is scored because the product is a ranked list. Nothing else
    # here knows that: a projection can improve its MAE while putting players
    # in a worse order, and the gate would call that a win.
    parts = crps_decomposition(observed[valid], samples[valid])
    positions = rows.get("position", pd.Series("", index=rows.index)).to_numpy()
    ordering = ordering_metrics(
        mean[valid], observed[valid], positions[valid], k=TOP_K
    )
    calibration = pit_calibration(observed[valid], samples[valid])
    return {
        "scoring": scoring,
        "n": int(valid.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "crps": float(crps.mean()),
        "coverage_80": float(coverage80["coverage"]),
        "coverage_95": float(coverage95["coverage"]),
        "reliability": parts["reliability"],
        "resolution": parts["resolution"],
        "uncertainty": parts["uncertainty"],
        "spearman": ordering.get("within_group_spearman", ordering["spearman"]),
        "concordance": ordering.get(
            "within_group_concordance", ordering["concordance"]
        ),
        "top_k": ordering.get("within_group_top_k", ordering["top_k"]),
        "pit_deviation": calibration["deviation"],
        "pit_shape": calibration["shape"],
        "pit_mean": calibration["mean_pit"],
    }
