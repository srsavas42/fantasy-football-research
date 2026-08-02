"""Walk-forward cross-fitting and ablations for season efficiency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffmodel.evaluation.season_average import RidgeRosterBaseline, persistence_shares
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_SPECS,
    SeasonAverageEfficiencyPipeline,
)
from ffmodel.models.volume_season_average import VOLUME_EFFICIENCY_FEATURES


VOLUME_OUTPUTS = {
    "pass": ("oof_pass_attempts_per_team_game", "prior_pass_attempts_per_game", "pass_att"),
    "target": ("oof_targets_per_team_game", "prior_targets_per_game", "targets"),
    "carry": ("oof_carries_per_team_game", "prior_rush_attempts_per_game", "rush_att"),
}

VOLUME_ABLATION_CONFIGURATIONS = (
    (False, False, "volume_only"),
    (True, False, "direct_share_efficiency"),
    (True, True, "all_candidates"),
)


def add_walk_forward_volume_features(
    data: SeasonAverageData,
    *,
    include_efficiency: bool = True,
    include_experimental: bool = False,
    feature_overrides: dict[str, tuple[str, ...]] | None = None,
    alpha: float = 10.0,
    estimator: str = "ridge",
    sample_kwargs: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Cross-fit leak-free point volume projections for every player-season.

    Each season is predicted using only earlier response seasons. The first
    transition season falls back to the lagged-role persistence allocator.

    ``estimator`` selects what produces the projection.

    ``"ridge"`` (default) is cheap: a regularized roster softmax times the
    team's prior-season per-game rate. ``"pipeline"`` cross-fits the production
    Bayesian volume pipeline instead and takes its posterior mean, which is what
    actually generates these columns at serving time
    (``simulation.season_scoring.volume_efficiency_rows``).

    The two are not interchangeable, and the default is the mismatched one. On
    holdout 2024 they correlate 0.90-0.92 and agree on level, but they differ
    row-by-row by 0.46-0.66 opportunities per team game, and the ridge is the
    *less* accurate of the two against realized volume (pass 0.931 against
    0.651 MAE). Training an efficiency coefficient on the noisier construction
    and applying it to the cleaner one attenuates that coefficient. Prefer
    ``"pipeline"`` for a production efficiency fit and accept the cost: it is
    one full pipeline fit per response season.
    """
    if estimator not in {"ridge", "pipeline"}:
        raise ValueError("estimator must be 'ridge' or 'pipeline'")
    rows = data.player_rows.copy().reset_index(drop=True)
    rows["_oof_row"] = np.arange(len(rows))
    team_rows = data.team_rows.copy()
    efficiency_columns = {
        feature
        for features in VOLUME_EFFICIENCY_FEATURES.values()
        for feature in features
    }
    for output, _, _ in VOLUME_OUTPUTS.values():
        rows[output] = np.nan
    rows["oof_volume_training_seasons"] = 0

    for holdout in sorted(pd.to_numeric(rows["season"], errors="coerce").dropna().unique()):
        train = rows[rows["season"] < holdout].copy()
        test = rows[rows["season"] == holdout].copy()
        if include_efficiency:
            train_model = train
            test_model = test
        else:
            train_model = train.drop(columns=efficiency_columns, errors="ignore")
            test_model = test.drop(columns=efficiency_columns, errors="ignore")
        training_seasons = int(train["season"].nunique())
        rows.loc[test.index, "oof_volume_training_seasons"] = training_seasons

        if estimator == "pipeline" and not train_model.empty:
            projected = _pipeline_fold_projection(
                team_rows, train_model, test_model, holdout, sample_kwargs
            )
            if projected is not None:
                for output, values in projected.items():
                    rows.loc[test.index, output] = values
                continue

        team_lookup = team_rows[team_rows["season"] == holdout].set_index(
            ["season", "team"]
        )
        keys = pd.MultiIndex.from_frame(test[["season", "team"]])
        for stream, (output, rate_column, _) in VOLUME_OUTPUTS.items():
            if train_model.empty:
                shares = persistence_shares(test_model, stream)
            else:
                shares = RidgeRosterBaseline(
                    stream,
                    alpha=alpha,
                    include_experimental_efficiency=include_experimental,
                    extra_volume_features=(feature_overrides or {}).get(stream),
                ).fit(train_model).predict_shares(test_model)
            team_rate = team_lookup[rate_column].reindex(keys).to_numpy(dtype=float)
            rows.loc[test.index, output] = shares * team_rate

    rows["oof_fumble_opportunities_per_team_game"] = (
        rows["oof_pass_attempts_per_team_game"]
        + rows["oof_targets_per_team_game"]
        + rows["oof_carries_per_team_game"]
    )

    return rows.sort_values("_oof_row").drop(columns="_oof_row").reset_index(drop=True)


def _pipeline_fold_projection(
    team_rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    holdout: float,
    sample_kwargs: dict[str, object] | None,
) -> dict[str, np.ndarray] | None:
    """Posterior-mean volume for one fold, built the way serving builds it.

    Returns ``None`` when the fold cannot support a pipeline fit — early folds
    have too little history, and a source without quarterback snaps cannot fit
    the QB layers at all — so the caller falls back to the ridge path rather
    than losing the fold.
    """
    from ffmodel.models.volume_season_average import (
        SeasonAverageVolumePipeline,
        volume_input_problems,
    )
    from ffmodel.simulation.season_scoring import volume_efficiency_rows

    train = SeasonAverageData(
        team_rows[team_rows["season"] < holdout].copy(), train_rows
    )
    test = SeasonAverageData(
        team_rows[team_rows["season"] == holdout].copy(), test_rows
    )
    if volume_input_problems(train) or volume_input_problems(test):
        return None
    pipeline = SeasonAverageVolumePipeline().fit(train, **(sample_kwargs or {}))
    served = volume_efficiency_rows(pipeline.predict_samples(test, seed=0))

    # The pipeline sorts its own rows and adds replacement buckets, so align
    # back onto the caller's row order by identity rather than by position.
    keys = ["season", "team", "player_key"]
    outputs = [output for output, _, _ in VOLUME_OUTPUTS.values()]
    lookup = served.drop_duplicates(keys).set_index(keys)[outputs]
    wanted = pd.MultiIndex.from_frame(test_rows[keys])
    aligned = lookup.reindex(wanted)
    return {output: aligned[output].to_numpy(dtype=float) for output in outputs}


def walk_forward_efficiency_predictions(
    rows: pd.DataFrame,
    *,
    use_volume: bool,
    use_advanced: bool,
    alpha: float = 20.0,
    min_training_seasons: int = 2,
) -> pd.DataFrame:
    """Fit efficiency only on earlier seasons and concatenate holdout rows."""
    predictions = []
    seasons = sorted(pd.to_numeric(rows["season"], errors="coerce").dropna().unique())
    for holdout in seasons:
        train = rows[rows["season"] < holdout].copy()
        if train["season"].nunique() < min_training_seasons:
            continue
        test = rows[rows["season"] == holdout].copy()
        model = SeasonAverageEfficiencyPipeline(
            alpha=alpha,
            use_volume=use_volume,
            use_advanced=use_advanced,
        ).fit(train)
        predicted = model.predict(test)
        predicted["efficiency_holdout_season"] = int(holdout)
        predictions.append(predicted)
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True, sort=False)


def efficiency_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Opportunity-aware point metrics for every modeled efficiency response."""
    records = []
    for spec in EFFICIENCY_MODEL_SPECS:
        predicted = f"pred_{spec.target}"
        if predicted not in predictions:
            continue
        observed = pd.to_numeric(predictions[spec.target], errors="coerce")
        estimate = pd.to_numeric(predictions[predicted], errors="coerce")
        exposure = pd.to_numeric(predictions[spec.exposure], errors="coerce")
        prior = pd.to_numeric(predictions[spec.prior_feature], errors="coerce")
        valid = (
            predictions["position"].isin(spec.positions)
            & exposure.ge(spec.min_exposure)
            & observed.notna()
            & estimate.notna()
        )
        if not valid.any():
            continue
        error = estimate[valid] - observed[valid]
        weight = exposure[valid].to_numpy(dtype=float)
        prior_valid = valid & prior.notna()
        prior_mae = float(
            np.average(
                np.abs(prior[prior_valid] - observed[prior_valid]),
                weights=exposure[prior_valid],
            )
        ) if prior_valid.any() else np.nan
        records.append(
            {
                "target": spec.target,
                "n": int(valid.sum()),
                "opportunities": float(weight.sum()),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "weighted_mae": float(np.average(np.abs(error), weights=weight)),
                "prior_weighted_mae": prior_mae,
            }
        )
    return pd.DataFrame(records)


def volume_ablation_metrics(data: SeasonAverageData, *, alpha: float = 10.0) -> pd.DataFrame:
    """Compare cross-fitted volume with and without lagged efficiency."""
    records = []
    for include_efficiency, include_experimental, label in VOLUME_ABLATION_CONFIGURATIONS:
        predicted = add_walk_forward_volume_features(
            data,
            include_efficiency=include_efficiency,
            include_experimental=include_experimental,
            alpha=alpha,
        )
        named = pd.to_numeric(
            predicted.get("is_replacement_player", pd.Series(0, index=predicted.index)),
            errors="coerce",
        ).fillna(0).ne(1)
        team_games = pd.to_numeric(predicted["team_games"], errors="coerce").clip(lower=1)
        for stream, (output, _, observed_count) in VOLUME_OUTPUTS.items():
            observed = pd.to_numeric(predicted[observed_count], errors="coerce") / team_games
            estimate = pd.to_numeric(predicted[output], errors="coerce")
            valid = named & observed.notna() & estimate.notna()
            if stream == "pass":
                valid &= predicted["position"].eq("QB")
            error = estimate[valid] - observed[valid]
            records.append(
                {
                    "model": label,
                    "stream": stream,
                    "n": int(valid.sum()),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                }
            )
    return pd.DataFrame(records)


def volume_ablation_fold_metrics(
    data: SeasonAverageData, *, alpha: float = 300.0
) -> pd.DataFrame:
    """Season-level MAE/RMSE for feature acceptance stability checks."""
    records = []
    for include_efficiency, include_experimental, label in VOLUME_ABLATION_CONFIGURATIONS:
        predicted = add_walk_forward_volume_features(
            data,
            include_efficiency=include_efficiency,
            include_experimental=include_experimental,
            alpha=alpha,
        )
        named = pd.to_numeric(
            predicted.get("is_replacement_player", pd.Series(0, index=predicted.index)),
            errors="coerce",
        ).fillna(0).ne(1)
        team_games = pd.to_numeric(predicted["team_games"], errors="coerce").clip(lower=1)
        for stream, (output, _, observed_count) in VOLUME_OUTPUTS.items():
            observed = pd.to_numeric(predicted[observed_count], errors="coerce") / team_games
            estimate = pd.to_numeric(predicted[output], errors="coerce")
            valid = named & observed.notna() & estimate.notna()
            if stream == "pass":
                valid &= predicted["position"].eq("QB")
            scored = pd.DataFrame(
                {
                    "season": predicted.loc[valid, "season"].to_numpy(),
                    "error": (estimate[valid] - observed[valid]).to_numpy(dtype=float),
                }
            )
            for season, fold in scored.groupby("season"):
                records.append(
                    {
                        "model": label,
                        "stream": stream,
                        "season": int(season),
                        "n": int(len(fold)),
                        "mae": float(np.abs(fold["error"]).mean()),
                        "rmse": float(np.sqrt(np.mean(fold["error"] ** 2))),
                    }
                )
    return pd.DataFrame(records)


def run_efficiency_ablation(
    data: SeasonAverageData,
    *,
    volume_alpha: float = 10.0,
    efficiency_alpha: float = 20.0,
    min_training_seasons: int = 2,
) -> pd.DataFrame:
    """Run history, volume, advanced, and full efficiency feature sets."""
    rows = add_walk_forward_volume_features(
        data, include_efficiency=True, alpha=volume_alpha
    )
    frames = []
    configurations = (
        ("history", False, False),
        ("history_plus_volume", True, False),
        ("history_plus_advanced", False, True),
        ("full", True, True),
    )
    for label, use_volume, use_advanced in configurations:
        prediction = walk_forward_efficiency_predictions(
            rows,
            use_volume=use_volume,
            use_advanced=use_advanced,
            alpha=efficiency_alpha,
            min_training_seasons=min_training_seasons,
        )
        metrics = efficiency_metrics(prediction)
        metrics.insert(0, "model", label)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
