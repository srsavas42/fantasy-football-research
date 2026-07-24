"""End-to-end validation of volume-v3 plus posterior efficiency scoring."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import (
    ACCEPTED_POINT_BASELINE,
    score_fantasy_points_posterior,
)
from ffmodel.evaluation.efficiency_season_average import (
    add_walk_forward_volume_features,
)
from ffmodel.evaluation.posterior_comparison import (
    atomic_write_json,
    load_json,
    posterior_sample_count,
    select_posterior_samples,
)
from ffmodel.features.season_average import SeasonAverageData, build_season_average_data
from ffmodel.models.base import load_idata
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_SPECS,
    ExposureWeightedEfficiencyModel,
    SeasonAverageEfficiencyPrediction,
    SeasonAveragePosteriorEfficiencyPipeline,
)
from ffmodel.models.season_opportunity import (
    SeasonCarryEligibilityModel,
    SeasonSnapShareModel,
)
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline
from ffmodel.simulation.season_scoring import (
    apply_efficiency_copulas,
    estimate_efficiency_copulas,
    scale_efficiency_dispersion,
    scale_fantasy_point_dispersion,
    simulate_season_scoring,
    volume_efficiency_feature_samples,
    volume_efficiency_rows,
    volume_efficiency_exposures,
)


def _restore_feature_model(model, state: dict[str, object]) -> None:
    for attribute in ("feature_names", "positions"):
        if attribute in state:
            setattr(model, attribute, list(state[attribute]))
    for attribute in ("feature_fill", "feature_mean", "feature_scale"):
        setattr(
            model,
            attribute,
            {key: float(value) for key, value in state[attribute].items()},
        )
    if "extra_features" in state:
        model.extra_features = tuple(state["extra_features"] or ())
    projection = state.get("feature_projection")
    model.feature_projection = (
        None if projection is None else np.asarray(projection, dtype=float)
    )


def _candidate_component(
    component_dir: Path,
    label: str,
    model,
    draws: int,
):
    metadata = load_json(component_dir / f"{label}.metadata.json")
    _restore_feature_model(model, metadata["model_state"])
    idata = load_idata(component_dir / f"{label}.nc")
    posterior = select_posterior_samples(idata.posterior, draws)
    model.idata = SimpleNamespace(posterior=posterior)
    return model


def load_volume_v3_holdout(
    baseline_dir: Path, component_dir: Path
) -> SeasonAverageVolumePipeline:
    """Assemble the promoted v3 components over the accepted v2 checkpoint."""
    pipeline = SeasonAverageVolumePipeline.load(baseline_dir)
    draws = posterior_sample_count(pipeline.team_model.idata.posterior)
    pipeline.snap_model = _candidate_component(
        component_dir,
        "snap_history",
        SeasonSnapShareModel(),
        draws,
    )
    pipeline.carry_eligibility_model = _candidate_component(
        component_dir,
        "carry_eligibility_efficiency",
        SeasonCarryEligibilityModel(),
        draws,
    )
    return pipeline


def load_efficiency_holdout(
    root: Path, holdout: int
) -> SeasonAveragePosteriorEfficiencyPipeline:
    pipeline = SeasonAveragePosteriorEfficiencyPipeline()
    for spec in EFFICIENCY_MODEL_SPECS:
        fitted = SeasonAveragePosteriorEfficiencyPipeline.load(
            root / f"holdout-{holdout}" / spec.target / "posterior"
        )
        pipeline.models[spec.target] = fitted.models[spec.target]
    return pipeline


def accepted_point_efficiency(
    train_rows: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    fitted: SeasonAveragePosteriorEfficiencyPipeline,
    *,
    draws: int,
) -> SeasonAverageEfficiencyPrediction:
    """Accepted efficiency-v1 means repeated over the volume draw axis."""
    means = {}
    rates = {}
    for spec in EFFICIENCY_MODEL_SPECS:
        fitted_model = fitted.models[spec.target]
        if ACCEPTED_POINT_BASELINE[spec.target] == "ridge":
            point = ExposureWeightedEfficiencyModel(
                spec,
                alpha=fitted.ridge_alpha,
                use_volume=fitted.use_volume,
                use_advanced=fitted.use_advanced,
            ).fit(train_rows).predict(prediction_rows)
        else:
            point = fitted_model._prior_mean(prediction_rows)
            supported = prediction_rows["position"].isin(spec.positions).to_numpy()
            point = np.where(supported, point, np.nan)
        samples = np.repeat(point[:, None], draws, axis=1)
        means[spec.target] = samples
        rates[spec.target] = samples
    return SeasonAverageEfficiencyPrediction(prediction_rows, means, rates)


def _pooled(records, model: str, scoring: str, metric: str) -> float:
    selected = [
        record
        for record in records
        if record["model"] == model and record["scoring"] == scoring
    ]
    return float(
        np.average(
            [record[metric] for record in selected],
            weights=[record["n"] for record in selected],
        )
    )


def _scale_label(
    efficiency_scale: float, point_scale: float, dependence: str
) -> str:
    return (
        f"posterior_efficiency_x{efficiency_scale:g}"
        f"_points_x{point_scale:g}"
        f"_{dependence}"
    )


def _write_report(args, records) -> dict[str, object]:
    pooled = defaultdict(dict)
    gates = {}
    candidates = sorted(
        {record["model"] for record in records if record["model"] != "accepted_point"}
    )
    for scoring in ("standard", "half_ppr", "ppr"):
        for model in ("accepted_point", *candidates):
            pooled[model][scoring] = {
                metric: _pooled(records, model, scoring, metric)
                for metric in ("mae", "rmse", "crps", "coverage_80", "coverage_95")
            }
        baseline = pooled["accepted_point"][scoring]
        for candidate_name in candidates:
            candidate = pooled[candidate_name][scoring]
            fold_pairs = []
            for holdout in args.holdouts:
                base = next(
                    record
                    for record in records
                    if record["season"] == holdout
                    and record["model"] == "accepted_point"
                    and record["scoring"] == scoring
                )
                challenger = next(
                    record
                    for record in records
                    if record["season"] == holdout
                    and record["model"] == candidate_name
                    and record["scoring"] == scoring
                )
                fold_pairs.append((base, challenger))
            gate_name = f"{candidate_name}:{scoring}"
            gates[gate_name] = {
                "mae_relative_improvement": (
                    baseline["mae"] - candidate["mae"]
                )
                / baseline["mae"],
                "crps_relative_improvement": (
                    baseline["crps"] - candidate["crps"]
                )
                / baseline["crps"],
                "mae_fold_wins": sum(
                    c["mae"] < b["mae"] for b, c in fold_pairs
                ),
                "crps_fold_wins": sum(
                    c["crps"] < b["crps"] for b, c in fold_pairs
                ),
                "coverage_80": candidate["coverage_80"],
                "coverage_95": candidate["coverage_95"],
            }
            gates[gate_name]["passed"] = bool(
                gates[gate_name]["mae_relative_improvement"] > 0
                and gates[gate_name]["crps_relative_improvement"] > 0
                and gates[gate_name]["mae_fold_wins"]
                >= min(2, len(args.holdouts))
                and gates[gate_name]["crps_fold_wins"]
                >= min(2, len(args.holdouts))
                and 0.70 <= candidate["coverage_80"] <= 0.90
                and 0.90 <= candidate["coverage_95"] <= 0.99
            )
    report = {
        "configuration": {
            "holdouts": args.holdouts,
            "efficiency_dir": str(args.efficiency_dir),
            "volume_baseline_dir": str(args.volume_baseline_dir),
            "volume_v3_components_dir": str(args.volume_v3_components_dir),
            "dispersion_scales": args.dispersion_scales,
            "point_dispersion_scales": args.point_dispersion_scales,
            "dependence": args.dependence,
            "copula_shrinkage": args.copula_shrinkage,
            "draw_conditioned_efficiency": args.draw_conditioned_efficiency,
        },
        "folds": records,
        "pooled": dict(pooled),
        "gates": gates,
    }
    atomic_write_json(args.output_dir / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2014, 2025)))
    parser.add_argument("--holdouts", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument(
        "--efficiency-dir",
        type=Path,
        default=Path(".cache/season-average-validation/efficiency-v2-final"),
    )
    parser.add_argument(
        "--volume-baseline-dir",
        type=Path,
        default=Path(".cache/season-average-validation/volume-v2"),
    )
    parser.add_argument(
        "--volume-v3-components-dir",
        type=Path,
        default=Path(".cache/season-average-validation/volume-v3-promotion-final"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dispersion-scales",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.25, 1.5],
    )
    parser.add_argument(
        "--point-dispersion-scales",
        nargs="+",
        type=float,
        default=[1.0],
    )
    parser.add_argument(
        "--dependence",
        nargs="+",
        choices=("independent", "copula"),
        default=["independent", "copula"],
    )
    parser.add_argument("--copula-shrinkage", type=float, default=0.15)
    parser.add_argument(
        "--draw-conditioned-efficiency",
        action="store_true",
        help=(
            "evaluate the directed candidate that evaluates efficiency means "
            "at each simulated volume draw"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/season-average-validation/season-scoring-v1"),
    )
    args = parser.parse_args(argv)

    data = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    )
    efficiency_rows = add_walk_forward_volume_features(
        data, include_efficiency=True, alpha=300.0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for holdout in args.holdouts:
        test = SeasonAverageData(
            data.team_rows[data.team_rows["season"] == holdout].copy(),
            data.player_rows[data.player_rows["season"] == holdout].copy(),
        )
        train_rows = efficiency_rows[
            efficiency_rows["season"] < holdout
        ].reset_index(drop=True)
        volume_model = load_volume_v3_holdout(
            args.volume_baseline_dir / f"holdout-{holdout}-final",
            args.volume_v3_components_dir / f"holdout-{holdout}",
        )
        volume = volume_model.predict_samples(test, seed=args.seed)
        efficiency_model = load_efficiency_holdout(args.efficiency_dir, holdout)
        prediction_rows = volume_efficiency_rows(volume)
        efficiency = efficiency_model.predict_samples(
            prediction_rows,
            draws=volume.pass_attempts.shape[1],
            exposure_samples=volume_efficiency_exposures(volume),
            seed=args.seed + 1_000,
        )
        draw_conditioned_efficiency = (
            efficiency_model.predict_samples(
                prediction_rows,
                draws=volume.pass_attempts.shape[1],
                exposure_samples=volume_efficiency_exposures(volume),
                volume_feature_samples=volume_efficiency_feature_samples(volume),
                seed=args.seed + 1_000,
            )
            if args.draw_conditioned_efficiency
            else None
        )

        point_efficiency = accepted_point_efficiency(
            train_rows,
            prediction_rows,
            efficiency_model,
            draws=volume.pass_attempts.shape[1],
        )
        baseline = simulate_season_scoring(
            volume, point_efficiency, seed=args.seed + 2_000
        )
        predictions = [("accepted_point", baseline)]
        if draw_conditioned_efficiency is not None:
            predictions.append(
                (
                    "draw_conditioned_efficiency",
                    simulate_season_scoring(
                        volume,
                        draw_conditioned_efficiency,
                        seed=args.seed + 2_000,
                    ),
                )
            )
        correlations = estimate_efficiency_copulas(
            train_rows, shrinkage=args.copula_shrinkage
        )
        for efficiency_scale in args.dispersion_scales:
            calibrated = scale_efficiency_dispersion(
                efficiency, efficiency_scale
            )
            for dependence in args.dependence:
                dependent = (
                    apply_efficiency_copulas(
                        calibrated,
                        correlations,
                        seed=args.seed + 1_500,
                    )
                    if dependence == "copula"
                    else calibrated
                )
                scoring_prediction = simulate_season_scoring(
                    volume, dependent, seed=args.seed + 2_000
                )
                for point_scale in args.point_dispersion_scales:
                    predictions.append(
                        (
                            _scale_label(
                                efficiency_scale, point_scale, dependence
                            ),
                            scale_fantasy_point_dispersion(
                                scoring_prediction, point_scale
                            ),
                        )
                    )
        for model_name, prediction in predictions:
            for scoring in ("standard", "half_ppr", "ppr"):
                record = score_fantasy_points_posterior(
                    prediction, scoring=scoring
                )
                record["season"] = int(holdout)
                record["model"] = model_name
                records.append(record)
        atomic_write_json(args.output_dir / "folds.json", records)
        print(f"scored total fantasy points for {holdout}", flush=True)

    report = _write_report(args, records)
    print(report["gates"], flush=True)


if __name__ == "__main__":
    main()
