"""Frozen-posterior walk-forward screen for the role-regime volume challenger.

The accepted volume-v3 posteriors are loaded unchanged for each holdout. The
only difference is the shared player-season regime draw and its team-conserving
allocation tilt, so this isolates the value of the role-only coupling without
rerunning expensive Bayesian fits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.evaluation.posterior_comparison import (
    load_json,
    posterior_sample_count,
    select_posterior_samples,
)
from ffmodel.features.season_average import SeasonAverageData, build_season_average_data
from ffmodel.models.base import load_idata
from ffmodel.models.season_opportunity import (
    SeasonCarryEligibilityModel,
    SeasonSnapShareModel,
)
from ffmodel.models.season_regime import SeasonRegimeModel
from ffmodel.models.season_regime_coupling import SeasonRegimeRoleCoupling
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline


def _restore_feature_model(model, state: dict[str, object]) -> None:
    for attribute in ("feature_names", "positions"):
        if attribute in state:
            setattr(model, attribute, list(state[attribute]))
    for attribute in ("feature_fill", "feature_mean", "feature_scale"):
        setattr(
            model, attribute, {key: float(value) for key, value in state[attribute].items()}
        )
    if "extra_features" in state:
        model.extra_features = tuple(state["extra_features"] or ())
    projection = state.get("feature_projection")
    model.feature_projection = (
        None if projection is None else np.asarray(projection, dtype=float)
    )


def _component(component_dir: Path, label: str, model, draws: int):
    metadata = load_json(component_dir / f"{label}.metadata.json")
    _restore_feature_model(model, metadata["model_state"])
    idata = load_idata(component_dir / f"{label}.nc")
    model.idata = SimpleNamespace(
        posterior=select_posterior_samples(idata.posterior, draws)
    )
    return model


def _load_volume_v3(baseline_dir: Path, component_dir: Path) -> SeasonAverageVolumePipeline:
    pipeline = SeasonAverageVolumePipeline.load(baseline_dir)
    draws = posterior_sample_count(pipeline.team_model.idata.posterior)
    pipeline.snap_model = _component(
        component_dir, "snap_history", SeasonSnapShareModel(), draws
    )
    pipeline.carry_eligibility_model = _component(
        component_dir,
        "carry_eligibility_efficiency",
        SeasonCarryEligibilityModel(),
        draws,
    )
    return pipeline


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    samples = np.asarray(samples, dtype=float)
    return {
        "n": int(len(observed)),
        "mae": float(np.abs(observed - samples.mean(axis=1)).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
    }


def _player_metrics(prediction) -> dict[str, dict[str, float]]:
    rows = prediction.player_rows
    named = pd.to_numeric(
        rows.get("is_replacement_player", pd.Series(0, index=rows.index)), errors="coerce"
    ).fillna(0).ne(1).to_numpy()
    games = rows["team_games"].to_numpy(float)
    observed = {
        "pass": rows["pass_att"].to_numpy(float) / games,
        "target": rows["targets"].to_numpy(float) / games,
        "carry": rows["rush_att"].to_numpy(float) / games,
    }
    samples = {
        "pass": prediction.pass_attempts_per_team_game,
        "target": prediction.targets_per_team_game,
        "carry": prediction.carries_per_team_game,
    }
    return {name: _metrics(observed[name][named], samples[name][named]) for name in samples}


def _delta(candidate: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]):
    return {
        stream: {
            metric: float(candidate[stream][metric] - baseline[stream][metric])
            for metric in ("mae", "crps", "coverage_80")
        }
        for stream in candidate
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2014, 2025)))
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
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
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)

    rows = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    )
    report: dict[str, object] = {"holdouts": args.holdouts, "folds": {}}
    for holdout in args.holdouts:
        train = SeasonAverageData(
            rows.team_rows.loc[rows.team_rows["season"].lt(holdout)].copy(),
            rows.player_rows.loc[rows.player_rows["season"].lt(holdout)].copy(),
        )
        test = SeasonAverageData(
            rows.team_rows.loc[rows.team_rows["season"].eq(holdout)].copy(),
            rows.player_rows.loc[rows.player_rows["season"].eq(holdout)].copy(),
        )
        baseline = _load_volume_v3(
            args.volume_baseline_dir / f"holdout-{holdout}-final",
            args.volume_v3_components_dir / f"holdout-{holdout}",
        )
        baseline_prediction = baseline.predict_samples(test, seed=args.seed)
        baseline_metrics = _player_metrics(baseline_prediction)

        baseline.role_regime_coupling = True
        baseline.regime_model = SeasonRegimeModel().fit(train.player_rows)
        baseline.regime_coupler = SeasonRegimeRoleCoupling().fit(
            train.player_rows, thresholds=baseline.regime_model.thresholds
        )
        candidate_prediction = baseline.predict_samples(test, seed=args.seed)
        if candidate_prediction.regime_probability is None:
            raise AssertionError("role-regime candidate did not emit regime probabilities")
        candidate_metrics = _player_metrics(candidate_prediction)
        fold = {
            "baseline": baseline_metrics,
            "role_regime": candidate_metrics,
            "delta_role_regime_minus_baseline": _delta(candidate_metrics, baseline_metrics),
            "mean_regime_probability": {
                name: float(candidate_prediction.regime_probability[:, index].mean())
                for index, name in enumerate(("replacement", "inactive", "committee", "lead"))
            },
        }
        report["folds"][str(holdout)] = fold
        print(
            f"{holdout}: target CRPS {baseline_metrics['target']['crps']:.4f} -> "
            f"{candidate_metrics['target']['crps']:.4f}; carry CRPS "
            f"{baseline_metrics['carry']['crps']:.4f} -> {candidate_metrics['carry']['crps']:.4f}; "
            f"pass CRPS {baseline_metrics['pass']['crps']:.4f} -> "
            f"{candidate_metrics['pass']['crps']:.4f}"
        )
    deltas = pd.DataFrame(
        [
            {
                f"{stream}_{metric}": value
                for stream, metrics in fold["delta_role_regime_minus_baseline"].items()
                for metric, value in metrics.items()
            }
            for fold in report["folds"].values()
        ]
    )
    report["mean_delta_role_regime_minus_baseline"] = {
        name: float(value) for name, value in deltas.mean().items()
    }
    print(json.dumps(report["mean_delta_role_regime_minus_baseline"], sort_keys=True))
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
