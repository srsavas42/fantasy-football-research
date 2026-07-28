"""Compare continuous QB snap workload with the prior starter architecture.

This targeted walk-forward test reuses accepted team and availability
posteriors, so pass-allocation changes are attributable to the QB workload
layer rather than unrelated refits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps
from ffmodel.evaluation.season_average import RidgeRosterBaseline, persistence_shares
from ffmodel.features.season_average import build_season_average_data
from ffmodel.models.base import load_idata, sampling_quality, save_idata
from ffmodel.models.season_availability import (
    QBWorkloadShareModel,
    SeasonAvailabilityModel,
)
from ffmodel.models.volume_season_average import (
    TeamSeasonAverageModel,
    _align_group_draws,
    _allocate_season_counts,
)


def _restore_features(model, state):
    model.feature_names = list(state["feature_names"])
    for name in ("feature_fill", "feature_mean", "feature_scale"):
        setattr(model, name, {key: float(value) for key, value in state[name].items()})


def _team_rates(rows, team_rows):
    lookup = team_rows.set_index(["season", "team"])[
        "prior_pass_attempts_per_game"
    ]
    keys = pd.MultiIndex.from_frame(rows[["season", "team"]])
    return lookup.reindex(keys).to_numpy(dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2014, 2025)))
    parser.add_argument("--holdouts", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--nuts-sampler", choices=("pymc", "nutpie"), default="nutpie")
    parser.add_argument(
        "--reuse-workload",
        action="store_true",
        help="score workload.nc files already present in --output-dir",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(".cache/season-average-validation/large-walk-forward-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/season-average-validation/qb-workload-v2"),
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    )
    point = defaultdict(lambda: [0, 0.0, 0.0])
    distribution = defaultdict(lambda: [0, 0.0, 0])
    winners = defaultdict(lambda: [0, 0])

    def add_point(metric, method, observed, predicted):
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        values = point[(metric, method)]
        values[0] += len(observed)
        values[1] += np.abs(observed - predicted).sum()
        values[2] += np.square(observed - predicted).sum()

    def add_distribution(metric, observed, samples):
        observed = np.asarray(observed, dtype=float)
        samples = np.asarray(samples, dtype=float)
        lower, upper = np.quantile(samples, [0.1, 0.9], axis=1)
        values = distribution[metric]
        values[0] += len(observed)
        values[1] += empirical_crps(observed, samples).sum()
        values[2] += ((observed >= lower) & (observed <= upper)).sum()

    for holdout in args.holdouts:
        train = data.player_rows[data.player_rows["season"] < holdout].copy()
        test = data.player_rows[data.player_rows["season"] == holdout].copy()
        team_test = data.team_rows[data.team_rows["season"] == holdout].copy()
        baseline = args.baseline_dir / f"holdout-{holdout}"
        metadata = json.loads((baseline / "metadata.json").read_text())

        team = TeamSeasonAverageModel(**metadata["team"])
        team_path = baseline / (
            "team-2000.nc" if holdout == 2023 else "team-1000.nc"
        )
        team.idata = load_idata(team_path)

        state = metadata["availability"]
        availability = SeasonAvailabilityModel(positions=list(state["positions"]))
        _restore_features(availability, state)
        availability_path = baseline / (
            "availability-1000.nc"
            if holdout in (2022, 2024)
            else "availability.nc"
        )
        availability.idata = load_idata(availability_path)

        holdout_dir = args.output_dir / f"holdout-{holdout}"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        if args.reuse_workload:
            workload = QBWorkloadShareModel()
            workload._design(train, fit=True)
            workload.idata = load_idata(holdout_dir / "workload.nc")
        else:
            workload = QBWorkloadShareModel().fit(
                train,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                nuts_sampler=args.nuts_sampler,
            )
            save_idata(workload.idata, holdout_dir / "workload.nc")
        quality = sampling_quality(workload.idata, ["beta"])
        draws = workload.idata.posterior.sizes["chain"] * workload.idata.posterior.sizes["draw"]

        availability_prediction = availability.predict_samples(test, seed=43)
        workload_prediction = workload.predict_share_samples(
            availability_prediction.rows,
            availability_samples=availability_prediction.availability[:, :draws],
            seed=44,
        )
        team_prediction = team.predict_average_samples(team_test, seed=42)
        totals = _align_group_draws(
            workload_prediction.group_keys,
            team_prediction["rows"],
            team_prediction["pass_attempts"][:, :draws],
        )
        counts = _allocate_season_counts(workload_prediction, totals, seed=47)
        group = workload_prediction.rows["_group_idx"].to_numpy(dtype=int)
        games = _align_group_draws(
            workload_prediction.group_keys,
            team_prediction["rows"],
            team_prediction["games"],
        )[group, None]
        samples = counts / games

        rows = workload_prediction.rows.reset_index(drop=True)
        observed = rows["pass_att"].to_numpy(dtype=float) / rows[
            "team_games"
        ].to_numpy(dtype=float)
        team_rate = _team_rates(rows, team_test)
        persistence = persistence_shares(rows, "pass") * team_rate
        ridge = RidgeRosterBaseline("pass").fit(train).predict_shares(rows) * team_rate
        quarterback = rows["position"].eq("QB").to_numpy()
        for metric, mask in (
            ("player pass ALL", np.ones(len(rows), dtype=bool)),
            ("player pass QB", quarterback),
        ):
            add_point(metric, "persistence", observed[mask], persistence[mask])
            add_point(metric, "ridge", observed[mask], ridge[mask])
            add_point(metric, "workload", observed[mask], samples[mask].mean(axis=1))
            add_distribution(metric, observed[mask], samples[mask])

        quarterbacks = rows[quarterback].copy()
        quarterbacks["predicted"] = workload_prediction.shares[quarterback].mean(axis=1)
        quarterbacks["observed_workload"] = quarterbacks[
            "observed_qb_workload_share"
        ].fillna(0.0)
        role = quarterbacks["prior_qb_snap_share"].where(
            quarterbacks["prior_qb_snap_share"] > 0
        )
        role = role.combine_first(
            quarterbacks["prior_pass_role"].where(quarterbacks["prior_pass_role"] > 0)
        )
        role = role.combine_first(
            quarterbacks["draft_pass_prior"].where(
                quarterbacks["draft_pass_prior"] > 0
            )
        ).fillna(0.02)
        quarterbacks["prior"] = role.groupby(
            [quarterbacks["season"], quarterbacks["team"]]
        ).transform(lambda values: values / values.sum())
        depth_score = (
            quarterbacks["qb_listed_starter"].fillna(0) * 1_000_000
            - quarterbacks["qb_depth_rank"].fillna(99) * 1_000
            + quarterbacks["prior_pass_role"].fillna(0)
        )
        quarterbacks["depth"] = 0.0
        depth_selected = depth_score.groupby(
            [quarterbacks["season"], quarterbacks["team"]]
        ).idxmax()
        quarterbacks.loc[depth_selected, "depth"] = 1.0
        for method in ("prior", "depth", "predicted"):
            add_point(
                "QB workload share",
                method,
                quarterbacks["observed_workload"],
                quarterbacks[method],
            )
        add_distribution(
            "QB workload share",
            quarterbacks["observed_workload"],
            workload_prediction.shares[quarterback],
        )
        actual = set(
            quarterbacks.groupby(["season", "team"])[
                "observed_workload"
            ].idxmax()
        )
        year_accuracy = {}
        for method in ("prior", "depth", "predicted"):
            selected = quarterbacks.groupby(["season", "team"])[method].idxmax()
            winners[method][0] += len(selected)
            correct = sum(index in actual for index in selected)
            winners[method][1] += correct
            year_accuracy[method] = correct / len(selected)

        print(
            f"HOLDOUT {holdout}: passed={quality['passed']} "
            f"rhat={quality['max_rhat']:.4f} ess={quality['min_bulk_ess']:.0f} "
            f"divergences={quality['divergences']} "
            f"innovation={workload.role_innovation_scale:.3f} "
            f"non_qb_max={workload_prediction.shares[~quarterback].max():.1f} "
            f"QB_pass_MAE={np.abs(observed[quarterback] - samples[quarterback].mean(axis=1)).mean():.3f} "
            f"QB_workload_MAE={np.abs(quarterbacks['observed_workload'] - quarterbacks['predicted']).mean():.3f} "
            f"QB1_model={year_accuracy['predicted']:.3f} "
            f"QB1_depth={year_accuracy['depth']:.3f}"
        )

    print("\nPOOLED POINT METRICS")
    for metric in sorted({key[0] for key in point}):
        for (candidate, method), values in point.items():
            if candidate == metric:
                print(
                    f"{metric:20s} {method:11s} n={values[0]:4d} "
                    f"MAE={values[1] / values[0]:.3f} "
                    f"RMSE={np.sqrt(values[2] / values[0]):.3f}"
                )
    print("\nPOOLED DISTRIBUTION METRICS")
    for metric, values in distribution.items():
        print(
            f"{metric:20s} n={values[0]:4d} "
            f"CRPS={values[1] / values[0]:.3f} "
            f"80% coverage={values[2] / values[0]:.3f}"
        )
    print("\nQB1 BY SNAP WORKLOAD")
    for method, values in winners.items():
        print(f"{method:11s} {values[1]}/{values[0]} = {values[1] / values[0]:.3f}")


if __name__ == "__main__":
    main()
