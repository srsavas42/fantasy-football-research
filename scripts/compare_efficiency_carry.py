"""Posterior-controlled rushing-efficiency ablation for carry allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ffmodel.evaluation.metrics import empirical_crps
from ffmodel.features.season_average import SeasonAverageData, build_season_average_data
from ffmodel.models.base import save_idata
from ffmodel.models.volume_season_average import (
    SeasonAverageVolumePipeline,
    SeasonRosterShareModel,
)


CARRY_EFFICIENCY_CONFIGURATIONS = {
    "rush_epa": ("prior_rush_epa_per_carry",),
    "rush_epa_room": (
        "prior_rush_epa_per_carry",
        "prior_rush_epa_per_carry_centered_x_room",
    ),
}


def _metrics(prediction) -> dict[str, float]:
    rows = prediction.player_rows
    named = rows["is_replacement_player"].ne(1).to_numpy()
    observed = rows["rush_att"].to_numpy(dtype=float) / rows[
        "team_games"
    ].to_numpy(dtype=float)
    samples = prediction.carries_per_team_game
    error = samples.mean(axis=1)[named] - observed[named]
    return {
        "n": int(named.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "crps": float(empirical_crps(observed[named], samples[named]).mean()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2014, 2025)))
    parser.add_argument("--holdouts", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--nuts-sampler", choices=("pymc", "nutpie"), default="nutpie")
    parser.add_argument(
        "--feature-set",
        choices=tuple(CARRY_EFFICIENCY_CONFIGURATIONS),
        default="rush_epa",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(".cache/season-average-validation/volume-v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/season-average-validation/efficiency-carry"),
    )
    args = parser.parse_args(argv)
    candidate_label = args.feature_set
    candidate_features = CARRY_EFFICIENCY_CONFIGURATIONS[candidate_label]
    if args.nuts_sampler == "nutpie" and "NUMBA_CACHE_DIR" not in os.environ:
        os.environ["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "ffmodel-numba")

    data = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    )
    records = []
    for holdout in args.holdouts:
        train = SeasonAverageData(
            data.team_rows[data.team_rows["season"] < holdout].copy(),
            data.player_rows[data.player_rows["season"] < holdout].copy(),
        )
        test = SeasonAverageData(
            data.team_rows[data.team_rows["season"] == holdout].copy(),
            data.player_rows[data.player_rows["season"] == holdout].copy(),
        )
        baseline_path = args.baseline_dir / f"holdout-{holdout}-final"
        pipeline = SeasonAverageVolumePipeline.load(baseline_path)
        baseline = pipeline.predict_samples(test, seed=42)

        challenger = SeasonRosterShareModel(
            "carry",
            extra_efficiency_features=candidate_features,
        ).fit(
            train.player_rows,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            nuts_sampler=args.nuts_sampler,
        )
        pipeline.carry_model = challenger
        candidate = pipeline.predict_samples(test, seed=42)
        holdout_dir = args.output_dir / f"holdout-{holdout}"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        save_idata(challenger.idata, holdout_dir / "carry.nc")
        for label, prediction in (("volume_v2", baseline), (candidate_label, candidate)):
            result = {"season": int(holdout), "model": label, **_metrics(prediction)}
            records.append(result)
            print(result)

    pooled = []
    for label in ("volume_v2", candidate_label):
        subset = [record for record in records if record["model"] == label]
        n = sum(record["n"] for record in subset)
        pooled.append(
            {
                "model": label,
                "n": n,
                "mae": sum(record["n"] * record["mae"] for record in subset) / n,
                "rmse": float(
                    np.sqrt(
                        sum(record["n"] * record["rmse"] ** 2 for record in subset)
                        / n
                    )
                ),
                "crps": sum(record["n"] * record["crps"] for record in subset) / n,
            }
        )
    report = {
        "feature_set": candidate_label,
        "features": list(candidate_features),
        "folds": records,
        "pooled": pooled,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["pooled"], indent=2))


if __name__ == "__main__":
    main()
