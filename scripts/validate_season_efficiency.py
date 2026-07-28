"""Walk-forward validation for the season-average efficiency handoff.

This script compares volume with/without lagged efficiency, cross-fits volume
features for the future-efficiency models, and optionally tests nflverse
pass-play participation as a route-opportunity proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ffmodel.data import ingest
from ffmodel.evaluation.efficiency_season_average import (
    run_efficiency_ablation,
    volume_ablation_fold_metrics,
    volume_ablation_metrics,
)
from ffmodel.evaluation.qb_efficiency_volume import (
    qb_layer_efficiency_fold_metrics,
    qb_layer_efficiency_metrics,
)
from ffmodel.features.participation import attach_lagged_participation_features
from ffmodel.features.season_average import SeasonAverageData, build_season_average_data


def _print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=False))


def _fold_stability(folds: pd.DataFrame) -> pd.DataFrame:
    baseline = folds[folds["model"].eq("volume_only")].set_index(
        ["stream", "season"]
    )["mae"]
    records = []
    for (model, stream), rows in folds[
        folds["model"].ne("volume_only")
    ].groupby(["model", "stream"]):
        keys = pd.MultiIndex.from_frame(rows[["stream", "season"]])
        prior = baseline.reindex(keys).to_numpy(dtype=float)
        wins = rows["mae"].to_numpy(dtype=float) < prior
        recent = rows["season"].ge(2019).to_numpy()
        records.append(
            {
                "model": model,
                "stream": stream,
                "fold_wins": int(wins.sum()),
                "folds": int(len(wins)),
                "recent_wins": int(wins[recent].sum()),
                "recent_folds": int(recent.sum()),
            }
        )
    return pd.DataFrame(records)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2014, 2025)))
    parser.add_argument("--source", choices=("auto", "legacy", "nflverse"), default="nflverse")
    parser.add_argument(
        "--roster-mode",
        choices=("auto", "point_in_time", "inferred"),
        default="point_in_time",
    )
    parser.add_argument("--volume-alpha", type=float, default=300.0)
    parser.add_argument("--efficiency-alpha", type=float, default=500.0)
    parser.add_argument("--min-training-seasons", type=int, default=2)
    parser.add_argument("--participation", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/season-average-validation/efficiency-v1"),
    )
    args = parser.parse_args(argv)

    data = build_season_average_data(
        args.seasons,
        source=args.source,
        roster_mode=args.roster_mode,
    )
    print(
        f"team-seasons={len(data.team_rows):,}; player-seasons={len(data.player_rows):,}; "
        f"response seasons={int(data.player_rows.season.min())}-{int(data.player_rows.season.max())}"
    )

    volume = volume_ablation_metrics(data, alpha=args.volume_alpha)
    volume_folds = volume_ablation_fold_metrics(data, alpha=args.volume_alpha)
    qb_layer_folds = qb_layer_efficiency_fold_metrics(data)
    qb_layer = qb_layer_efficiency_metrics(qb_layer_folds)
    efficiency = run_efficiency_ablation(
        data,
        volume_alpha=args.volume_alpha,
        efficiency_alpha=args.efficiency_alpha,
        min_training_seasons=args.min_training_seasons,
    )
    _print_frame("Volume ablation", volume)
    _print_frame("Volume fold stability", _fold_stability(volume_folds))
    _print_frame("Production-structure QB layer ablation", qb_layer)
    _print_frame("Efficiency ablation", efficiency)

    participation_volume = pd.DataFrame()
    participation_efficiency = pd.DataFrame()
    participation_coverage = 0
    if args.participation:
        seasons = [season for season in args.seasons if 2016 <= season <= 2024]
        frames = [ingest.load_participation([season]) for season in seasons]
        participation = pd.concat(frames, ignore_index=True, sort=False)
        enriched_rows = attach_lagged_participation_features(
            data.player_rows,
            participation,
            players=ingest.load_ids(),
        )
        participation_coverage = int(enriched_rows["prior_participation_available"].sum())
        enriched = SeasonAverageData(data.team_rows, enriched_rows)
        participation_volume = volume_ablation_metrics(
            enriched, alpha=args.volume_alpha
        )
        participation_efficiency = run_efficiency_ablation(
            enriched,
            volume_alpha=args.volume_alpha,
            efficiency_alpha=args.efficiency_alpha,
            min_training_seasons=args.min_training_seasons,
        )
        print(
            f"\nParticipation coverage={participation_coverage:,}/"
            f"{len(enriched_rows):,} player-seasons"
        )
        _print_frame("Participation volume ablation", participation_volume)
        _print_frame("Participation efficiency ablation", participation_efficiency)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    volume.to_csv(args.output_dir / "volume-ablation.csv", index=False)
    volume_folds.to_csv(args.output_dir / "volume-fold-ablation.csv", index=False)
    qb_layer.to_csv(args.output_dir / "qb-layer-ablation.csv", index=False)
    qb_layer_folds.to_csv(args.output_dir / "qb-layer-fold-ablation.csv", index=False)
    efficiency.to_csv(args.output_dir / "efficiency-ablation.csv", index=False)
    if not participation_volume.empty:
        participation_volume.to_csv(
            args.output_dir / "participation-volume-ablation.csv", index=False
        )
        participation_efficiency.to_csv(
            args.output_dir / "participation-efficiency-ablation.csv", index=False
        )
    report = {
        "seasons": list(map(int, args.seasons)),
        "volume_alpha": args.volume_alpha,
        "efficiency_alpha": args.efficiency_alpha,
        "team_seasons": len(data.team_rows),
        "player_seasons": len(data.player_rows),
        "participation_coverage": participation_coverage,
        "volume": volume.to_dict(orient="records"),
        "volume_folds": volume_folds.to_dict(orient="records"),
        "qb_layer": qb_layer.to_dict(orient="records"),
        "qb_layer_folds": qb_layer_folds.to_dict(orient="records"),
        "efficiency": efficiency.to_dict(orient="records"),
        "participation_volume": participation_volume.to_dict(orient="records"),
        "participation_efficiency": participation_efficiency.to_dict(orient="records"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nwrote validation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
