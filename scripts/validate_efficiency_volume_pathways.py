"""Run indirect efficiency/production-to-volume pathway tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ffmodel.data import load_player_weeks
from ffmodel.evaluation.efficiency_volume_pathways import (
    competition_pathway_fold_metrics,
    pathway_summary,
    player_pathway_fold_metrics,
    team_pathway_fold_metrics,
)
from ffmodel.features.season_average import build_season_average_data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2014, 2025)))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/season-average-validation/efficiency-pathways"),
    )
    args = parser.parse_args(argv)

    data = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    )
    weeks = load_player_weeks(args.seasons, source="nflverse")
    player = player_pathway_fold_metrics(data)
    team = team_pathway_fold_metrics(data, weeks)
    competition = competition_pathway_fold_metrics(data)
    folds = pd.concat([player, team, competition], ignore_index=True)
    summary = pathway_summary(folds)
    print(summary.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "folds.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    report = {
        "seasons": list(map(int, args.seasons)),
        "team_seasons": len(data.team_rows),
        "player_seasons": len(data.player_rows),
        "summary": summary.to_dict(orient="records"),
        "folds": folds.to_dict(orient="records"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nwrote validation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
