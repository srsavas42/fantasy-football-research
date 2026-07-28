"""Walk-forward conditional efficiency-to-volume screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffmodel.evaluation.conditional_efficiency_volume import (
    conditional_volume_fold_metrics,
    conditional_volume_metrics,
)
from ffmodel.features.season_average import build_season_average_data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2014, 2025)))
    parser.add_argument("--alpha", type=float, default=300.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/season-average-validation/efficiency-conditional"),
    )
    args = parser.parse_args(argv)

    data = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    )
    folds = conditional_volume_fold_metrics(data, alpha=args.alpha)
    summary = conditional_volume_metrics(folds)
    print(summary.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "folds.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    report = {
        "seasons": list(map(int, args.seasons)),
        "alpha": args.alpha,
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
