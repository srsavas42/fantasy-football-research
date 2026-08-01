"""Decompose total-scoring coverage by position and projected size.

Scoring v1 was rejected on pooled coverage, and every repair attempted after it
was a global transform: dispersion scaling, point shrink and expand, a residual
copula, a draw-conditioned handoff. All of them assume the miscalibration is
uniform. This checks that assumption before another one is designed.

    python scripts/diagnose_scoring_coverage.py --holdout 2024
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ffmodel.data import load_player_weeks
from ffmodel.evaluation.metrics import coverage_by_group
from ffmodel.features import crossseason
from ffmodel.features.season_average import (
    build_projection_data,
    build_season_average_data,
    load_preseason_roster_snapshot,
)
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points


def realized_points(season: int, scoring: str) -> pd.DataFrame:
    """Season fantasy points scoped to the team a player is joined on."""
    weeks = load_player_weeks([season])
    weeks["player_key"] = crossseason.player_key(weeks)
    totals = weeks.groupby(["player_key", "team"], as_index=False).sum(numeric_only=True)
    totals["actual"] = fantasy_points(totals, scoring)
    return totals[["player_key", "team", "actual"]]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument("--first-season", type=int, default=2015)
    parser.add_argument("--scoring", default="ppr", choices=("standard", "half_ppr", "ppr"))
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--report-json", default=None)
    args = parser.parse_args(argv)

    history = range(args.first_season, args.holdout)
    train = build_season_average_data(history, source="auto", roster_mode="point_in_time")
    snapshot = load_preseason_roster_snapshot([args.holdout], cutoff_week=1)
    test = build_projection_data(
        args.holdout, roster_snapshot=snapshot, history_seasons=history
    )

    sample_kwargs = dict(draws=args.draws, tune=args.draws, chains=args.chains, seed=0)
    pipeline = SeasonAverageScoringPipeline().fit(
        train,
        volume_sample_kwargs=sample_kwargs,
        efficiency_sample_kwargs=sample_kwargs,
    )
    prediction = pipeline.predict_samples(test, seed=0)
    points = prediction.fantasy_points[args.scoring]

    rows = prediction.player_rows.reset_index(drop=True)
    frame = rows[["team", "player_key", "position", "player_name"]].reset_index()
    frame = frame.merge(
        realized_points(args.holdout, args.scoring), on=["player_key", "team"], how="inner"
    )
    aligned = points[frame["index"].to_numpy()]
    observed = frame["actual"].to_numpy(dtype=float)
    frame["projected"] = np.median(aligned, axis=1)

    report = {
        "holdout": args.holdout,
        "scoring": args.scoring,
        "n": int(len(frame)),
        "pooled": coverage_by_group(observed, aligned, ["pooled"] * len(frame)),
        "by_position": coverage_by_group(observed, aligned, frame["position"].to_numpy()),
    }
    quartile = pd.qcut(frame["projected"], 4, labels=["q1", "q2", "q3", "q4"])
    report["by_projected_quartile"] = coverage_by_group(
        observed, aligned, quartile.astype(str).to_numpy()
    )

    for section in ("pooled", "by_position", "by_projected_quartile"):
        print(f"\n=== {section} ===")
        for label, entry in sorted(report[section].items(), key=lambda item: str(item[0])):
            levels = " ".join(
                f"{int(level * 100)}%={value:.3f}"
                for level, value in sorted(entry["coverage"].items())
            )
            print(
                f"  {str(label):8s} n={entry['n']:4d}  {levels}"
                f"  below={entry['below']:3d} above={entry['above']:3d}"
            )

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, default=str)
        print(f"\nwrote {args.report_json}")


if __name__ == "__main__":
    main()
