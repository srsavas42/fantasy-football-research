"""Walk-forward both weekly responses, and report the ladder for each.

Every estimator is fitted on seasons strictly before its holdout and scored on
identical rows, so a rung that does not pay for itself shows up as a rung that
does not pay for itself. Results are written as JSON for the document to quote.

    python scripts/validate_weekly.py --holdouts 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.weekly import FEATURES_CACHE, PANEL_CACHE
from ffmodel.weekly.evaluate import report, walk_forward
from ffmodel.weekly.features import add_features
from ffmodel.weekly.frame import load_panel
from ffmodel.weekly.market import attach_adp
from ffmodel.weekly.news import add_news_features
from ffmodel.weekly.expected import attach_expected
from ffmodel.weekly.pedigree import add_pedigree_features
from ffmodel.weekly.nextweek import next_week_ladder
from ffmodel.weekly.restofseason import (
    OFFSET,
    TARGET,
    add_rest_of_season_target,
    rest_of_season_ladder,
)

COLUMNS = [
    "mae",
    "rmse",
    "crps",
    "bias",
    "coverage_80",
    "coverage_95",
    "within_group_spearman",
    "within_group_top_k",
    "pit_deviation",
]


def _show(results: dict, title: str, populations=("relevant", "panel")) -> None:
    print(f"\n=== {title} ===")
    for population in populations:
        table = report(results, population)
        if table.empty:
            continue
        keep = ["estimator", "n"] + [c for c in COLUMNS if c in table.columns]
        print(f"\n-- {population} --")
        print(table[keep].round(4).to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--seasons", type=int, nargs=2, default=[2016, 2025])
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument(
        "--features", type=Path, default=FEATURES_CACHE
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--only", choices=["next-week", "rest-of-season"], default=None
    )
    args = parser.parse_args(argv)

    if args.features.exists():
        frame = pd.read_pickle(args.features)
    else:
        panel = load_panel(range(args.seasons[0], args.seasons[1] + 1))
        # ADP is attached before the features so the market curve and every
        # model see exactly the same rows.
        frame = add_pedigree_features(
            add_news_features(add_features(attach_expected(attach_adp(panel))))
        )
        args.features.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(args.features)
    print(f"panel {frame.shape[0]} rows, seasons {frame.season.min()}-{frame.season.max()}")

    payload: dict[str, object] = {}

    if args.only in (None, "next-week"):
        weekly = walk_forward(
            add_rest_of_season_target(frame),
            next_week_ladder(),
            target="points",
            holdouts=args.holdouts,
            draws=args.draws,
        )
        # Broken out by week because the ADP baseline goes stale as the season
        # runs: in week 1 the board knows everything anyone knows, and by week
        # 12 the model has eleven weeks the board has never seen.
        _show(
            weekly,
            "Model 1: next week",
            populations=("relevant", "panel", "relevant_early", "relevant_mid", "relevant_late"),
        )
        payload["next_week"] = weekly

    if args.only in (None, "rest-of-season"):
        seasonal = add_rest_of_season_target(frame)
        totals = walk_forward(
            seasonal,
            rest_of_season_ladder(),
            target=TARGET,
            holdouts=args.holdouts,
            draws=args.draws,
        )
        _show(
            totals,
            "Model 2: rest of season",
            populations=(
                "relevant",
                "panel",
                "relevant_early",
                "relevant_mid",
                "relevant_late",
            ),
        )
        payload["rest_of_season"] = totals

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
