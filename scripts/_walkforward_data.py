"""Shared cached nflverse frames for the walk-forward validation scripts.

Building the season-average dataset takes a couple of minutes and every
comparison needs the *same* frames — a candidate scored against a baseline built
from a separate pull is not a controlled comparison. These helpers cache one
build and hand the identical frames to every configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CACHE = Path(".cache/ffmodel-walkforward")
DEFAULT_SEASONS = range(2014, 2025)
HOLDOUTS = (2022, 2023, 2024)


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("label", help="name for the output JSON, e.g. 'baseline'")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--holdouts", nargs="+", type=int, default=list(HOLDOUTS)
    )
    parser.add_argument(
        "--couple-gate",
        action="store_true",
        help="force the availability-coupled QB gate on (it is on by default)",
    )
    parser.add_argument(
        "--no-couple-gate",
        action="store_true",
        help="force the availability-coupled QB gate off, for ablations",
    )
    parser.add_argument(
        "--play-transition",
        action="store_true",
        help="restore the per-row play-rate random effect (redundant with the "
             "NegativeBinomial's own dispersion; off by default)",
    )
    parser.add_argument(
        "--postseason",
        action="store_true",
        help="enable the lagged postseason role features",
    )
    parser.add_argument(
        "--calibrated-innovation",
        action="store_true",
        help="solve for the input noise scale that realizes the churn the "
             "estimator measured, instead of using the measurement directly",
    )
    parser.add_argument(
        "--mean-preserving-innovation",
        action="store_true",
        help="correct the softmax renormalization bias the role innovation "
             "introduces in the workload, target and carry allocations",
    )
    parser.add_argument(
        "--efficiency-exposure-floor",
        type=int,
        default=None,
        help="lower every efficiency response's training exposure floor to at "
             "most this many opportunities (scoring runs only)",
    )
    parser.add_argument(
        "--volume-feature-draws",
        type=int,
        default=200,
        help="sampler budget for the per-season cross-fits that build the "
             "training-time oof_* covariates under --volume-feature-estimator "
             "pipeline. Only their posterior *mean* is consumed, so this can be "
             "far cheaper than the fits being validated (scoring runs only)",
    )
    parser.add_argument(
        "--volume-feature-estimator",
        choices=("ridge", "pipeline"),
        default=None,
        help="how training-time oof_* volume covariates are built (scoring runs only)",
    )
    return parser


def load_frames(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached player/team season rows, building them on first use."""
    cache_dir = Path(cache_dir)
    player_path = cache_dir / "player_rows.pkl"
    team_path = cache_dir / "team_rows.pkl"
    if player_path.exists() and team_path.exists():
        return pd.read_pickle(player_path), pd.read_pickle(team_path)

    from ffmodel.features.season_average import build_season_average_data

    data = build_season_average_data(
        DEFAULT_SEASONS, source="nflverse", roster_mode="point_in_time"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    data.player_rows.to_pickle(player_path)
    data.team_rows.to_pickle(team_path)
    return data.player_rows, data.team_rows


def gate_override(args: argparse.Namespace) -> bool | None:
    """Explicit coupling choice, or None to keep the model's own default."""
    if args.couple_gate and args.no_couple_gate:
        raise SystemExit("choose at most one of --couple-gate / --no-couple-gate")
    if args.couple_gate:
        return True
    if args.no_couple_gate:
        return False
    return None
