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
