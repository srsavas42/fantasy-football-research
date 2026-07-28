"""Command-line data acquisition and source-configuration checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd

from ffmodel.data import cfbd, ingest, odds, sleeper, weather
from ffmodel.data.identity import load_player_dim
from ffmodel.data.sources import source_status


def _report(name: str, frame: pd.DataFrame) -> None:
    print(f"{name}: {len(frame):,} rows x {len(frame.columns):,} columns")


def _try_pull(name: str, function: Callable[[], pd.DataFrame], strict: bool) -> bool:
    try:
        _report(name, function())
        return True
    except Exception as exc:
        if strict:
            raise
        print(f"WARNING {name}: {exc}", file=sys.stderr)
        return False


def _nflverse_loader(dataset: str, seasons: list[int], args) -> Callable[[], pd.DataFrame]:
    seasons = [
        season
        for season in seasons
        if season >= _MIN_SEASON.get(dataset, season)
        and season <= _MAX_SEASON.get(dataset, season)
    ]
    if not seasons and dataset not in {"players", "contracts", "rankings"}:
        return pd.DataFrame
    simple = {
        "weekly": ingest.load_weekly,
        "pbp": ingest.load_pbp,
        "team_stats": ingest.load_team_stats,
        "schedules": ingest.load_schedules,
        "snaps": ingest.load_snap_counts,
        "rosters": ingest.load_rosters,
        "weekly_rosters": ingest.load_weekly_rosters,
        "injuries": ingest.load_injuries,
        "depth_charts": ingest.load_depth_charts,
        "participation": ingest.load_participation,
        "ftn_charting": ingest.load_ftn_charting,
        "draft_picks": ingest.load_draft_picks,
        "combine": ingest.load_combine,
        "ff_opportunity": ingest.load_ff_opportunity,
    }
    if dataset in simple:
        return lambda: simple[dataset](
            seasons, refresh=args.refresh, cache_dir=args.cache_dir
        )
    if dataset == "players":
        return lambda: load_player_dim(refresh=args.refresh, cache_dir=args.cache_dir)
    if dataset == "contracts":
        return lambda: ingest.load_contracts(refresh=args.refresh, cache_dir=args.cache_dir)
    if dataset == "rankings":
        return lambda: ingest.load_ff_rankings(
            args.ranking_type,
            snapshot_at=args.snapshot_at,
            refresh=args.refresh,
            cache_dir=args.cache_dir,
        )
    if dataset.startswith("ngs_"):
        stat_type = dataset.removeprefix("ngs_")
        return lambda: ingest.load_nextgen_stats(
            seasons, stat_type, refresh=args.refresh, cache_dir=args.cache_dir
        )
    if dataset.startswith("pfr_"):
        stat_type = dataset.removeprefix("pfr_")
        return lambda: ingest.load_pfr_advstats(
            seasons, stat_type, refresh=args.refresh, cache_dir=args.cache_dir
        )
    raise ValueError(f"unknown nflverse dataset {dataset!r}")


NFLVERSE_CHOICES = [
    "players",
    "weekly",
    "team_stats",
    "schedules",
    "snaps",
    "rosters",
    "weekly_rosters",
    "injuries",
    "depth_charts",
    "draft_picks",
    "combine",
    "contracts",
    "rankings",
    "ngs_passing",
    "ngs_receiving",
    "ngs_rushing",
    "pfr_pass",
    "pfr_rec",
    "pfr_rush",
    "participation",
    "ftn_charting",
    "ff_opportunity",
    "pbp",
]

BOOTSTRAP_DATASETS = [
    "players",
    "weekly",
    "team_stats",
    "schedules",
    "snaps",
    "rosters",
    "weekly_rosters",
    "injuries",
    "depth_charts",
    "draft_picks",
    "combine",
    "rankings",
]

_MIN_SEASON = {
    "weekly": 1999,
    "team_stats": 1999,
    "pbp": 1999,
    "weekly_rosters": 2002,
    "depth_charts": 2001,
    "injuries": 2009,
    "snaps": 2012,
    "ngs_passing": 2016,
    "ngs_receiving": 2016,
    "ngs_rushing": 2016,
    "participation": 2016,
    "pfr_pass": 2018,
    "pfr_rec": 2018,
    "pfr_rush": 2018,
    "ftn_charting": 2022,
    "draft_picks": 1980,
    "combine": 2000,
}
_MAX_SEASON = {"injuries": 2024}


def _run_nflverse(args) -> int:
    datasets = list(args.datasets)
    if getattr(args, "include_pbp", False) and "pbp" not in datasets:
        datasets.append("pbp")
    failures = 0
    for dataset in datasets:
        ok = _try_pull(
            f"nflverse/{dataset}",
            _nflverse_loader(dataset, args.seasons, args),
            args.strict,
        )
        failures += not ok
    return 1 if failures else 0


def _run_cfbd(args) -> int:
    functions: dict[str, Callable[[int], pd.DataFrame]] = {
        "stats": lambda year: cfbd.load_player_season_stats(
            year, refresh=args.refresh, cache_dir=args.cache_dir
        ),
        "usage": lambda year: cfbd.load_player_usage(
            year, refresh=args.refresh, cache_dir=args.cache_dir
        ),
        "roster": lambda year: cfbd.load_roster(
            year, refresh=args.refresh, cache_dir=args.cache_dir
        ),
        "recruits": lambda year: cfbd.load_recruits(
            year, refresh=args.refresh, cache_dir=args.cache_dir
        ),
        "draft": lambda year: cfbd.load_draft_picks(
            year, refresh=args.refresh, cache_dir=args.cache_dir
        ),
    }
    planned = [
        (year, dataset)
        for year in args.seasons
        for dataset in args.datasets
        if args.refresh
        or not cfbd.default_request_is_cached(
            dataset, year, cache_dir=args.cache_dir
        )
    ]
    cached = len(args.seasons) * len(args.datasets) - len(planned)
    if len(planned) > args.max_new_requests:
        raise cfbd.CfbdQuotaError(
            f"CFBD pull would make {len(planned)} new requests, above this run's "
            f"--max-new-requests={args.max_new_requests}; split the pull or raise "
            "the cap intentionally"
        )
    local = cfbd.local_request_budget(args.cache_dir)
    if len(planned) > int(local["local_remaining"]):
        raise cfbd.CfbdQuotaError(
            f"local CFBD ledger has {local['local_remaining']} calls remaining, "
            f"but this pull needs {len(planned)}"
        )
    remote = cfbd.account_quota() if planned else {}
    remote_remaining = remote.get("remainingCalls")
    if remote_remaining is not None and len(planned) > int(remote_remaining):
        raise cfbd.CfbdQuotaError(
            f"CFBD reports only {remote_remaining} calls remaining, but this pull "
            f"needs {len(planned)}"
        )
    print(
        f"CFBD plan: {len(planned)} new request(s), {cached} cache hit(s); "
        f"local month ledger {local['local_used']}/{local['local_limit']}; "
        f"provider remaining {remote.get('remainingCalls', 'not queried')}"
    )
    failures = 0
    for year in args.seasons:
        for dataset in args.datasets:
            ok = _try_pull(
                f"cfbd/{dataset}/{year}",
                lambda d=dataset, y=year: functions[d](y),
                args.strict,
            )
            failures += not ok
    return 1 if failures else 0


def _run_cfbd_quota(args) -> int:
    local = cfbd.local_request_budget(args.cache_dir)
    remote = cfbd.account_quota()
    print(
        f"CFBD local ledger: {local['local_used']}/{local['local_limit']} used "
        f"for {local['month']}"
    )
    print(
        "CFBD provider: "
        f"tier={remote.get('tierName', 'unknown')}, "
        f"used={remote.get('usedCalls', 'unknown')}, "
        f"remaining={remote.get('remainingCalls', 'unknown')}, "
        f"limit={remote.get('monthlyLimit', 'unknown')}, "
        f"reset={remote.get('resetAt', 'unknown')}"
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--refresh", action="store_true", help="replace cached artifacts")
    parser.add_argument("--cache-dir", help="override FFMODEL_CACHE_DIR")
    parser.add_argument("--strict", action="store_true", help="stop at the first failed source")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ffmodel-data")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check dependencies and required environment variables")

    nfl = sub.add_parser("nflverse", help="pull nflverse datasets")
    nfl.add_argument("--seasons", nargs="+", type=int, required=True)
    nfl.add_argument("--datasets", nargs="+", choices=NFLVERSE_CHOICES, required=True)
    nfl.add_argument("--ranking-type", choices=["draft", "week", "all"], default="draft")
    nfl.add_argument("--snapshot-at")
    _add_common(nfl)

    bootstrap = sub.add_parser("bootstrap", help="pull the recommended open NFL core")
    bootstrap.add_argument("--seasons", nargs="+", type=int, required=True)
    bootstrap.add_argument("--include-pbp", action="store_true", help="also download large PBP files")
    bootstrap.add_argument("--ranking-type", choices=["draft", "week", "all"], default="draft")
    bootstrap.add_argument("--snapshot-at")
    bootstrap.set_defaults(datasets=BOOTSTRAP_DATASETS)
    _add_common(bootstrap)

    sleeper_parser = sub.add_parser("sleeper", help="archive today's live player snapshot")
    sleeper_parser.add_argument("--snapshot-at")
    _add_common(sleeper_parser)

    college = sub.add_parser("cfbd", help="pull CollegeFootballData prospect inputs")
    college.add_argument("--seasons", nargs="+", type=int, required=True)
    college.add_argument(
        "--datasets", nargs="+", choices=["stats", "usage", "roster", "recruits", "draft"],
        default=["stats", "usage", "roster", "recruits"],
    )
    college.add_argument(
        "--max-new-requests",
        type=int,
        default=100,
        help="hard cap for cache misses in this run (default: 100)",
    )
    _add_common(college)

    cfbd_quota = sub.add_parser(
        "cfbd-quota", help="show local and provider CFBD usage (provider check is free)"
    )
    cfbd_quota.add_argument("--cache-dir", help="override FFMODEL_CACHE_DIR")

    odds_parser = sub.add_parser("odds", help="archive current or historical NFL odds")
    odds_parser.add_argument("--historical-at", help="ISO-8601 prediction cutoff (paid history)")
    odds_parser.add_argument("--snapshot-at", help="cache timestamp for current odds")
    odds_parser.add_argument("--regions", default="us")
    odds_parser.add_argument("--markets", default="spreads,totals")
    _add_common(odds_parser)

    weather_parser = sub.add_parser("weather", help="pull hourly weather for one location")
    weather_parser.add_argument("--latitude", type=float, required=True)
    weather_parser.add_argument("--longitude", type=float, required=True)
    weather_parser.add_argument("--start-date", required=True)
    weather_parser.add_argument("--end-date", required=True)
    weather_parser.add_argument("--historical", action="store_true")
    weather_parser.add_argument(
        "--lead-days", type=int, choices=range(1, 8), help="archived forecast lead time"
    )
    _add_common(weather_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print(source_status().to_string(index=False))
        return 0
    if args.command in {"nflverse", "bootstrap"}:
        return _run_nflverse(args)
    if args.command == "sleeper":
        return 0 if _try_pull(
            "sleeper/players",
            lambda: sleeper.load_players(
                snapshot_at=args.snapshot_at,
                refresh=args.refresh,
                cache_dir=args.cache_dir,
            ),
            args.strict,
        ) else 1
    if args.command == "cfbd":
        return _run_cfbd(args)
    if args.command == "cfbd-quota":
        return _run_cfbd_quota(args)
    if args.command == "odds":
        return 0 if _try_pull(
            "the_odds_api/nfl_odds",
            lambda: odds.load_nfl_odds(
                regions=args.regions,
                markets=args.markets,
                historical_at=args.historical_at,
                snapshot_at=args.snapshot_at,
                refresh=args.refresh,
                cache_dir=args.cache_dir,
            ),
            args.strict,
        ) else 1
    if args.command == "weather":
        if args.lead_days is not None:
            function = lambda: weather.load_previous_run_forecast(
                args.latitude,
                args.longitude,
                args.start_date,
                args.end_date,
                lead_days=args.lead_days,
                refresh=args.refresh,
                cache_dir=args.cache_dir,
            )
        else:
            function = lambda: weather.load_hourly_forecast(
                args.latitude,
                args.longitude,
                args.start_date,
                args.end_date,
                historical=args.historical,
                refresh=args.refresh,
                cache_dir=args.cache_dir,
            )
        return 0 if _try_pull(
            "open_meteo/hourly_forecast",
            function,
            args.strict,
        ) else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
