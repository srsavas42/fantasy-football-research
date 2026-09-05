"""Walk-forward the kicker and team-defense responses.

Two panels the package has never modelled, scored the way every other rung in it
is scored: fitted on seasons strictly before the holdout, reported against a
climatology floor and a persistence baseline, and broken out by week because a
start/sit decision in week 2 and one in week 15 are not the same decision.

Both ladders end on a weather rung, because the physical conditions have a more
credible claim on a kicked ball than on anything the skill panel does -- and that
claim is measured here rather than assumed. The same ceiling caveat applies: the
readings are recorded at the game, so the rung reports what perfect foreknowledge
would buy.

    python scripts/validate_specialists.py --holdouts 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.weekly.evaluate import report, walk_forward
from ffmodel.weekly.restofseason import TARGET, add_rest_of_season_target
from ffmodel.weekly.specialists import (
    add_defense_features,
    add_league_baseline,
    add_kicker_features,
    attach_market,
    build_defense_panel,
    build_kicker_panel,
    defense_ladder,
    defense_season_ladder,
    kicker_ladder,
    kicker_season_ladder,
)
from ffmodel.weekly.weather import attach_weather

KICKER_CACHE = Path(".cache/weekly_kickers_2016_2025.pkl")
DEFENSE_CACHE = Path(".cache/weekly_defenses_2016_2025.pkl")

COLUMNS = [
    "mae",
    "rmse",
    "crps",
    "bias",
    "coverage_80",
    "coverage_95",
    "within_group_spearman",
    "pit_deviation",
]


def build_kickers(seasons: range) -> pd.DataFrame:
    return add_league_baseline(
        add_kicker_features(attach_weather(attach_market(build_kicker_panel(seasons))))
    )


def build_defenses(seasons: range) -> pd.DataFrame:
    return add_league_baseline(
        add_defense_features(attach_weather(attach_market(build_defense_panel(seasons))))
    )


def _load(cache: Path, builder, seasons: range, refresh: bool) -> pd.DataFrame:
    if cache.exists() and not refresh:
        return pd.read_pickle(cache)
    frame = builder(seasons)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache)
    return frame


def _show(results: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    for population in ("relevant", "panel", "relevant_early", "relevant_mid", "relevant_late"):
        table = report(results, population)
        if table.empty:
            continue
        keep = ["estimator", "n"] + [c for c in COLUMNS if c in table.columns]
        print(f"\n-- {population} --")
        print(table[keep].round(4).to_string(index=False))

    rows = []
    for fold in results.get("folds", []):
        entry = {"holdout": fold["holdout"]}
        for name, block in fold.get("estimators", {}).items():
            scored = block.get("relevant")
            if scored is not None:
                entry[name] = round(float(scored["crps"]), 4)
        rows.append(entry)
    if rows:
        print("\n-- CRPS by fold, relevant population --")
        print(pd.DataFrame(rows).to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--seasons", type=int, nargs=2, default=[2016, 2025])
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--only", choices=["kicker", "defense"], default=None)
    parser.add_argument(
        "--response", choices=["next-week", "rest-of-season"], default=None
    )
    args = parser.parse_args(argv)

    seasons = range(args.seasons[0], args.seasons[1] + 1)
    payload: dict[str, object] = {}

    if args.only in (None, "kicker"):
        kickers = _load(KICKER_CACHE, build_kickers, seasons, args.refresh)
        print(
            f"kickers {kickers.shape[0]} rows, "
            f"{kickers.player_key.nunique()} kickers, "
            f"played {kickers.played.mean():.1%}"
        )
        if args.response in (None, "next-week"):
            results = walk_forward(
                kickers,
                kicker_ladder(),
                target="points",
                holdouts=args.holdouts,
                draws=args.draws,
            )
            _show(results, "Kickers: next week")
            payload["kicker"] = results
        if args.response in (None, "rest-of-season"):
            totals = walk_forward(
                add_rest_of_season_target(kickers),
                kicker_season_ladder(),
                target=TARGET,
                holdouts=args.holdouts,
                draws=args.draws,
            )
            _show(totals, "Kickers: rest of season")
            payload["kicker_rest_of_season"] = totals

    if args.only in (None, "defense"):
        defenses = _load(DEFENSE_CACHE, build_defenses, seasons, args.refresh)
        print(f"\ndefenses {defenses.shape[0]} rows, {defenses.player_key.nunique()} clubs")
        if args.response in (None, "next-week"):
            results = walk_forward(
                defenses,
                defense_ladder(),
                target="points",
                holdouts=args.holdouts,
                draws=args.draws,
            )
            _show(results, "Team defenses: next week")
            payload["defense"] = results
        if args.response in (None, "rest-of-season"):
            totals = walk_forward(
                add_rest_of_season_target(defenses),
                defense_season_ladder(),
                target=TARGET,
                holdouts=args.holdouts,
                draws=args.draws,
            )
            _show(totals, "Team defenses: rest of season")
            payload["defense_rest_of_season"] = totals

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
