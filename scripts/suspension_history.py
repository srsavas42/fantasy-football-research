"""The suspension record the roster feed can reconstruct, and its limits.

Two questions this answers, which need different treatment:

*How long is a ban that was announced before the season?* Known exactly, so
this prints the population for reference rather than to fit anything to.

*How long does an open-ended absence last -- the exempt list, an indefinite
ban?* This is a real duration problem, and the answer here is that the roster
feed cannot support a fitted one. The sample is single figures once the COVID
season is netted out, and it is right-censored in a direction that makes it look
shorter than it is. The table is printed with the censoring beside it so that is
visible rather than inferred.

    python scripts/suspension_history.py --seasons 2016 2025
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.data import ingest
from ffmodel.features.suspensions import (
    exempt_duration_table,
    preseason_suspension_games,
    suspension_spells,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs=2, default=(2016, 2025))
    parser.add_argument("--output", type=Path, default=Path("reports/suspensions.json"))
    args = parser.parse_args(argv)

    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    rosters = ingest.load_weekly_rosters(seasons)
    spells = suspension_spells(rosters)

    print(f"=== skill-position suspension spells, {seasons[0]}-{seasons[-1]} ===")
    summary = spells.groupby("susp_kind").agg(
        spells=("flagged_weeks", "size"),
        median=("flagged_weeks", "median"),
        mean=("flagged_weeks", "mean"),
        longest=("flagged_weeks", "max"),
        censored=("censored", "sum"),
    )
    print(summary.to_string())

    known = preseason_suspension_games(rosters)
    print(f"\n=== definite bans in force at week 1 (n={len(known)}) ===")
    print("These are arithmetic, not risk: the length is public in August.")
    counts = known["suspended_games"].value_counts().sort_index()
    print("\ngames  n")
    for games, n in counts.items():
        print(f"{int(games):>5}  {n}")
    print(f"\nper season: {len(known) / len(seasons):.1f}")

    print("\n=== open-ended absences, COVID season removed ===")
    open_ended = exempt_duration_table(rosters)
    if open_ended.empty:
        print("none")
    else:
        print(
            open_ended[
                [
                    "season", "player_name", "team", "susp_kind",
                    "flagged_weeks", "roster_absent_weeks", "censored",
                ]
            ].to_string(index=False)
        )
        print(
            f"\nn={len(open_ended)}, of which {int(open_ended['censored'].sum())} "
            "censored. Too few to fit a hazard to; see features/suspensions.py."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "seasons": seasons,
                "by_kind": summary.to_dict("index"),
                "preseason_definite": known.to_dict("records"),
                "preseason_length_counts": {
                    int(k): int(v) for k, v in counts.items()
                },
                "open_ended": open_ended.to_dict("records"),
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
