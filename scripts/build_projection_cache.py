"""Build the season-average cache, including rows for an unplayed season.

A projection season has no feeds to read, so it needs a roster handed to it.
nflverse serves two: ``load_rosters_weekly`` refuses a season whose week 1 has
not happened, and ``load_rosters`` will serve it, so observed seasons take the
weekly snapshot and the projection season takes the season-level one. Mixing
them is deliberate and the reason this is a script rather than a call: the two
feeds carry different columns, and the merge has to happen before
``build_season_average_data`` sees either.

Rebuild whenever the feature layer changes. A cache built before a column
existed does not have it, and a validation run against such a cache silently
scores the challenger as the baseline -- which is why the walk-forward scripts
refuse a cache missing the columns they need.

    python scripts/build_projection_cache.py --seasons 2014 2026 \
        --projection-season 2026 --out .cache/ffmodel-2026
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.features.season_average import (
    build_season_average_data,
    preseason_roster_snapshot,
)


def _projection_roster(season: int) -> pd.DataFrame:
    """Season-level roster for a season the weekly feed will not serve yet."""
    import nflreadpy as nfl

    rosters = nfl.load_rosters(seasons=[season]).to_pandas()
    if rosters.empty:
        raise SystemExit(
            f"nflverse served no {season} roster rows; the projection season "
            "cannot be built without one"
        )
    # The weekly feed's snapshot builder wants a week and a game type; the
    # season feed carries neither, so present it as week 1 of the regular
    # season, which is what a preseason snapshot means.
    rosters = rosters.copy()
    rosters["week"] = 1
    rosters["game_type"] = "REG"
    return rosters


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs=2, default=(2014, 2026))
    parser.add_argument("--projection-season", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path(".cache/ffmodel-2026"))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    projection = int(args.projection_season)
    observed = [s for s in seasons if s != projection]

    import nflreadpy as nfl

    weekly = nfl.load_rosters_weekly(seasons=observed).to_pandas()
    depth = None
    snapshot = preseason_roster_snapshot(weekly, depth)
    if projection in seasons:
        forward = preseason_roster_snapshot(_projection_roster(projection), depth)
        snapshot = pd.concat([snapshot, forward], ignore_index=True)
    print(f"roster snapshot {len(snapshot)} rows", flush=True)

    data = build_season_average_data(
        seasons,
        source="nflverse",
        roster_mode="point_in_time",
        roster_snapshot=snapshot,
        projection_seasons=[projection] if projection in seasons else (),
    )
    print(f"player_rows {data.player_rows.shape}", flush=True)
    print(f"team_rows   {data.team_rows.shape}", flush=True)
    if projection in seasons:
        print(f"{projection} rows {int(data.player_rows.season.eq(projection).sum())}")

    args.out.mkdir(parents=True, exist_ok=True)
    data.player_rows.to_pickle(args.out / "player_rows.pkl")
    data.team_rows.to_pickle(args.out / "team_rows.pkl")
    for name in (
        "roster_reserve", "roster_suspended", "suspended_games",
        "roster_injured_reserve", "roster_pup", "roster_nfi",
        "mandatory_missed_games",
    ):
        if name in data.player_rows:
            total = pd.to_numeric(data.player_rows[name], errors="coerce").fillna(0).gt(0).sum()
            print(f"  {name:24s} nonzero on {int(total)} rows")
        else:
            print(f"  {name:24s} MISSING")
    print(f"wrote {args.out} in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
