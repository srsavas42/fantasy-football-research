"""What is actually knowable about next season's games played?

Three questions the availability layer's design turns on, all answered from
nflverse weekly rosters rather than from argument.

**1. How much of next season's availability is forecastable at all?** The layer
projects a posterior mean per player, and the spread of those means across
players is bounded by how much year-over-year signal exists. If the realized
year-over-year correlation is r and the realized spread is s, the best possible
history-only forecast has spread r*s. Comparing that with the shipped
`projected_games` says whether a narrow projection is a modelling failure or an
honest reading of a weak signal.

**2. Does a prior-season injury add anything beyond prior-season availability?**
The obvious feature -- "he got hurt last year, and badly" -- is already implied
by "he played eleven games last year", which the layer carries as
`prior_availability`. Splitting next-season availability on whether the player
finished season Y on a reserve list, *holding his season-Y active weeks fixed*,
separates the two. This is the cheap version of the question
`docs/injury-availability-2026-08.md` answered expensively and inconclusively
over six holdouts.

**3. What is the Week-1 roster status worth?** `roster_reserve` is already an
availability feature. Its historical meaning is a post-cutdown reserve
designation, which carries a mandatory multi-game absence. Measuring what each
Week-1 status predicts gives the coefficient's real scale -- and shows what a
live preseason snapshot would be asserting if it mapped a July designation onto
the same column.

The script also reports the statuses nflverse emits at Week 1 against
`features.season_average.ROSTER_STATUSES`, because a status the snapshot does
not list is not a shaded-down player: it is a player dropped from the roster.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.data import ingest
from ffmodel.features.season_average import ROSTER_STATUSES
from ffmodel.features.volume import MODEL_POSITIONS

# A prior season short enough to carry no role information tells us nothing
# about the player's next one, and including such rows measures roster churn
# rather than availability.
MIN_PRIOR_ACTIVE_WEEKS = 4

# Statuses that mean the player was on a reserve list when the season ended.
# `RSN` is reserve/season; `PUP` and `INA` are included because the question is
# "did he finish the year unavailable", not which list he finished it on.
SEASON_ENDING_STATUSES = ("RES", "PUP", "INA", "RSN")


def season_rows(rosters: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season: active weeks, and how the season ended."""
    out = rosters.copy()
    out["status"] = out["status"].astype(str).str.upper()
    out["position"] = out["position"].astype(str).str.upper()
    out = out[out["position"].isin(MODEL_POSITIONS)]
    out = out[out["game_type"].astype(str).eq("REG")]
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    out = out[out["week"].notna() & out["gsis_id"].notna()]

    team_weeks = out.groupby("season")["week"].max().rename("team_weeks")
    active = (
        out[out["status"].eq("ACT")]
        .groupby(["season", "gsis_id"])["week"]
        .nunique()
        .rename("active_weeks")
    )
    final = (
        out.sort_values("week")
        .groupby(["season", "gsis_id"])["status"]
        .last()
        .rename("final_status")
    )
    position = (
        out.sort_values("week")
        .groupby(["season", "gsis_id"])["position"]
        .last()
        .rename("position")
    )
    frame = pd.concat([active, final, position], axis=1).reset_index()
    frame = frame.join(team_weeks, on="season")
    frame["active_weeks"] = frame["active_weeks"].fillna(0.0)
    frame["availability"] = frame["active_weeks"] / frame["team_weeks"]
    frame["ended_on_reserve"] = (
        frame["final_status"].isin(SEASON_ENDING_STATUSES).astype(int)
    )
    return frame


def pairs(frame: pd.DataFrame) -> pd.DataFrame:
    later = frame[
        ["season", "gsis_id", "active_weeks", "availability", "team_weeks"]
    ].copy()
    later["season"] -= 1
    later = later.rename(
        columns={
            "active_weeks": "active_weeks_next",
            "availability": "availability_next",
            "team_weeks": "team_weeks_next",
        }
    )
    return frame.merge(later, on=["season", "gsis_id"], how="inner")


def forecastable(paired: pd.DataFrame) -> pd.DataFrame:
    """The ceiling on a history-only availability forecast's spread.

    ``predictable_sd_games`` is r * sd(realized games). No forecast built from
    the prior season alone can spread its player means wider than this and stay
    calibrated, so it is the number the shipped projection's spread should be
    read against.
    """
    rows = []
    for position, block in paired.groupby("position"):
        if len(block) < 100:
            continue
        r = float(np.corrcoef(block["availability"], block["availability_next"])[0, 1])
        realized = float(block["active_weeks_next"].std())
        rows.append(
            {
                "position": position,
                "n": len(block),
                "yoy_r": r,
                "realized_sd_games": realized,
                "predictable_sd_games": abs(r) * realized,
            }
        )
    pooled_r = float(
        np.corrcoef(paired["availability"], paired["availability_next"])[0, 1]
    )
    pooled_sd = float(paired["active_weeks_next"].std())
    rows.append(
        {
            "position": "ALL",
            "n": len(paired),
            "yoy_r": pooled_r,
            "realized_sd_games": pooled_sd,
            "predictable_sd_games": abs(pooled_r) * pooled_sd,
        }
    )
    return pd.DataFrame(rows).set_index("position")


def injury_increment(paired: pd.DataFrame) -> pd.DataFrame:
    """Next-season games by prior-season ending, holding prior games fixed.

    Without the control this comparison mostly restates that a player who went
    on reserve played fewer weeks, which `prior_availability` already carries.
    With it, the column is what a prior-injury feature could add.
    """
    out = paired.copy()
    out["prior_active_bucket"] = pd.cut(
        out["active_weeks"],
        [MIN_PRIOR_ACTIVE_WEEKS - 1, 8, 12, 15, 19],
        labels=["4-8", "9-12", "13-15", "16+"],
    )
    table = (
        out.groupby(["prior_active_bucket", "ended_on_reserve"], observed=True)
        .agg(n=("active_weeks_next", "size"), next_games=("active_weeks_next", "mean"))
        .reset_index()
        .pivot(index="prior_active_bucket", columns="ended_on_reserve")
    )
    table.columns = [f"{name}_{'reserve' if flag else 'healthy'}" for name, flag in table.columns]
    table["increment_games"] = table["next_games_reserve"] - table["next_games_healthy"]
    return table


def week_one_status(frame: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """What each Week-1 roster status predicts, and whether it survives the filter."""
    out = rosters.copy()
    out["status"] = out["status"].astype(str).str.upper()
    out["position"] = out["position"].astype(str).str.upper()
    out = out[out["position"].isin(MODEL_POSITIONS)]
    out = out[out["game_type"].astype(str).eq("REG")]
    out = out[pd.to_numeric(out["week"], errors="coerce").eq(1)]
    out = out[out["gsis_id"].notna()].drop_duplicates(["season", "gsis_id"])

    joined = out[["season", "gsis_id", "status"]].merge(
        frame[["season", "gsis_id", "active_weeks"]], on=["season", "gsis_id"], how="left"
    )
    joined["active_weeks"] = joined["active_weeks"].fillna(0.0)
    table = joined.groupby("status").agg(
        n=("active_weeks", "size"),
        mean_active_weeks=("active_weeks", "mean"),
        median_active_weeks=("active_weeks", "median"),
        pct_never_active=("active_weeks", lambda s: 100.0 * float((s == 0).mean())),
    )
    table["kept_by_roster_snapshot"] = [
        status in ROSTER_STATUSES for status in table.index
    ]
    return table.sort_values("n", ascending=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2015, 2025)))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    cache = args.cache_dir / "weekly_rosters.pkl"
    if cache.exists():
        rosters = pd.read_pickle(cache)
    else:
        rosters = ingest.load_weekly_rosters(args.seasons)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        rosters.to_pickle(cache)

    frame = season_rows(rosters)
    paired = pairs(frame)
    paired = paired[paired["active_weeks"].ge(MIN_PRIOR_ACTIVE_WEEKS)]

    ceiling = forecastable(paired)
    increment = injury_increment(paired)
    statuses = week_one_status(frame, rosters)

    print("\n=== 1. Ceiling on a history-only availability forecast ===")
    print(ceiling.round(3).to_string())
    print(
        "\n  Population: players rostered in both seasons with "
        f"{MIN_PRIOR_ACTIVE_WEEKS}+ active weeks in the\n  first, so it is "
        "narrower than the frame the projection covers and the comparison\n  "
        "below is indicative rather than a like-for-like bound. The shipped "
        "2026\n  projection's per-player mean has sd 1.47 games pooled."
    )

    print("\n=== 2. Prior-season injury, net of prior-season availability ===")
    print(increment.round(2).to_string())
    print(
        "\n  A positive increment means finishing season Y on a reserve list "
        "predicts\n  MORE availability in Y+1 than a player who missed the "
        "same number of weeks\n  without one."
    )

    print("\n=== 3. Week-1 roster status vs. active weeks that season ===")
    print(statuses.round(2).to_string())
    dropped = statuses[~statuses["kept_by_roster_snapshot"]]
    dropped = dropped[dropped["n"].ge(10)]
    if not dropped.empty:
        print(
            "\n  Statuses NOT in ROSTER_STATUSES are dropped from the preseason "
            "roster\n  entirely, not carried as an availability outcome:\n  "
            + ", ".join(sorted(dropped.index))
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "forecast_ceiling": ceiling.reset_index().to_dict("records"),
                    "injury_increment": increment.reset_index().astype(str).to_dict("records"),
                    "week_one_status": statuses.reset_index().astype(str).to_dict("records"),
                },
                indent=2,
            ),
            "utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
