"""Is a player less efficient the year after a lost season, and does he recover?

The folk claim has two halves and they need separating, because one of them is
mostly an artifact of who is left to measure.

**The penalty.** A player coming back from a serious injury is worse per touch
in his first season back. If true and large, it is a gap in the efficiency
layer, whose entire feature contract is one season deep -- `lagged_efficiency_rows`
shifts `Y` onto `Y+1` and there is no `prior2_*` anywhere in the package, so a
two-year recovery curve is not a feature the model is missing, it is a shape the
feature contract cannot express.

**The rebound.** He recovers further in year two. This is where the measurement
has to be careful. Scoring year two on whoever is still playing in year two,
while scoring year one on a larger group, guarantees a rebound whether or not
one exists: the players who never came back are absent from the year-two column
and present in neither. Every row in the balanced panel here has a healthy
baseline, a qualifying `Y+1` and a qualifying `Y+2`, so both columns describe
the same players. The survivorship table reports separately on who does not
make it, which turns out to be the larger effect by some distance.

Design. Season `Y-1` is a healthy baseline: availability at or above
`BASE_AVAILABILITY` with `MIN_OPPORTUNITIES` opportunities. Season `Y` is
classified lost (availability at or below `LOST_AVAILABILITY`) or a healthy
control (at or above the baseline threshold); the band between them is dropped
rather than assigned. The outcome is the change in points per opportunity from
the baseline, and the reported figure is a difference in differences against
the control, which removes the ageing and mean-reversion drift both groups
share.

The efficiency metric is PPR points per opportunity over targets plus carries,
which is the per-touch quantity the question is about and which pools the
receiving and rushing responses that would otherwise be measured separately at
half the sample. Quarterbacks are excluded: their per-touch denominator is a
different quantity.

Availability is a proxy for injury and an imperfect one -- it cannot tell a torn
ACL from a benching, a holdout or a lost job. The body-group cut, which uses the
nflverse injury reports from 2009, reports how often the proxy has no severe
injury behind it at all.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.data import ingest, load_player_weeks
from ffmodel.features import crossseason
from ffmodel.features.season_efficiency import player_season_efficiency
from ffmodel.features.season_injury import normalise_injury_reports
from ffmodel.features.volume import MODEL_POSITIONS, normalize_model_positions

POSITIONS = ("RB", "WR", "TE")
MIN_OPPORTUNITIES = 50
BASE_AVAILABILITY = 0.85
LOST_AVAILABILITY = 0.65
# Doubtful or worse. Questionable appears on too many healthy players to
# identify the injury that cost the season.
MIN_REPORT_SEVERITY = 2
BOOTSTRAP = 4000

PANEL_COLUMNS = [
    "season",
    "player_key",
    "position",
    "availability",
    "opportunities",
    "points_per_opportunity",
    "career_year",
]


def player_seasons(efficiency: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season with availability and per-touch efficiency."""
    out = efficiency.merge(
        games[["season", "player_key", "availability"]],
        on=["season", "player_key"],
        how="inner",
    )
    out["opportunities"] = out[["targets", "rush_att"]].fillna(0).sum(axis=1)
    points = (
        0.1 * out["rec_yds"].fillna(0)
        + 6 * out["rec_td"].fillna(0)
        + out["receptions"].fillna(0)
        + 0.1 * out["rush_yds"].fillna(0)
        + 6 * out["rush_td"].fillna(0)
        - 2 * out["fumbles_lost"].fillna(0)
    )
    out["points_per_opportunity"] = points / out["opportunities"].replace(0, np.nan)
    out["career_year"] = out.groupby("player_key")["season"].rank(method="dense")
    return out[PANEL_COLUMNS]


def panel(seasons: pd.DataFrame, *, require_second_year: bool) -> pd.DataFrame:
    """Join each anchor season to its baseline and its one or two follow-ups."""

    def shifted(offset: int, suffix: str) -> pd.DataFrame:
        out = seasons.copy()
        out["season"] = out["season"] - offset
        return out.rename(
            columns={
                name: name + suffix
                for name in PANEL_COLUMNS
                if name not in ("season", "player_key")
            }
        )

    anchor = seasons.rename(
        columns={
            name: name + "_y"
            for name in PANEL_COLUMNS
            if name not in ("season", "player_key")
        }
    )
    out = anchor.merge(shifted(-1, "_b"), on=["season", "player_key"]).merge(
        shifted(1, "_p1"), on=["season", "player_key"]
    )
    if require_second_year:
        out = out.merge(shifted(2, "_p2"), on=["season", "player_key"])

    out = out[out["position_y"].isin(POSITIONS)]
    out = out[
        out["availability_b"].ge(BASE_AVAILABILITY)
        & out["opportunities_b"].ge(MIN_OPPORTUNITIES)
        & out["opportunities_p1"].ge(MIN_OPPORTUNITIES)
    ]
    out = out.dropna(subset=["points_per_opportunity_b", "points_per_opportunity_p1"])
    out["delta_year1"] = (
        out["points_per_opportunity_p1"] - out["points_per_opportunity_b"]
    )
    if require_second_year:
        out = out[out["opportunities_p2"].ge(MIN_OPPORTUNITIES)]
        out = out.dropna(subset=["points_per_opportunity_p2"])
        out["delta_year2"] = (
            out["points_per_opportunity_p2"] - out["points_per_opportunity_b"]
        )

    out["group"] = np.where(
        out["availability_y"].le(LOST_AVAILABILITY),
        "lost season",
        np.where(out["availability_y"].ge(BASE_AVAILABILITY), "healthy control", "partial"),
    )
    return out[out["group"].isin(("lost season", "healthy control"))].reset_index(drop=True)


def difference(frame: pd.DataFrame, column: str, rng) -> tuple[float, float, float]:
    """Lost-minus-control mean change, with a percentile bootstrap interval."""
    lost = frame.loc[frame["group"].eq("lost season"), column].dropna().to_numpy()
    control = frame.loc[frame["group"].eq("healthy control"), column].dropna().to_numpy()
    if len(lost) < 5 or len(control) < 5:
        return float("nan"), float("nan"), float("nan")
    draws = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        draws[i] = (
            rng.choice(lost, len(lost), replace=True).mean()
            - rng.choice(control, len(control), replace=True).mean()
        )
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(lost.mean() - control.mean()), float(low), float(high)


def severe_injury_body_group(seasons: range | list[int]) -> pd.DataFrame:
    """The body group a player's most-reported severe episode belonged to."""
    reports = normalise_injury_reports(ingest.load_injuries(seasons))
    reports = reports[reports["injury_severity"].ge(MIN_REPORT_SEVERITY)]
    if reports.empty:
        return pd.DataFrame(columns=["season", "player_key", "injury_body_group"])
    counted = (
        reports.groupby(["season", "player_key", "injury_body_group"])
        .size()
        .rename("weeks")
        .reset_index()
        .sort_values("weeks")
        .drop_duplicates(["season", "player_key"], keep="last")
    )
    return counted[["season", "player_key", "injury_body_group"]]


def survivorship(seasons: pd.DataFrame, balanced: pd.DataFrame) -> pd.DataFrame:
    """Two-stage attrition, measured from the anchor season.

    The balanced panel can only describe players who came back and held a role
    for two seasons. This is who did not, and it is where most of the injury
    effect turns out to live.
    """
    def shifted(offset: int, suffix: str) -> pd.DataFrame:
        out = seasons.copy()
        out["season"] = out["season"] - offset
        return out.rename(
            columns={
                name: name + suffix
                for name in PANEL_COLUMNS
                if name not in ("season", "player_key")
            }
        )

    anchor = seasons.rename(
        columns={
            name: name + "_y"
            for name in PANEL_COLUMNS
            if name not in ("season", "player_key")
        }
    ).merge(shifted(-1, "_b"), on=["season", "player_key"])
    anchor = anchor[anchor["position_y"].isin(POSITIONS)]
    anchor = anchor[
        anchor["availability_b"].ge(BASE_AVAILABILITY)
        & anchor["opportunities_b"].ge(MIN_OPPORTUNITIES)
    ]
    anchor = anchor[anchor["season"].le(int(seasons["season"].max()) - 2)]
    anchor["group"] = np.where(
        anchor["availability_y"].le(LOST_AVAILABILITY),
        "lost season",
        np.where(
            anchor["availability_y"].ge(BASE_AVAILABILITY), "healthy control", "partial"
        ),
    )
    anchor = anchor[anchor["group"].isin(("lost season", "healthy control"))]

    qualifying = seasons[seasons["opportunities"].ge(MIN_OPPORTUNITIES)]
    year1 = {
        (int(row[0]) - 1, row[1])
        for row in qualifying[["season", "player_key"]].to_numpy()
    }
    arrived = set(map(tuple, balanced[["season", "player_key"]].to_numpy()))
    keys = [tuple(row) for row in anchor[["season", "player_key"]].to_numpy()]
    anchor["held_a_role_in_year1"] = [(int(k[0]), k[1]) in year1 for k in keys]
    anchor["and_again_in_year2"] = [(k[0], k[1]) in arrived for k in keys]
    return anchor.groupby("group").agg(
        n=("held_a_role_in_year1", "size"),
        pct_back_in_year1=("held_a_role_in_year1", lambda s: 100.0 * float(s.mean())),
        pct_still_there_in_year2=("and_again_in_year2", lambda s: 100.0 * float(s.mean())),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(1999, 2026)))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(7)

    efficiency_path = args.cache_dir / "season_efficiency.pkl"
    games_path = args.cache_dir / "player_season_games.pkl"
    if efficiency_path.exists() and games_path.exists():
        efficiency = pd.read_pickle(efficiency_path)
        games = pd.read_pickle(games_path)
    else:
        weeks = load_player_weeks(args.seasons)
        efficiency = player_season_efficiency(weeks)
        rows = normalize_model_positions(weeks)
        rows = rows[rows["position"].isin(MODEL_POSITIONS)].copy()
        if "player_id" not in rows:
            rows["player_id"] = pd.NA
        rows["player_key"] = crossseason.player_key(rows)
        games = (
            rows.groupby(["season", "player_key"])["week"]
            .nunique()
            .rename("games")
            .reset_index()
        )
        games = games.join(
            rows.groupby("season")["week"].max().rename("team_weeks"), on="season"
        )
        games["availability"] = games["games"] / games["team_weeks"]
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        efficiency.to_pickle(efficiency_path)
        games.to_pickle(games_path)

    seasons = player_seasons(efficiency, games)
    balanced = panel(seasons, require_second_year=True)
    control_baseline = float(
        balanced.loc[balanced["group"].eq("healthy control"), "points_per_opportunity_b"].mean()
    )

    print("=== Balanced panel: healthy baseline, qualifying Y+1 AND Y+2 ===")
    print(
        f"n = {len(balanced)}, seasons {int(balanced.season.min())}"
        f"-{int(balanced.season.max())}, PPR points per opportunity\n"
    )
    summary = balanced.groupby("group").agg(
        n=("delta_year1", "size"),
        availability_y=("availability_y", "mean"),
        career_year=("career_year_y", "mean"),
        baseline=("points_per_opportunity_b", "mean"),
        year1=("points_per_opportunity_p1", "mean"),
        year2=("points_per_opportunity_p2", "mean"),
    )
    print(summary.round(4).to_string())

    results = {}
    print("\ndifference in differences against the control, with a 95% bootstrap interval:")
    for column, label in (("delta_year1", "year 1 after"), ("delta_year2", "year 2 after")):
        estimate, low, high = difference(balanced, column, rng)
        results[column] = {"estimate": estimate, "low": low, "high": high}
        print(
            f"  {label}: {estimate:+.4f}  [{low:+.4f}, {high:+.4f}]"
            f"   = {100 * estimate / control_baseline:+.2f}% of baseline"
            f"  [{100 * low / control_baseline:+.2f}%, {100 * high / control_baseline:+.2f}%]"
        )

    balanced["rebound"] = balanced["delta_year2"] - balanced["delta_year1"]
    estimate, low, high = difference(balanced, "rebound", rng)
    results["rebound"] = {"estimate": estimate, "low": low, "high": high}
    print(
        f"\nthe rebound itself (year 2 minus year 1, same players): {estimate:+.4f}"
        f"  [{low:+.4f}, {high:+.4f}]"
    )
    print("  positive would mean recovering further in the second season.")

    print("\n=== Survivorship: who is still in the sample to be measured ===")
    print("Measured from the anchor season, before any follow-up filter, so it")
    print("counts the players the panel above cannot see.")
    survival = survivorship(seasons, balanced)
    print(survival.round(1).to_string())
    results["survival"] = survival.reset_index().to_dict("records")

    print("\n=== Year 1 only, 2009+, by primary injury body group ===")
    print("Unbalanced on purpose: dropping the Y+2 requirement roughly doubles the")
    print("lost-season sample, and the body-group cells need every row they can get.")
    year_one = panel(seasons, require_second_year=False)
    injuries = severe_injury_body_group(range(2009, 2025))
    year_one = year_one.merge(injuries, on=["season", "player_key"], how="left")
    control_delta = float(
        year_one.loc[year_one["group"].eq("healthy control"), "delta_year1"].mean()
    )
    control_base = float(
        year_one.loc[year_one["group"].eq("healthy control"), "points_per_opportunity_b"].mean()
    )
    lost = year_one[year_one["group"].eq("lost season") & year_one["season"].ge(2009)].copy()
    lost["injury_body_group"] = lost["injury_body_group"].fillna("no severe report")
    table = lost.groupby("injury_body_group").agg(
        n=("delta_year1", "size"),
        baseline=("points_per_opportunity_b", "mean"),
        delta_year1=("delta_year1", "mean"),
    )
    table["difference"] = table["delta_year1"] - control_delta
    table["pct_of_baseline"] = 100.0 * table["difference"] / control_base
    print(table.round(4).sort_values("n", ascending=False).to_string())
    results["body_groups"] = table.reset_index().to_dict("records")

    control_deltas = (
        year_one.loc[year_one["group"].eq("healthy control"), "delta_year1"].dropna().to_numpy()
    )
    print("\nbootstrap intervals, as a percentage of the control baseline:")
    for group in table.sort_values("n", ascending=False).index:
        values = lost.loc[lost["injury_body_group"].eq(group), "delta_year1"].dropna().to_numpy()
        if len(values) < 5:
            continue
        draws = np.empty(BOOTSTRAP)
        for i in range(BOOTSTRAP):
            draws[i] = (
                rng.choice(values, len(values), replace=True).mean()
                - rng.choice(control_deltas, len(control_deltas), replace=True).mean()
            )
        low, high = np.percentile(draws, [2.5, 97.5])
        estimate = values.mean() - control_deltas.mean()
        print(
            f"  {group:<18s} n={len(values):3d}  {100 * estimate / control_base:+6.2f}%"
            f"  [{100 * low / control_base:+.1f}%, {100 * high / control_base:+.1f}%]"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
