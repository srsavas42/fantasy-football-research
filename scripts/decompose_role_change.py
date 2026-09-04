"""Does a player's role move between seasons, or between weeks?

The two answers call for opposite work. If role is stable inside a season and
resets over the offseason, the model's problem is the season boundary: its
history features are career averages that carry last year's role into week 1, and
the fix is a discontinuity at the boundary plus better preseason information. If
role instead drifts week to week inside a season, the boundary is a side issue
and the fix has to be a faster or better in-season signal.

Three measurements, because no single one settles it.

**A variance decomposition of the role itself**, which is unconditional and so
does not inherit the selection problem in the "role grew" segment. A player's
share of his team's carries or targets varies for three reasons: players differ
from each other, the same player differs between his seasons, and his weeks
differ inside a season. The last of those contains sampling noise -- a share
measured over eleven carries is not a precise quantity -- and that part is
estimated and removed rather than counted as drift.

**Which lag predicts better, by week.** At week 1 the only history is last
season. By week 6 there is a current-season average to use instead. Scoring both
against the realised share at each week shows exactly where the season boundary
stops mattering, and whether the model's blended average is beating either.

**Where the error actually sits.** Role-change events are counted by the week
they happen, so the question of whether this is a week-1 problem or an
all-season problem is answered by counting rather than by intuition.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly.features import relevant_population
from ffmodel.weekly.nextweek import Hurdle

MIN_GAMES = 6


def _shipped() -> Hurdle:
    return Hurdle(
        use_team=True, use_matchup=True, use_phase=True,
        use_script=True, use_adp=True, use_news=True, by_position=True,
    )


def share_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Realised share, its denominator, and the two candidate lags."""
    is_back = frame["position"].eq("RB").to_numpy()
    numerator = np.where(is_back, frame["rush_att"], frame["targets"]).astype(float)
    denominator = np.where(
        is_back, frame["team_rush_att"], frame["team_targets"]
    ).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.divide(
            numerator, denominator,
            out=np.full(len(frame), np.nan), where=denominator > 0,
        )
    lagged = np.where(
        is_back,
        pd.to_numeric(frame["prior_rush_share_recent"], errors="coerce"),
        pd.to_numeric(frame["prior_target_share_recent"], errors="coerce"),
    )
    return pd.DataFrame(
        {
            "player_key": frame["player_key"].to_numpy(),
            "season": frame["season"].to_numpy(),
            "week": frame["week"].to_numpy(),
            "position": frame["position"].to_numpy(),
            "played": frame["played"].to_numpy(),
            "share": share,
            "opportunities": denominator,
            "model_lag": lagged,
        }
    )


def variance_components(shares: pd.DataFrame) -> dict:
    """Split the variance of role into player, season-within-player, and week.

    The within-season term is corrected for sampling noise: a share observed over
    ``n`` team opportunities carries binomial variance ``p(1-p)/n`` that is not
    drift. Without removing it the week-to-week term is inflated by exactly the
    amount that makes a stable role look unstable.
    """
    played = shares[shares["played"].eq(1) & shares["share"].notna()]
    seasons = (
        played.groupby(["player_key", "season"])
        .agg(
            games=("share", "size"),
            season_mean=("share", "mean"),
            within_var=("share", lambda s: s.var(ddof=1)),
            mean_opp=("opportunities", "mean"),
        )
        .reset_index()
    )
    seasons = seasons[seasons["games"] >= MIN_GAMES]
    # Only players with at least two qualifying seasons can speak to the
    # between-season term at all.
    counts = seasons.groupby("player_key")["season"].transform("size")
    multi = seasons[counts >= 2]

    player_means = multi.groupby("player_key")["season_mean"].mean()
    between_player = float(player_means.var(ddof=1))
    between_season = float(
        multi.assign(dev=multi["season_mean"] - multi["player_key"].map(player_means))[
            "dev"
        ].pow(2).mean()
    )
    raw_within = float(seasons["within_var"].mean())
    sampling = float(
        np.mean(
            seasons["season_mean"] * (1 - seasons["season_mean"]) / seasons["mean_opp"]
        )
    )
    within_season = max(raw_within - sampling, 0.0)

    total = between_player + between_season + within_season
    return {
        "n_player_seasons": int(len(seasons)),
        "n_players_multi_season": int(multi["player_key"].nunique()),
        "between_player": between_player,
        "between_season_within_player": between_season,
        "within_season_raw": raw_within,
        "within_season_sampling_noise": sampling,
        "within_season_drift": within_season,
        "pct_between_player": 100.0 * between_player / total if total else np.nan,
        "pct_between_season": 100.0 * between_season / total if total else np.nan,
        "pct_within_season": 100.0 * within_season / total if total else np.nan,
    }


def lag_comparison(shares: pd.DataFrame) -> pd.DataFrame:
    """Last season's share against this season's, as predictors, by week."""
    played = shares[shares["played"].eq(1) & shares["share"].notna()].copy()
    season_mean = (
        played.groupby(["player_key", "season"])["share"].mean().rename("season_mean")
    )
    previous = season_mean.reset_index()
    previous["season"] = previous["season"] + 1
    previous = previous.rename(columns={"season_mean": "last_season_share"})

    played = played.sort_values(["player_key", "season", "week"])
    grouped = played.groupby(["player_key", "season"], sort=False)["share"]
    # Expanding mean of this season only, lagged so the current week is excluded.
    played["this_season_share"] = (
        grouped.expanding().mean().droplevel([0, 1]).reindex(played.index)
    )
    played["this_season_share"] = played.groupby(
        ["player_key", "season"], sort=False
    )["this_season_share"].shift(1)
    played = played.merge(previous, on=["player_key", "season"], how="left")

    rows = []
    for week, block in played.groupby("week"):
        if week > 14:
            continue
        entry = {"week": int(week), "n": int(len(block))}
        for name, column in (
            ("last_season", "last_season_share"),
            ("this_season", "this_season_share"),
            ("model_lag", "model_lag"),
        ):
            ok = block[column].notna()
            entry[f"{name}_n"] = int(ok.sum())
            entry[f"{name}_mae"] = (
                float((block.loc[ok, column] - block.loc[ok, "share"]).abs().mean())
                if ok.any()
                else np.nan
            )
        rows.append(entry)
    return pd.DataFrame(rows)


def events_by_week(shares: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """When do role step-ups happen, and how big are they?"""
    played = shares[shares["played"].eq(1) & shares["share"].notna()].copy()
    played["delta"] = played["share"] - played["model_lag"]
    played["event"] = (played["delta"] >= threshold) & (played["share"] >= 0.20)
    first = (
        played[played["event"]]
        .sort_values(["player_key", "season", "week"])
        .groupby(["player_key", "season"], as_index=False)
        .first()
    )
    return (
        first.groupby("week")
        .agg(events=("delta", "size"), mean_step=("delta", "mean"))
        .reset_index()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(".cache/weekly_features_news_2016_2025.pkl"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    frame = pd.read_pickle(args.features)
    keep = relevant_population(frame).to_numpy(bool) | pd.to_numeric(
        frame["adp_drafted"], errors="coerce"
    ).eq(1).to_numpy()
    shares = share_columns(frame[keep].reset_index(drop=True))

    components = variance_components(shares)
    print("=== where does role variance live? ===")
    print(
        f"  player-seasons {components['n_player_seasons']}, "
        f"multi-season players {components['n_players_multi_season']}"
    )
    for label, key in (
        ("between players", "pct_between_player"),
        ("between seasons, same player", "pct_between_season"),
        ("week to week, within a season", "pct_within_season"),
    ):
        print(f"  {label:32s} {components[key]:5.1f}%")
    print(
        f"  (raw week-to-week variance {components['within_season_raw']:.5f}, of "
        f"which {components['within_season_sampling_noise']:.5f} is sampling noise)"
    )

    print("\n=== which lag predicts this week's share better? (MAE) ===")
    table = lag_comparison(shares)
    print(
        table[
            ["week", "last_season_n", "last_season_mae", "this_season_mae", "model_lag_mae"]
        ].round(4).to_string(index=False)
    )

    print("\n=== when do role step-ups first happen? ===")
    events = events_by_week(shares, args.threshold)
    print(events.round(4).to_string(index=False))
    total = int(events["events"].sum())
    early = int(events[events["week"] <= 2]["events"].sum())
    print(f"  {early} of {total} first step-ups ({100.0 * early / max(total,1):.1f}%) land in weeks 1-2")

    print(
        "\nReading it: if role were a year-over-year phenomenon, the "
        "between-season term\nwould dominate the within-season one, last season's "
        "share would beat this\nseason's for most of the year, and step-ups would "
        "pile up in week 1."
    )

    payload = {
        "variance_components": components,
        "lag_comparison": table.to_dict("records"),
        "events_by_week": events.to_dict("records"),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
