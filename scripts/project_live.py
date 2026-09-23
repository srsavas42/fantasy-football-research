"""Project a week that has not been played yet.

``scripts/project_week.py`` reproduces what the model would have said about a
week already in the books, which is what validation needs and not what a manager
needs. The panel is built from stat lines, so a week nobody has played has no
rows at all, and every forward question -- who do I start on Sunday, who is
worth a waiver claim -- is about exactly that week.

    python scripts/project_live.py --season 2026 --week 2

So the rows for the target week are *constructed* rather than read: the clubs
playing come from the schedule, the players from this week's roster filings, the
opponent and the closing line from the schedule again. Nothing about the week's
outcome is invented -- the stat columns are left at zero and never read, because
every feature the models consume is lagged by :func:`_prior` and therefore
describes weeks already played.

Two things have to be got right or the projection is quietly wrong:

**The schedule, not the panel, counts the games left.** ``team_games_remaining``
derives its count from the team-weeks present in the panel, which in a live
season is "weeks played so far" -- it would tell a week-2 projection that one
game remains and turn a rest-of-season total into a one-week one. Here the count
comes from the published schedule, and a club's bye is a week it does not
appear, so it is never counted.

**The pre-game feeds are the point, not a detail.** The injury report and the
depth chart for the coming week are published before it and are the only inputs
that describe the week being projected rather than the ones before it. They are
what separates "he is questionable and his backup just got promoted" from a
projection built entirely out of history.

Both horizons are written, because they answer different questions:
``week_<n>.csv`` is the start/sit call and ``rest_of_season.csv`` is the waiver
call -- a player worth rostering for fourteen weeks and a player worth starting
on Sunday are not the same player.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.data import ingest
from ffmodel.models.market_blend import blend_samples
from ffmodel.weekly.availability_rate import (
    PlayedRate,
    add_played_rate_target,
)
from ffmodel.weekly.charting import attach_charting
from ffmodel.weekly.expected import attach_expected
from ffmodel.weekly.features import add_features
from ffmodel.weekly.frame import (
    PANEL_POSITIONS,
    STAT_COLUMNS,
    _market_lines,
    _roster_weeks,
    build_panel,
)
from ffmodel.weekly.market import (
    WeeklyRankCurve,
    attach_adp,
    bucket_labels,
    fit_blend_weights,
)
from ffmodel.weekly.news import add_news_features
from ffmodel.weekly.nextweek import Hurdle
from ffmodel.weekly.pedigree import add_pedigree_features
from ffmodel.weekly.restofseason import (
    OFFSET,
    TARGET,
    DirectTotal,
    add_rest_of_season_target,
)
from ffmodel.weekly.tendency import attach_tendency

FIRST_SEASON = 2016


def schedule_for(season: int) -> pd.DataFrame:
    """Regular-season games, one row per club-week, with the closing line."""
    raw = ingest.load_schedules([season])
    if "game_type" in raw.columns:
        raw = raw[raw["game_type"] == "REG"]
    frames = []
    for side, other in (("home_team", "away_team"), ("away_team", "home_team")):
        block = raw[["season", "week", side, other]].copy()
        block.columns = ["season", "week", "team", "opponent"]
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("int64")
    return out.dropna(subset=["team"]).drop_duplicates(subset=["season", "week", "team"])


def games_remaining_from_schedule(schedule: pd.DataFrame, week: int) -> pd.Series:
    """Games each club has left, counting the one about to be played.

    A bye is a week the club does not appear in the schedule, so counting rows
    rather than subtracting week numbers handles it without a special case.
    """
    ahead = schedule[schedule["week"] >= week]
    return ahead.groupby("team")["week"].size()


def build_live_rows(season: int, week: int, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per rostered skill player on a club playing in ``week``.

    Shaped exactly like a panel row so the feature layer cannot tell the
    difference. Every stat column is zero and none of them is ever read: the
    models consume lagged features only, and this row's own week has not
    happened.
    """
    schedule = schedule_for(season)
    playing = schedule[schedule["week"] == week]
    if playing.empty:
        raise SystemExit(f"no scheduled {season} week {week} games")

    roster = _roster_weeks([season])
    roster = roster[roster["week"] == week]
    if roster.empty:
        # Filings for the coming week are not always up yet; the most recent
        # week's roster is the best available statement of who is employed.
        latest = int(_roster_weeks([season])["week"].max())
        roster = _roster_weeks([season])
        roster = roster[roster["week"] == latest].assign(week=week)
        print(f"  week {week} roster filings absent; using week {latest}")

    rows = roster.merge(playing[["team", "opponent"]], on="team", how="inner")
    rows = rows[rows["position"].isin(PANEL_POSITIONS)]

    lines = _market_lines([season])
    lines = lines[lines["week"].astype("Int64") == week]
    rows = rows.merge(
        lines[["team", "spread", "game_total", "implied_team_total",
               "implied_opponent_total"]],
        on="team", how="left",
    )

    rows["season"] = season
    rows["week"] = week
    rows["player_key"] = rows["player_id"].astype(str)
    rows["played"] = 0
    rows["points"] = 0.0
    for column in STAT_COLUMNS:
        rows[column] = 0.0
    # This week's team totals and snaps are unknown and unused -- the team
    # history features are lagged, so week 2 reads week 1's.
    for column in panel.columns:
        if column not in rows.columns:
            rows[column] = np.nan
    rows["scoring"] = panel["scoring"].iloc[0] if "scoring" in panel.columns else "ppr"
    return rows[panel.columns]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--outdir", type=Path, default=Path("projections"))
    args = parser.parse_args(argv)

    print(f"building the panel through {args.season} week {args.week - 1} ...")
    panel = build_panel(range(FIRST_SEASON, args.season + 1))
    played = panel[panel["season"] == args.season]["week"]
    print(f"  {args.season} weeks in the panel: {sorted(played.unique().tolist())}")

    live = build_live_rows(args.season, args.week, panel)
    print(f"  built {len(live)} rows for {args.season} week {args.week}")
    panel = pd.concat([panel, live], ignore_index=True)
    panel = panel.sort_values(["player_key", "season", "week"], kind="mergesort")
    panel = panel.reset_index(drop=True)

    print("attaching features ...")
    frame = add_pedigree_features(
        add_news_features(
            add_features(
                attach_charting(attach_expected(attach_tendency(attach_adp(panel))))
            )
        )
    )
    frame = add_rest_of_season_target(frame)

    # The schedule, not the panel, says how many games are left. Overridden
    # after `add_rest_of_season_target` because that helper counts panel
    # team-weeks, which in a live season stop at the last week played.
    schedule = schedule_for(args.season)
    left = games_remaining_from_schedule(schedule, args.week)
    target = (frame["season"].eq(args.season) & frame["week"].eq(args.week)).to_numpy()
    frame.loc[target, OFFSET] = (
        frame.loc[target, "team"].map(left).astype(float).to_numpy()
    )

    train = frame[frame["season"] < args.season]
    rows = frame[target]
    if rows.empty:
        raise SystemExit(f"no rows for {args.season} week {args.week}")
    print(
        f"  fitted on {int(train.season.min())}-{int(train.season.max())}; "
        f"{len(rows)} players, {rows[OFFSET].min():.0f}-{rows[OFFSET].max():.0f} games left"
    )

    weekly_target = train["points"].to_numpy(float)
    seed = args.season * 100 + args.week
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- week
    hurdle = Hurdle(
        use_team=True, use_matchup=True, use_phase=True, use_script=True,
        use_adp=True, use_news=True, use_snaps=True, use_recent=True,
        use_pedigree=True, use_charting=True, by_position=True,
    ).fit(train, weekly_target)
    samples = hurdle.predict_samples(rows, draws=args.draws, seed=seed)
    week_out = pd.DataFrame({
        "player": rows["player_name"].to_numpy(),
        "position": rows["position"].to_numpy(),
        "team": rows["team"].to_numpy(),
        "opponent": rows["opponent"].to_numpy(),
        "projected_points": samples.mean(axis=1),
        "p10": np.quantile(samples, 0.10, axis=1),
        "p50": np.quantile(samples, 0.50, axis=1),
        "p90": np.quantile(samples, 0.90, axis=1),
        "p_plays": hurdle.play_probability(rows),
        "inj_status": rows["inj_status"].to_numpy(),
    }).sort_values("projected_points", ascending=False).reset_index(drop=True)
    # Rank within position, because that is the shape of the decision: a flex
    # spot is a choice among running backs and receivers, never among everybody.
    week_out.insert(0, "overall_rank", week_out.index + 1)
    week_out.insert(1, "pos_rank", week_out.groupby("position").cumcount() + 1)
    week_path = args.outdir / f"{args.season}_week{args.week}_projections.csv"
    week_out.round(3).to_csv(week_path, index=False)

    # ------------------------------------------------------- rest of season
    def build():
        return DirectTotal(use_team=True, use_phase=True, use_adp=True, use_role=True)

    direct = build().fit(train, train[TARGET].to_numpy(float))
    ros = direct.predict_samples(rows, draws=args.draws, seed=seed)
    # Early in a season the board still knows things the model does not, so the
    # drafted rows are blended toward it at the fitted per-horizon weight. The
    # weight goes to 1.0 once the model has usage the board never saw.
    weights = fit_blend_weights(train, build, TARGET, seed=seed)
    curve = WeeklyRankCurve(per_game=False, offset=OFFSET).fit(train, weekly_target)
    curve_samples = curve.predict_samples(rows, draws=args.draws, seed=seed)
    labels = bucket_labels(rows["week"].to_numpy(float))
    drafted = pd.to_numeric(rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
    for name, weight in weights.items():
        want = (labels == name) & drafted
        if want.any():
            ros[want] = blend_samples(ros[want], curve_samples[want], weight, seed=seed + 1)
    print(f"  blend weight on the model, by horizon: {weights}")

    # Per game, two ways, because they answer different questions and the naive
    # one is a trap. The rest-of-season total already prices availability, so
    # dividing it by the *scheduled* games left discounts the absence twice and
    # describes a player who suits up every week -- which is not the player the
    # total was about. `points_per_active_game` divides by the games he is
    # actually expected to play, and is the rate to compare two players on.
    played_rate = PlayedRate().fit(add_played_rate_target(train))
    rate = played_rate.predict(rows)
    games = rows[OFFSET].to_numpy(float)
    expected_games = games * rate
    total = ros.mean(axis=1)
    ros_out = pd.DataFrame({
        "player": rows["player_name"].to_numpy(),
        "position": rows["position"].to_numpy(),
        "team": rows["team"].to_numpy(),
        "games_left": games,
        "expected_games_played": expected_games,
        "rest_of_season_points": total,
        "points_per_active_game": total / np.where(expected_games > 0, expected_games, np.nan),
        "points_per_scheduled_game": total / np.where(games > 0, games, np.nan),
        "p10": np.quantile(ros, 0.10, axis=1),
        "p50": np.quantile(ros, 0.50, axis=1),
        "p90": np.quantile(ros, 0.90, axis=1),
        "adp_rank": rows["adp_rank"].to_numpy(),
    }).sort_values("rest_of_season_points", ascending=False).reset_index(drop=True)
    ros_out.insert(0, "overall_rank", ros_out.index + 1)
    ros_out.insert(1, "pos_rank", ros_out.groupby("position").cumcount() + 1)
    ros_path = args.outdir / f"{args.season}_week{args.week}_rest_of_season.csv"
    ros_out.round(3).to_csv(ros_path, index=False)

    print(f"\n=== {args.season} week {args.week}: start/sit ===")
    print(week_out.head(args.top).round(2).to_string(index=False))
    print(f"\n=== {args.season} rest of season, from week {args.week} ===")
    print(ros_out.head(args.top).round(2).to_string(index=False))
    print(f"\nwrote {week_path}\nwrote {ros_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
