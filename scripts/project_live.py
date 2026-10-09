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
``start_sit.csv`` is the start/sit call and ``rest_of_season.csv`` is the waiver
call -- a player worth rostering for fourteen weeks and a player worth starting
on Sunday are not the same player. They land in
``projections/weekly/<season>/week<NN>/``; see ``projections/README.md``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.data import ingest
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
from ffmodel.weekly.injury_type import attach_injury_type
from ffmodel.weekly.market import attach_adp
from ffmodel.weekly.news import add_news_features
from ffmodel.weekly.nextweek import Hurdle
from ffmodel.weekly.pedigree import add_pedigree_features
from ffmodel.weekly.restofseason import OFFSET, TARGET, add_rest_of_season_target
from ffmodel.weekly.ros_reconciled import ReconciledROS, add_absence_state, summarise
from ffmodel.weekly.tendency import attach_tendency

FIRST_SEASON = 2016

# A player placed on injured reserve must miss at least this many games. It is
# what turns "he is on the reserve list" from a flag into a bound: a player put
# there this week cannot be back for the game being projected, however healthy
# the rest of the record looks.
IR_MINIMUM_GAMES = 4

MANUAL_STATUS = Path("projections/overrides/manual_status_2026.csv")


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


def reserve_out_through(season: int, week: int, schedule: pd.DataFrame) -> pd.Series:
    """Last week a reserve-list player is guaranteed to miss, by ``player_id``.

    The roster feed says who is on reserve *this* week. The placement week is the
    start of the unbroken run of reserve snapshots ending now, and the minimum
    is counted in **club games**, not calendar weeks: a bye inside the window
    pushes the earliest return out a week, and treating weeks as games would let
    a player back a game early.

    This is a floor, not a forecast. It says nothing about when he actually
    returns, only when he cannot, which is the part the feed knows for certain.
    """
    roster = _roster_weeks([season])
    roster = roster[roster["position"].isin(PANEL_POSITIONS)]
    status = roster.set_index(["player_id", "week"])["status"]
    now = roster[(roster["week"] == week) & (roster["status"] == "RES")]
    games = schedule.groupby("team")["week"].apply(lambda w: sorted(int(x) for x in w))

    out: dict[str, int] = {}
    for player_id, team in zip(now["player_id"], now["team"]):
        placed = week
        for earlier in range(week - 1, 0, -1):
            if status.get((player_id, earlier)) == "RES":
                placed = earlier
            else:
                break
        club = [w for w in games.get(team, []) if w >= placed]
        # The week of his IR_MINIMUM_GAMES-th missed game. A club with fewer games
        # left than the minimum is simply out for whatever remains.
        last = club[min(IR_MINIMUM_GAMES, len(club)) - 1] if club else week
        out[str(player_id)] = max(int(last), week)
    return pd.Series(out, dtype=float)


def manual_out_through(rows: pd.DataFrame) -> pd.Series:
    """News the feed has not caught up to, keyed to the rows by index."""
    if not MANUAL_STATUS.exists():
        return pd.Series(0.0, index=rows.index)
    manual = pd.read_csv(MANUAL_STATUS)
    key = manual.set_index(["player_name", "team"])["out_through_week"]
    got = [key.get((n, t), 0) for n, t in zip(rows["player_name"], rows["team"])]
    return pd.Series(np.asarray(got, float), index=rows.index)


def build_live_rows(season: int, week: int, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per rostered skill player, including those on a club on its bye.

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

    # A club on its bye is kept, with no opponent and no line: it plays no game this
    # week, so it has no start/sit row, but its players still have a rest of season and
    # dropping them would hide exactly the players a waiver or trade call is about.
    # The filings for a week leave out a club on its bye, so such a club is carried
    # forward from the last week it filed.
    everyone = _roster_weeks([season])
    absent = sorted(set(schedule["team"]) - set(roster["team"]))
    for club in absent:
        filed = everyone[(everyone["team"] == club) & (everyone["week"] < week)]
        if not filed.empty:
            roster = pd.concat(
                [roster, filed[filed["week"] == filed["week"].max()].assign(week=week)],
                ignore_index=True,
            )
    rows = roster.merge(playing[["team", "opponent"]], on="team", how="left")
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
    parser.add_argument(
        "--outdir", type=Path, default=Path("projections/weekly"),
        help="root for weekly output; files land in <outdir>/<season>/week<NN>/",
    )
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
    # Why and for how long each player has been out, and what was on his report.
    # Needs the whole frame (it reads the previous row), so it is done before the
    # live rows are sliced off.
    frame = add_absence_state(attach_injury_type(frame))

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

    # Clubs on a bye have no start/sit row; the rest-of-season file keeps them.
    playing_teams = set(schedule[schedule["week"] == args.week]["team"])
    all_rows = rows
    rows = all_rows[all_rows["team"].isin(playing_teams)]
    weekly_target = train["points"].to_numpy(float)
    seed = args.season * 100 + args.week
    outdir = args.outdir / str(args.season) / f"week{args.week:02d}"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- week
    hurdle = Hurdle(
        use_team=True, use_matchup=True, use_phase=True, use_script=True,
        use_adp=True, use_news=True, use_snaps=True, use_recent=True,
        use_pedigree=True, use_charting=True, use_partial=True, by_position=True,
    ).fit(train, weekly_target)
    samples = hurdle.predict_samples(rows, draws=args.draws, seed=seed)
    # Who cannot play, and for how long. Reserve status from the roster feed sets
    # a floor; the manual file can only extend it.
    schedule = schedule_for(args.season)
    floor = reserve_out_through(args.season, args.week, schedule)
    out_through = rows["player_key"].map(floor).fillna(0.0)
    out_through = np.maximum(out_through, manual_out_through(rows)).to_numpy(float)
    unavailable = out_through >= args.week
    samples[unavailable] = 0.0
    print(f"  {int(unavailable.sum())} players out for week {args.week} "
          f"(reserve list or manual override)")
    week_out = pd.DataFrame({
        "player": rows["player_name"].to_numpy(),
        "position": rows["position"].to_numpy(),
        "team": rows["team"].to_numpy(),
        "opponent": rows["opponent"].to_numpy(),
        "projected_points": samples.mean(axis=1),
        "p10": np.quantile(samples, 0.10, axis=1),
        "p50": np.quantile(samples, 0.50, axis=1),
        "p90": np.quantile(samples, 0.90, axis=1),
        "p_plays": np.where(unavailable, 0.0, hurdle.play_probability(rows)),
        "inj_status": rows["inj_status"].to_numpy(),
        "roster_status": rows["status"].to_numpy(),
        "out_through_week": np.where(unavailable, out_through, np.nan),
    }).sort_values("projected_points", ascending=False).reset_index(drop=True)
    # Rank within position, because that is the shape of the decision: a flex
    # spot is a choice among running backs and receivers, never among everybody.
    week_out.insert(0, "overall_rank", week_out.index + 1)
    week_out.insert(1, "pos_rank", week_out.groupby("position").cumcount() + 1)
    week_path = outdir / "start_sit.csv"
    week_out.round(3).to_csv(week_path, index=False)

    # ------------------------------------------------------- rest of season
    # One calculation, not three: total = expected games x points per game played,
    # so every column of the row is a breakdown of the same number and none can
    # disagree with another. See ffmodel.weekly.ros_reconciled.
    rows = all_rows
    floor = reserve_out_through(args.season, args.week, schedule)
    out_through = rows["player_key"].map(floor).fillna(0.0)
    out_through = np.maximum(out_through, manual_out_through(rows)).to_numpy(float)
    fit_rows = train[np.isfinite(pd.to_numeric(train[TARGET], errors="coerce"))]
    model = ReconciledROS().fit(fit_rows).fit_intervals(fit_rows)
    parts = model.predict(rows)
    grid = model.quantile_grid(rows, parts["total"].to_numpy())

    # Games he is certain to miss. The model has no way to know a reserve stint is
    # a fixed length, so the certain absence is taken off afterwards: expected
    # games and the quantiles are scaled by the share of remaining games left once
    # it is removed. The rate -- his rate when he plays -- is untouched, so the
    # identity holds on the scaled row.
    club_games = schedule.groupby("team")["week"].apply(lambda w: np.asarray(w, int))
    certain_out = np.array([
        int(((club_games.get(t, np.array([])) >= args.week)
             & (club_games.get(t, np.array([])) <= o)).sum()) if o >= args.week else 0
        for t, o in zip(rows["team"], out_through)
    ], float)
    left = rows[OFFSET].to_numpy(float)
    keep = np.clip((left - certain_out) / np.where(left > 0, left, np.nan), 0.0, 1.0)
    keep = np.nan_to_num(keep, nan=0.0)
    games = left
    expected_games = parts["expected_games"].to_numpy() * keep
    active_rate = np.where(keep > 0, parts["rate"].to_numpy(), np.nan)
    total = np.where(keep > 0, expected_games * active_rate, 0.0)
    quant = summarise(grid * keep[:, None])
    ros_out = pd.DataFrame({
        "player": rows["player_name"].to_numpy(),
        "position": rows["position"].to_numpy(),
        "team": rows["team"].to_numpy(),
        "games_left": games,
        "expected_games_played": expected_games,
        "rest_of_season_points": total,
        "points_per_active_game": active_rate,
        "points_per_scheduled_game": total / np.where(games > 0, games, np.nan),
        "p10": quant["p10"],
        "p50": quant["p50"],
        "p90": quant["p90"],
        "adp_rank": rows["adp_rank"].to_numpy(),
        "roster_status": rows["status"].to_numpy(),
        "out_through_week": np.where(out_through >= args.week, out_through, np.nan),
        "on_bye_this_week": ~rows["team"].isin(playing_teams).to_numpy(),
    }).sort_values("rest_of_season_points", ascending=False).reset_index(drop=True)
    ros_out.insert(0, "overall_rank", ros_out.index + 1)
    ros_out.insert(1, "pos_rank", ros_out.groupby("position").cumcount() + 1)
    ros_path = outdir / "rest_of_season.csv"
    ros_out.round(3).to_csv(ros_path, index=False)

    print(f"\n=== {args.season} week {args.week}: start/sit ===")
    print(week_out.head(args.top).round(2).to_string(index=False))
    print(f"\n=== {args.season} rest of season, from week {args.week} ===")
    print(ros_out.head(args.top).round(2).to_string(index=False))
    print(f"\nwrote {week_path}\nwrote {ros_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
