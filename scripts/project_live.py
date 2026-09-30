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
    RATE_TARGET,
    PlayedRate,
    add_played_rate_target,
    add_points_per_active_game_target,
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

# A player placed on injured reserve must miss at least this many games. It is
# what turns "he is on the reserve list" from a flag into a bound: a player put
# there this week cannot be back for the game being projected, however healthy
# the rest of the record looks.
IR_MINIMUM_GAMES = 4

MANUAL_STATUS = Path("projections/manual_status_2026.csv")


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
    # Games he is certain to miss. The model priced ordinary availability and
    # knows nothing about a reserve stint, so its total is scaled by the share of
    # remaining games left once the certain absence is removed. The scaling is
    # applied to the draws, so the quantiles, the total and the expected games
    # all move together and the per-active-game rate -- his rate when he plays --
    # is untouched. It is approximate: the model's own availability estimate
    # already reflects the recent absence, so a returning player is discounted a
    # little twice, which errs conservative.
    club_games = schedule.groupby("team")["week"].apply(lambda w: np.asarray(w, int))
    certain_out = np.array([
        int(((club_games.get(t, np.array([])) >= args.week)
             & (club_games.get(t, np.array([])) <= o)).sum()) if o >= args.week else 0
        for t, o in zip(rows["team"], out_through)
    ], float)
    left = rows[OFFSET].to_numpy(float)
    keep = np.clip((left - certain_out) / np.where(left > 0, left, np.nan), 0.0, 1.0)
    keep = np.nan_to_num(keep, nan=0.0)
    ros = ros * keep[:, None]

    played_rate = PlayedRate().fit(add_played_rate_target(train))
    rate = played_rate.predict(rows)
    games = rows[OFFSET].to_numpy(float)
    expected_games = games * rate * keep
    total = ros.mean(axis=1)
    # His rate when he plays, modelled directly rather than read off the total.
    # Dividing the blended total by expected games mixes a numerator that is
    # pulled toward a healthy-season board curve with a denominator discounted
    # for this player's own absences, and put Nacua at 32 a game.
    rate_train = add_points_per_active_game_target(train)
    rate_train = rate_train[rate_train[RATE_TARGET].notna()]
    rate_model = build().fit(rate_train, rate_train[RATE_TARGET].to_numpy(float))
    active_rate = rate_model.predict_samples(rows, draws=args.draws, seed=seed).mean(axis=1)
    active_rate = np.where(keep > 0, active_rate, np.nan)
    ros_out = pd.DataFrame({
        "player": rows["player_name"].to_numpy(),
        "position": rows["position"].to_numpy(),
        "team": rows["team"].to_numpy(),
        "games_left": games,
        "expected_games_played": expected_games,
        "rest_of_season_points": total,
        "points_per_active_game": active_rate,
        "points_per_scheduled_game": total / np.where(games > 0, games, np.nan),
        "p10": np.quantile(ros, 0.10, axis=1),
        "p50": np.quantile(ros, 0.50, axis=1),
        "p90": np.quantile(ros, 0.90, axis=1),
        "adp_rank": rows["adp_rank"].to_numpy(),
        "roster_status": rows["status"].to_numpy(),
        "out_through_week": np.where(out_through >= args.week, out_through, np.nan),
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
