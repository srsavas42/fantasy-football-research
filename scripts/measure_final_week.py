"""Is the last week of the regular season worth predicting?

Teams with nothing left to play for rest their starters, so the final week's
production is partly a coaching decision made in December that no preseason
input can see. If that is a large effect, the season totals this package
predicts carry noise nobody could forecast, and the model is being charged for
it.

Two separate claims are bundled in the question and they need separating,
because one is about accuracy and the other is about what the number means.

**The noise claim.** The final week is less predictable than a typical week, so
including it inflates the error floor. This is measurable: predict each week
from the player's own average over the *other* weeks and compare how badly that
does in the final week against a mid-season one. If the final week is no worse,
the resting story is real but too small to matter.

**The relevance claim.** Almost no fantasy league plays the NFL's final week --
the season ends at week 17 of 18 since 2021, and week 16 of 17 before that. A
season total that includes it is answering a question nobody asked. This does
not depend on the noise measurement at all: even if the final week were
perfectly predictable, it does not belong in the target.

Both are reported. "Rested" is inferred from snap share rather than from a
transaction, because a healthy scratch and a benching look the same in the box
score and only the snap count distinguishes either from a normal game.
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
from ffmodel.simulation.scoring import fantasy_points

MODEL_POSITIONS = ("QB", "RB", "WR", "TE")
# What counts as an established starter going into the final week. Both are
# needed: a share threshold alone admits a player who started twice in
# December, and a games threshold alone admits a career backup.
STARTER_SHARE = 0.55
STARTER_GAMES = 10
# Below this multiple of his own usual snap share, a starter was not playing a
# normal game. Deliberately generous -- a starter at 40% of his usual load was
# rested or hurt whichever way the team described it.
RESTED_MULTIPLE = 0.5


def regular_season(weekly: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Regular-season weeks, with the last one of each season marked.

    Read from the schedule rather than hard-coded, because the regular season
    was 17 weeks through 2020 and 18 from 2021. Taking the season's last week
    rather than each team's own is safe only because byes fall in weeks 5-14, so
    every team plays in the finale; if that ever changes this needs to become a
    per-team maximum.
    """
    reg = schedules[schedules.game_type.eq("REG")]
    last = reg.groupby("season")["week"].max().rename("final_week")
    out = weekly.merge(last, left_on="season", right_index=True, how="inner")
    return out[out.week.le(out.final_week)].copy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2015, 2025)))
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path("scripts/validation_runs/final_week.json")

    weekly = ingest.load_weekly(args.seasons)
    weekly = weekly[weekly.position.isin(MODEL_POSITIONS)].copy()
    schedules = ingest.load_schedules(args.seasons)
    snaps = ingest.load_snap_counts(args.seasons)

    weekly = regular_season(weekly, schedules)
    weekly["points"] = fantasy_points(weekly, args.scoring)
    weekly["is_final"] = weekly.week.eq(weekly.final_week)

    share = (
        snaps.assign(
            offense_pct=pd.to_numeric(snaps.get("offense_pct"), errors="coerce")
        )
        .groupby(["season", "week", "player"], as_index=False)["offense_pct"]
        .max()
    )
    # nflverse reports the snap percentage as 0-100 in some seasons and 0-1 in
    # others; normalise rather than trust either.
    if share.offense_pct.max() > 1.5:
        share["offense_pct"] = share.offense_pct / 100.0
    weekly = weekly.merge(
        share.rename(columns={"player": "player_name", "offense_pct": "snap_share"}),
        on=["season", "week", "player_name"],
        how="left",
    )

    report: dict[str, object] = {"seasons": args.seasons, "scoring": args.scoring}

    # ---- 1. What share of a season sits in the final week, and who rests? ----
    prior = weekly[~weekly.is_final]
    usual = (
        prior.groupby(["season", "player_id"])
        .agg(
            usual_snap=("snap_share", "mean"),
            usual_points=("points", "mean"),
            games=("week", "count"),
        )
        .reset_index()
    )
    final = weekly[weekly.is_final][
        ["season", "player_id", "player_name", "position", "points", "snap_share"]
    ]
    paired = usual.merge(final, on=["season", "player_id"], how="inner")
    starters = paired[
        paired.usual_snap.ge(STARTER_SHARE) & paired.games.ge(STARTER_GAMES)
    ].copy()
    starters["snap_ratio"] = starters.snap_share / starters.usual_snap
    rested = starters.snap_ratio.lt(RESTED_MULTIPLE)

    report["starters"] = {
        "n": int(len(starters)),
        "rested_or_out_share": float(rested.mean()),
        "mean_points_normal": float(starters.loc[~rested, "points"].mean()),
        "mean_points_rested": float(starters.loc[rested, "points"].mean()),
        "mean_usual_points": float(starters.usual_points.mean()),
    }
    print(
        f"\nESTABLISHED STARTERS IN THE FINAL WEEK "
        f"(snap share >= {STARTER_SHARE:.0%}, {STARTER_GAMES}+ games)\n"
    )
    s = report["starters"]
    print(f"  n                                {s['n']:>8d}")
    print(f"  played under {RESTED_MULTIPLE:.0%} of usual snaps  {s['rested_or_out_share']:>8.1%}")
    print(f"  their usual weekly points        {s['mean_usual_points']:>8.2f}")
    print(f"  final week, normal load          {s['mean_points_normal']:>8.2f}")
    print(f"  final week, rested or out        {s['mean_points_rested']:>8.2f}")

    # ---- 2. Is the final week actually harder to predict? ----
    #
    # The honest comparison. For each week, predict a player's points from his
    # own mean over every *other* week of that season, and score it. A week that
    # is merely low-scoring is not the problem; a week whose deviation from a
    # player's own level is larger than usual is.
    rows = []
    totals = weekly.groupby(["season", "player_id"])["points"].transform("sum")
    counts = weekly.groupby(["season", "player_id"])["points"].transform("count")
    eligible = counts.ge(STARTER_GAMES)
    leave_one_out = (totals - weekly.points) / (counts - 1).clip(lower=1)
    weekly["loo_error"] = (weekly.points - leave_one_out).abs()
    pool = weekly[eligible]
    for label, mask in (
        ("final week", pool.is_final),
        ("weeks 1 to n-1", ~pool.is_final),
    ):
        block = pool[mask]
        rows.append(
            {
                "week_group": label,
                "n": int(len(block)),
                "mean_points": float(block.points.mean()),
                "mae_from_own_average": float(block.loo_error.mean()),
            }
        )
    report["predictability"] = rows
    print("\nPREDICTING A WEEK FROM THE PLAYER'S OWN AVERAGE OVER THE OTHERS\n")
    print(f"  {'week group':16s} {'n':>7s} {'mean pts':>9s} {'MAE':>8s}")
    for row in rows:
        print(
            f"  {row['week_group']:16s} {row['n']:>7d} {row['mean_points']:>9.2f} "
            f"{row['mae_from_own_average']:>8.3f}"
        )
    penalty = (
        rows[0]["mae_from_own_average"] / rows[1]["mae_from_own_average"] - 1.0
    )
    report["final_week_extra_error"] = float(penalty)
    print(f"\n  the final week is {penalty:+.1%} harder to predict than a typical one")

    # ---- 3. How much of a season total does it carry? ----
    season_total = weekly.groupby(["season", "player_id"])["points"].sum()
    final_only = (
        weekly[weekly.is_final].groupby(["season", "player_id"])["points"].sum()
    )
    aligned = pd.concat([season_total.rename("total"), final_only.rename("final")], axis=1)
    aligned = aligned[aligned.total.gt(0)].fillna({"final": 0.0})
    # Ratio of sums, not mean of ratios: a player with eleven points on the
    # season and four of them in the final week would otherwise contribute a 36%
    # observation to an average about aggregate production.
    report["final_week_share_of_season"] = {
        "share": float(aligned["final"].sum() / aligned["total"].sum()),
        "points_mean": float(aligned["final"].mean()),
        "points_sd": float(aligned["final"].std()),
    }
    f = report["final_week_share_of_season"]
    print(
        f"\n  the final week carries {f['share']:.1%} of all season points "
        f"({f['points_mean']:.1f} per player, sd {f['points_sd']:.1f})"
    )

    # ---- 4. Who rests, and could a preseason model have known? ----
    #
    # Resting is a December decision about playoff seeding. Even where it is
    # perfectly identifiable in week 18, the standings that drive it do not
    # exist in August, so from this model's vantage it is noise whatever its
    # size. Splitting by the team's record entering the final week shows
    # whether that is what is actually happening.
    reg = schedules[schedules.game_type.eq("REG")].copy()
    played = reg[reg.home_score.notna() & reg.away_score.notna()]
    results = pd.concat(
        [
            played.assign(
                team=played.home_team, won=played.home_score > played.away_score
            )[["season", "week", "team", "won"]],
            played.assign(
                team=played.away_team, won=played.away_score > played.home_score
            )[["season", "week", "team", "won"]],
        ],
        ignore_index=True,
    )
    final_week = weekly.groupby("season")["final_week"].max()
    before = results.merge(final_week.rename("fw"), left_on="season", right_index=True)
    standing = (
        before[before.week < before.fw]
        .groupby(["season", "team"])["won"]
        .sum()
        .rename("wins_entering")
        .reset_index()
    )
    team_of = weekly[weekly.is_final][["season", "player_id", "team"]].drop_duplicates(
        ["season", "player_id"]
    )
    staked = starters.merge(team_of, on=["season", "player_id"], how="left").merge(
        standing, on=["season", "team"], how="left"
    )
    staked["rested"] = staked.snap_ratio.lt(RESTED_MULTIPLE)
    bands = [
        ("11+ wins (seed likely settled)", staked.wins_entering.ge(11)),
        ("6 to 10 wins (still alive)", staked.wins_entering.between(6, 10)),
        ("5 or fewer wins (eliminated)", staked.wins_entering.le(5)),
    ]
    stakes = []
    print("\nREST RATE BY THE TEAM'S RECORD ENTERING THE FINAL WEEK\n")
    print(f"  {'band':32s} {'n':>6s} {'rested':>8s}")
    for label, mask in bands:
        block = staked[mask & staked.wins_entering.notna()]
        if len(block) < 20:
            continue
        entry = {
            "band": label,
            "n": int(len(block)),
            "rested_share": float(block.rested.mean()),
        }
        stakes.append(entry)
        print(f"  {label:32s} {entry['n']:>6d} {entry['rested_share']:>8.1%}")
    report["rest_by_stakes"] = stakes

    # ---- 5. The arithmetic that decides the question ----
    #
    # Dropping a week cuts the target's mean by roughly its own share while
    # cutting the error by less, because errors across weeks partly cancel and
    # the mean does not. A shorter season is therefore *harder* to project in
    # relative terms unless the dropped week is much noisier than the rest.
    # This scores the same simple projection -- a player's own per-game rate
    # over the season, which no real model beats -- against both targets.
    scored = []
    # One population for both targets. Qualifying separately let the shorter
    # season keep a different set of players, and the gap between the rows was
    # then partly a different pool rather than a different target.
    trimmed = weekly[~weekly.is_final]
    qualified = trimmed.groupby(["season", "player_id"])["points"].count()
    qualified = qualified[qualified.ge(STARTER_GAMES)].index
    for label, frame in (
        ("full season (as shipped)", weekly),
        ("without the final week", trimmed),
    ):
        grouped = frame.groupby(["season", "player_id"])["points"]
        total, games = grouped.sum(), grouped.count()
        total, games = total.reindex(qualified), games.reindex(qualified)
        slate = total.index.get_level_values("season").map(
            frame.groupby("season")["week"].nunique()
        ).to_numpy(dtype=float)
        # The same oracle used on the ADP pool: perfect knowledge of the
        # player's per-game rate, none of who misses time, so everyone is
        # projected at the pool's average availability. Unbiased by
        # construction, and its whole error is games missed.
        rate = (total / games).to_numpy()
        availability = float((games.to_numpy() / slate).mean())
        error = rate * slate * availability - total.to_numpy()
        entry = {
            "target": label,
            "n": int(len(total)),
            "observed_mean": float(total.mean()),
            "mae": float(np.abs(error).mean()),
            "mae_pct": float(np.abs(error).mean() / total.mean()),
        }
        scored.append(entry)
    report["target_length"] = scored
    print("\nSAME PROJECTION, TWO TARGETS (per-game rate times expected slate)\n")
    print(f"  {'target':24s} {'n':>6s} {'obs mean':>9s} {'MAE':>8s} {'MAE %':>7s}")
    for entry in scored:
        print(
            f"  {entry['target']:24s} {entry['n']:>6d} {entry['observed_mean']:>9.2f} "
            f"{entry['mae']:>8.2f} {entry['mae_pct']:>6.1%}"
        )
    swing = scored[1]["mae_pct"] - scored[0]["mae_pct"]
    report["relative_error_change"] = float(swing)
    print(
        f"\n  dropping the final week moves relative error by {swing:+.2%} "
        "in absolute percentage points"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
