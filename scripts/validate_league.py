"""What is a better projection actually worth, in wins?

The weekly layer measures projections in CRPS. A manager is not paid in CRPS. This
runs each policy through the league environment for real seasons and reports what
it earns in the currency that matters -- points, wins, and finishing position --
against eleven opponents all playing the same naive strategy.

Four policies, in increasing order of what they know:

``adp``
    Start the preseason board, all season. Never updates.
``ewma``
    Start whoever has been scoring. The naive baseline the weekly work measured.
``adp-then-ewma``
    The board for three weeks, then recent form. The environment's standard
    opponent, and therefore the line every other policy has to clear.
``oracle``
    Starts the players who actually scored. Not a competitor -- it is the
    ceiling, and without it a small win is indistinguishable from a large one.

**Every result is averaged over seeds, and the reason is measured rather than
assumed.** A policy playing the field's own strategy averages 7.08 wins of 14
but ranges from 4 to 11 across seeds: one season is roughly plus or minus three
wins of pure luck, so a single-season comparison of two policies is worth almost
nothing. The seed controls the draft order and the schedule, which is exactly
the variance a real manager faces and cannot control.

    python scripts/validate_league.py --seasons 2023 2024 2025 --seeds 20
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.league.config import LeagueConfig
from ffmodel.league.env import FantasyLeagueEnv, run_episode
from ffmodel.league.policies import AdpPolicy, EwmaPolicy, PerfectPolicy, SeasonPolicy
from ffmodel.league.pool import build_player_pool


def policies(pool: pd.DataFrame, season: int) -> dict:
    truth = pool[pool["season"] == season][["player_key", "week", "points"]]
    return {
        "adp": AdpPolicy(),
        "ewma": EwmaPolicy(),
        "adp-then-ewma": SeasonPolicy(),
        "oracle": PerfectPolicy(truth=truth),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--last-week", type=int, default=14)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    pool = build_player_pool(args.seasons)
    config = LeagueConfig(last_week=args.last_week)
    print(
        f"{len(args.seasons)} season(s) x {args.seeds} seeds = "
        f"{len(args.seasons) * args.seeds} episodes per policy, "
        f"{config.teams} teams, {config.slots.size}-man rosters, "
        f"weeks {config.first_week}-{config.last_week}"
    )

    rows = []
    for season in args.seasons:
        for name, policy in policies(pool, season).items():
            for seed in range(args.seeds):
                env = FantasyLeagueEnv(pool, season=season, config=config, seed=seed)
                result = run_episode(env, policy)
                standings = result.standings
                rank = int(standings.loc[standings["is_agent"], "rank"].iloc[0])
                rows.append(
                    {
                        "season": season,
                        "policy": name,
                        "seed": seed,
                        "wins": result.wins,
                        "points": result.total_points,
                        "reward": result.total_reward,
                        "rank": rank,
                        "title": int(rank == 1),
                    }
                )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("policy")
        .agg(
            episodes=("wins", "size"),
            wins=("wins", "mean"),
            points=("points", "mean"),
            points_per_week=("points", lambda s: s.mean() / len(config.weeks)),
            rank=("rank", "mean"),
            title_rate=("title", "mean"),
            reward=("reward", "mean"),
        )
        .sort_values("wins")
    )

    print("\n=== averaged over every episode ===")
    print(summary.round(3).to_string())

    if "adp-then-ewma" in summary.index:
        base = summary.loc["adp-then-ewma"]
        print("\n=== against the opponents' own strategy ===")
        for name, row in summary.iterrows():
            if name == "adp-then-ewma":
                continue
            print(
                f"  {name:16s} {row['wins'] - base['wins']:+.2f} wins, "
                f"{row['points'] - base['points']:+.1f} points, "
                f"rank {row['rank'] - base['rank']:+.2f}"
            )

    print("\n=== by season (wins) ===")
    print(
        frame.pivot_table(index="policy", columns="season", values="wins")
        .round(2)
        .to_string()
    )

    # The spread across seeds, which is what says whether any of the above is
    # a real difference or a draft that happened to go well.
    print("\n=== win spread across seeds (std, min-max) ===")
    spread = frame.groupby("policy")["wins"].agg(["std", "min", "max"])
    print(spread.round(2).to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "summary": summary.reset_index().to_dict("records"),
                    "episodes": frame.to_dict("records"),
                },
                indent=2,
                default=str,
            ),
            "utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
