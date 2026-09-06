"""Is the waiver reward a signal an agent could actually learn from?

The credit in :mod:`ffmodel.league.credit` is the reward an RL agent will be
trained on for add/drop decisions, and a reward with the wrong sign, no spread,
or no correlation to winning teaches the wrong thing very efficiently. So before
anything trains on it, this checks the three properties it has to have.

**It has to separate good claims from bad ones.** Four waiver policies are run in
the same seats: standing pat, claiming the best free agent by recent form,
claiming at random, and an oracle that claims whoever will actually score most
from here on. If the credit does not order those correctly it is not measuring
what it claims to.

**It has to agree with the scoreboard.** The credit is a rest-of-season lineup
differential, computed after the fact; wins are what the season pays. If a policy
banks credit and does not win more games, the credit is measuring something other
than value and would train an agent to chase it.

**The two formulations have to be compared, not assumed.** ``gross`` -- the added
player's points minus the dropped player's -- and ``marginal`` -- what the swap
was worth over the lineup that would otherwise have been set -- are both
reported, because the gap between them is the size of the churn incentive that
choosing ``gross`` would build into the agent.

    python scripts/validate_waivers.py --seasons 2023 2024 2025 --seeds 10
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
from ffmodel.league.credit import grade_claims
from ffmodel.league.env import FantasyLeagueEnv, WaiverClaim
from ffmodel.league.policies import EwmaPolicy, SeasonPolicy
from ffmodel.league.pool import build_player_pool


def _droppable(env, observation, values):
    """The worst player the agent could cut: bench only, never a starter."""
    from ffmodel.league.lineup import optimal_lineup

    roster = observation["roster"]
    lineup = optimal_lineup(roster, env.positions, values, env.config.slots)
    bench = [key for key in roster if key not in set(lineup.starting_keys())]
    if not bench:
        return None
    return min(bench, key=lambda key: (values.get(key, 0.0), key))


def stand_pat(env, observation):
    return None


def best_available(env, observation):
    """Claim the free agent recent form likes most, cut the worst bench player.

    The obvious heuristic, and the one an agent has to beat to be worth training.
    """
    shortlist = observation["free_agents"]
    if not shortlist:
        return None
    policy = EwmaPolicy()
    week, history, board = observation["week"], observation["history"], observation["board"]
    values = policy.score(
        list(observation["roster"]) + list(shortlist), history, week, board
    )
    drop = _droppable(env, observation, values)
    if drop is None:
        return None
    add = max(shortlist, key=lambda key: (values.get(key, 0.0), key))
    if values.get(add, 0.0) <= values.get(drop, 0.0):
        return None
    return WaiverClaim(add_key=add, drop_key=drop)


def random_claim(rng):
    def choose(env, observation):
        shortlist = observation["free_agents"]
        if not shortlist:
            return None
        policy = EwmaPolicy()
        values = policy.score(
            observation["roster"], observation["history"],
            observation["week"], observation["board"],
        )
        drop = _droppable(env, observation, values)
        if drop is None:
            return None
        return WaiverClaim(add_key=str(rng.choice(shortlist)), drop_key=drop)

    return choose


def oracle_claim(env_frame):
    """Claim whoever will really score most from here. The ceiling, not a rival."""

    def choose(env, observation):
        shortlist = observation["free_agents"]
        if not shortlist:
            return None
        week = observation["week"]
        future = env_frame[env_frame["week"] >= week].groupby("player_key")["points"].sum()
        policy = EwmaPolicy()
        values = policy.score(
            observation["roster"], observation["history"], week, observation["board"]
        )
        drop = _droppable(env, observation, values)
        if drop is None:
            return None
        add = max(shortlist, key=lambda key: (future.get(key, 0.0), key))
        return WaiverClaim(add_key=add, drop_key=drop)

    return choose


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--last-week", type=int, default=14)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    pool = build_player_pool(args.seasons)
    config = LeagueConfig(last_week=args.last_week)
    rng = np.random.default_rng(0)

    rows = []
    for season in args.seasons:
        block = pool[pool["season"] == season]
        waivers = {
            "stand-pat": stand_pat,
            "best-available": best_available,
            "random": random_claim(rng),
            "oracle-waiver": oracle_claim(block),
        }
        for name, chooser in waivers.items():
            for seed in range(args.seeds):
                env = FantasyLeagueEnv(pool, season=season, config=config, seed=seed)
                policy = SeasonPolicy()
                observation = env.reset()
                while not env.done:
                    claim = chooser(env, observation)
                    if claim is not None:
                        observation = env.submit_claim(claim)
                    scores = policy.score(
                        observation["roster"], observation["history"],
                        observation["week"], observation["board"],
                    )
                    observation, _, _, _ = env.step(scores)
                credits = grade_claims(env)
                standings = env.result.standings
                rows.append(
                    {
                        "season": season,
                        "waiver": name,
                        "seed": seed,
                        "wins": env.result.wins,
                        "points": env.result.total_points,
                        "rank": int(standings.loc[standings["is_agent"], "rank"].iloc[0]),
                        "claims": len(credits),
                        "marginal": sum(credit.marginal for credit in credits),
                        "gross": sum(credit.gross for credit in credits),
                    }
                )

    frame = pd.DataFrame(rows)
    summary = frame.groupby("waiver").agg(
        episodes=("wins", "size"),
        wins=("wins", "mean"),
        points=("points", "mean"),
        rank=("rank", "mean"),
        claims=("claims", "mean"),
        marginal=("marginal", "mean"),
        gross=("gross", "mean"),
    ).sort_values("wins")
    print("\n=== waiver policies, averaged over every episode ===")
    print(summary.round(2).to_string())

    # Paired against standing pat, which is the honest control: same seat, same
    # draft, same schedule, the only difference being whether a claim was made.
    wide = frame.pivot_table(index=["season", "seed"], columns="waiver", values="wins")
    print("\n=== against standing pat (paired on seed) ===")
    print(f"  {'waiver':16s} {'wins':>7s} {'SE':>6s} {'t':>7s}")
    for name in summary.index:
        if name == "stand-pat":
            continue
        diff = wide[name] - wide["stand-pat"]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"  {name:16s} {diff.mean():+7.2f} {se:6.2f} {diff.mean() / se:+7.2f}")

    # Does banked credit actually show up as wins? If it does not, the reward is
    # measuring something other than value.
    made = frame[frame["claims"] > 0]
    if len(made) > 2:
        print("\n=== does credit predict the scoreboard? ===")
        for column in ("marginal", "gross"):
            print(
                f"  {column:8s} vs wins   r = {made[column].corr(made['wins']):+.3f}"
                f"   vs points  r = {made[column].corr(made['points']):+.3f}"
            )
        print(
            f"\n  gross exceeds marginal by "
            f"{(made['gross'] - made['marginal']).mean():+.1f} points per season "
            "on average -- the churn incentive that picking gross would build in."
        )

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
