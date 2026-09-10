"""Does the agent's edge survive a league that plays as well as it does?

Every number this repo has reported for the trained agent was measured against
one field: `SeasonPolicy` in the other eleven seats -- the draft board for three
weeks, then a one-game average -- and a wire nobody else competes for. That is a
specific, beatable opponent, and arm D has the most capacity of anything trained
here (a hidden layer, 107 parameters). Capacity is exactly what lets a policy
learn an opponent's blind spots instead of the game, so "+1.656 wins" is
consistent with two very different stories:

**general skill**
    D sets better lineups and makes better claims than the alternative, and
    would against anybody.
**a field-specific exploit**
    D found the standard opponent's particular slowness -- it is three weeks
    behind on role changes and never contests a waiver -- and the margin is a
    statement about `SeasonPolicy` rather than about fantasy football.

Absolute wins cannot separate them, because a league is zero-sum: make all
twelve teams better and every record slides toward 7-7 by construction. So the
measurement here is a *margin*, run twice. In each field, two policies play the
same seat -- the trained agent, and the honest control it was always scored
against (its own parameterisation carrying a single weight on a two-game
average) -- and the reported quantity is the paired difference between them. A
margin that holds as the field improves is skill. A margin that collapses was
rent on the opponent's mistakes.

Three fields, in order of how much of the standard opponent's weakness they fix:

``standard``
    What every previous number was measured against. The line to reproduce.
``lineups``
    The other eleven set their cards with the trained policy, but still touch
    the wire only when a rule forces them to.
``full``
    They also transact: the same claim rule, one add per team per phase, taking
    their turn in the same rolling priority queue. This is the field that
    contests the wire, and the wire is where an uncontested policy has the most
    to lose.

Every comparison is paired on (season, seed) -- a season swings about three wins
on the draft slot alone -- and the same seeds are used in all three fields, so
the fields are paired against each other too.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.league.agent import (
    PARAMETER_COUNT,
    Scaler,
    as_opponent_claims,
    as_waiver_policy,
)
from ffmodel.league.config import LeagueConfig
from ffmodel.league.env import FantasyLeagueEnv, run_episode
from ffmodel.league.features import FEATURE_COLUMNS, build_feature_tables
from ffmodel.league.policies import EwmaPolicy, SeasonPolicy
from ffmodel.league.pool import build_player_pool
from ffmodel.league.train import Task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_agent import build_agent, ewma2_vector  # noqa: E402

FIELDS = ("standard", "lineups", "full")


def make_environment(pool, tables, config, task: Task, field: str, opponent_agent):
    """One environment, with the other eleven seats set to ``field``."""
    table = tables[task.season]
    if field == "standard":
        opponent, claims = SeasonPolicy(table=table), None
    elif field == "lineups":
        opponent, claims = opponent_agent, None
    elif field == "full":
        opponent, claims = opponent_agent, as_opponent_claims(opponent_agent)
    else:
        raise ValueError(f"unknown field {field!r}")
    return FantasyLeagueEnv(
        pool,
        season=task.season,
        config=config,
        seed=task.seed,
        opponent=opponent,
        roster_valuation=EwmaPolicy(table=table),
        opponent_claims=claims,
    )


def play(env, agent) -> dict:
    result = run_episode(env, agent, waiver_policy=as_waiver_policy(agent))
    standings = result.standings
    return {
        "wins": result.wins,
        "points": result.total_points,
        "rank": int(standings.loc[standings["is_agent"], "rank"].iloc[0]),
        "title": int(standings.loc[standings["is_agent"], "rank"].iloc[0] == 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("artifacts/league_agent.json"))
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--fields", nargs="+", default=list(FIELDS))
    parser.add_argument("--output", type=Path, default=Path("reports/league_selfplay.json"))
    args = parser.parse_args(argv)

    saved = json.loads(args.agent.read_text("utf-8"))
    theta = np.asarray(saved["theta"], float)
    scaler = Scaler.from_dict(saved["scaler"])

    pool = build_player_pool(args.seasons)
    tables = build_feature_tables(pool, args.seasons)
    config = LeagueConfig()
    tasks = [Task(s, seed) for s in args.seasons for seed in range(args.seeds)]

    # The control's claim threshold is read off the learned agent exactly as
    # `evaluate_agent.py` reads it, so the two seats differ in what they know
    # and not in how eager they are to transact.
    threshold = float(theta[len(FEATURE_COLUMNS)]) if not saved.get("hidden") else 0.5
    seats = {
        "learned": (theta, saved),
        "ewma2": (ewma2_vector(threshold), None),
    }
    print(
        f"{len(tasks)} seats x {len(args.fields)} fields x {len(seats)} policies "
        f"= {len(tasks) * len(args.fields) * len(seats)} episodes"
    )

    rows = []
    for field in args.fields:
        for task in tasks:
            table = tables[task.season]
            # The field's own policy is the trained agent in every case it is
            # used; rebuilt per season because it carries that season's table.
            opponent_agent = build_agent(theta, table, scaler, saved)
            for name, (vector, meta) in seats.items():
                env = make_environment(pool, tables, config, task, field, opponent_agent)
                agent = build_agent(vector, table, scaler, meta)
                record = play(env, agent)
                record.update(
                    field=field, seat=name, season=task.season, seed=task.seed
                )
                rows.append(record)
        print(f"  field {field} done", flush=True)

    frame = pd.DataFrame(rows)
    summary = frame.groupby(["field", "seat"]).agg(
        wins=("wins", "mean"), points=("points", "mean"),
        rank=("rank", "mean"), title_rate=("title", "mean"),
    )
    print("\n=== averaged over every seat ===")
    print(summary.round(3).to_string())

    print("\n=== the margin, paired on (season, seed) ===")
    print(f"  {'field':10s} {'wins':>8s} {'SE':>6s} {'t':>7s}   {'points':>9s} {'t':>7s}")
    margins = {}
    for field in args.fields:
        block = frame[frame["field"] == field]
        wide = {
            metric: block.pivot_table(
                index=["season", "seed"], columns="seat", values=metric
            )
            for metric in ("wins", "points")
        }
        line = []
        for metric in ("wins", "points"):
            diff = wide[metric]["learned"] - wide[metric]["ewma2"]
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            line.append((diff.mean(), se, diff.mean() / se if se else float("nan")))
        margins[field] = {"wins": line[0], "points": line[1]}
        (w, wse, wt), (p, _, pt) = line
        print(f"  {field:10s} {w:+8.3f} {wse:6.3f} {wt:+7.2f}   {p:+9.1f} {pt:+7.2f}")

    if "standard" in margins:
        base = margins["standard"]["wins"][0]
        print("\n=== how much of the margin survives a better field ===")
        for field in args.fields:
            got = margins[field]["wins"][0]
            share = 100.0 * got / base if base else float("nan")
            print(f"  {field:10s} {got:+.3f} wins   {share:5.0f}% of the standard-field margin")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "agent": str(args.agent),
                    "summary": summary.reset_index().to_dict("records"),
                    "margins": {k: {m: list(v) for m, v in d.items()} for k, d in margins.items()},
                    "episodes": rows,
                },
                indent=2, default=str,
            ),
            "utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
