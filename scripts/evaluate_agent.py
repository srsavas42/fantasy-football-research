"""Put the trained agent next to the alternatives it has to beat.

    python scripts/evaluate_agent.py --agent artifacts/league_agent.json

"The agent beat the field by N" is not on its own an argument for having trained
anything. The field is a specific, beatable strategy, and most of the distance to
it can be walked by a one-line heuristic. So this scores four things in the same
seats, paired on seed:

``field``
    The standard opponent in the agent's own seat -- the draft board for three
    weeks, then a one-game average. The line to clear.
``ewma2``
    The agent's own parameterisation with a single weight on a two-game average
    and nothing else. This is the honest control: it isolates *what the search
    found* from what the feature set and the waiver machinery were worth before
    any searching happened.
``learned``
    The trained parameters.
``oracle``
    Perfect start/sit. The ceiling, so a gain can be read as a fraction of what
    was available rather than as a bare number.

Every comparison is paired on (season, seed), because a season swings about plus
or minus three wins on the draft slot alone and an unpaired difference of half a
win is unmeasurable at any sample size this repo can afford.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.league.agent import PARAMETER_COUNT, LinearAgent, Scaler, as_waiver_policy
from ffmodel.league.config import LeagueConfig
from ffmodel.league.credit import grade_claims
from ffmodel.league.env import run_episode
from ffmodel.league.features import FEATURE_COLUMNS, build_feature_tables
from ffmodel.league.policies import PerfectPolicy, SeasonPolicy
from ffmodel.league.pool import build_player_pool
from ffmodel.league.train import Arena, Task


def ewma2_vector(claim_threshold: float) -> np.ndarray:
    theta = np.zeros(PARAMETER_COUNT)
    theta[FEATURE_COLUMNS.index("ewma2")] = 1.0
    theta[-1] = claim_threshold
    return theta


def play(arena: Arena, task: Task, kind: str, theta=None, scaler=None) -> dict:
    table = arena.tables[task.season]
    env = arena.environment(task)
    if kind == "field":
        result = run_episode(env, SeasonPolicy(table=table))
        claims = []
    elif kind == "oracle":
        truth = arena.pool[arena.pool["season"] == task.season][
            ["player_key", "week", "points"]
        ]
        result = run_episode(env, PerfectPolicy(truth=truth))
        claims = []
    else:
        agent = LinearAgent.from_parameters(theta, table, scaler)
        result = run_episode(env, agent, waiver_policy=as_waiver_policy(agent))
        claims = grade_claims(env)
    standings = result.standings
    return {
        "wins": result.wins,
        "points": result.total_points,
        "reward": result.total_reward,
        "rank": int(standings.loc[standings["is_agent"], "rank"].iloc[0]),
        "title": int(standings.loc[standings["is_agent"], "rank"].iloc[0] == 1),
        "claims": len(claims),
        "credit": float(sum(c.marginal for c in claims)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("artifacts/league_agent.json"))
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("reports/league_agent_eval.json"))
    args = parser.parse_args(argv)

    saved = json.loads(args.agent.read_text("utf-8"))
    theta = np.asarray(saved["theta"], float)
    scaler = Scaler.from_dict(saved["scaler"])

    pool = build_player_pool(args.seasons)
    tables = build_feature_tables(pool, args.seasons)
    arena = Arena(pool=pool, tables=tables, config=LeagueConfig())
    tasks = [Task(s, seed) for s in args.seasons for seed in range(args.seeds)]
    print(f"{len(tasks)} seats: seasons {args.seasons} x {args.seeds} seeds")

    contenders = {
        "field": ("field", None),
        "ewma2": ("agent", ewma2_vector(theta[-1])),
        "learned": ("agent", theta),
        "oracle": ("oracle", None),
    }
    rows = []
    for name, (kind, vector) in contenders.items():
        for task in tasks:
            record = play(arena, task, kind, theta=vector, scaler=scaler)
            record.update(policy=name, season=task.season, seed=task.seed)
            rows.append(record)
    frame = pd.DataFrame(rows)

    summary = frame.groupby("policy").agg(
        wins=("wins", "mean"), points=("points", "mean"), rank=("rank", "mean"),
        title_rate=("title", "mean"), claims=("claims", "mean"),
        credit=("credit", "mean"),
    )
    order = ["field", "ewma2", "learned", "oracle"]
    print("\n=== averaged over every seat ===")
    print(summary.loc[[o for o in order if o in summary.index]].round(3).to_string())

    wide = {
        metric: frame.pivot_table(index=["season", "seed"], columns="policy", values=metric)
        for metric in ("wins", "points")
    }

    def paired(a: str, b: str, metric: str):
        diff = wide[metric][a] - wide[metric][b]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        return diff.mean(), se, (diff.mean() / se if se else float("nan"))

    print("\n=== paired on seed ===")
    print(f"  {'comparison':28s} {'wins':>7s} {'SE':>6s} {'t':>7s}   {'points':>9s} {'t':>7s}")
    comparisons = [
        ("learned", "field", "the line to clear"),
        ("ewma2", "field", "what the features alone give"),
        ("learned", "ewma2", "what the search added"),
        ("oracle", "field", "the ceiling"),
    ]
    for a, b, _ in comparisons:
        if a not in wide["wins"] or b not in wide["wins"]:
            continue
        w, wse, wt = paired(a, b, "wins")
        p, _, pt = paired(a, b, "points")
        print(f"  {a + ' - ' + b:28s} {w:+7.3f} {wse:6.3f} {wt:+7.2f}   {p:+9.1f} {pt:+7.2f}")

    ceiling, _, _ = paired("oracle", "field", "wins")
    gain, _, _ = paired("learned", "field", "wins")
    print(
        f"\n  the agent captured {100.0 * gain / ceiling:.0f}% of the "
        f"{ceiling:.2f}-win band a perfect start/sit would have taken"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "summary": summary.reset_index().to_dict("records"),
                    "episodes": frame.to_dict("records"),
                    "theta": theta.tolist(),
                    "features": list(FEATURE_COLUMNS),
                },
                indent=2, default=str,
            ),
            "utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
