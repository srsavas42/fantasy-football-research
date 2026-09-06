"""Train the league agent, and measure it on seasons it has never seen.

    python scripts/train_agent.py --generations 40 --workers 4

Trains on 2016-2022 and evaluates on 2023-2025 -- the same holdout the weekly
model uses. The split is the whole experiment: an agent scored on the seasons it
searched over would be reporting how well sixteen parameters can memorise seven
seasons of draft order, which is a question nobody asked.

**The search starts from a heuristic, not from nothing.** The initial mean puts
its weight on a two-game exponential average, which is roughly the standard
opponent's own strategy and lands within noise of it. That is a deliberate choice
and it is what makes the result readable: the number reported at the end is *what
the search added to a sensible starting point*, rather than a number that mostly
reflects how long it took to rediscover that recent scoring matters. Pass
``--cold-start`` to begin at zeros instead, which is a worse policy by 190 reward
a season and takes several generations to climb out of.

Everything is paired on seed against the standard opponent in the same seat, and
the oracle is reported alongside, because "+15 reward" means nothing without both
the line it clears and the ceiling it is a fraction of.
"""

from __future__ import annotations

import argparse
import json
import time
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
from ffmodel.league.train import Arena, CrossEntropyTrainer, Task, save_agent

# The walk-forward projections need two prior seasons to fit, so they start in
# 2018 and the training window starts with them. Training on 2016-17 without
# them and 2018+ with them would let the search learn "trust the projection"
# from seasons where it is a constant zero.
TRAIN_SEASONS = (2018, 2019, 2020, 2021, 2022)
EVAL_SEASONS = (2023, 2024, 2025)


def warm_start() -> np.ndarray:
    """Weight on a two-game average, and a threshold a claim has to clear."""
    theta = np.zeros(PARAMETER_COUNT)
    theta[FEATURE_COLUMNS.index("ewma2")] = 1.0
    theta[-1] = 0.5
    return theta


def evaluate(arena: Arena, theta, scaler, tasks, *, label: str) -> pd.DataFrame:
    """Play the agent, the standard opponent and the oracle in the same seats."""
    rows = []
    truths = {
        season: arena.pool[arena.pool["season"] == season][
            ["player_key", "week", "points"]
        ]
        for season in {task.season for task in tasks}
    }
    for task in tasks:
        table = arena.tables[task.season]
        agent = LinearAgent.from_parameters(theta, table, scaler)
        # One episode, then graded from the environment it was played in. The
        # claims are a property of that episode, so replaying it to read them
        # would double the cost of every evaluation for nothing.
        env = arena.environment(task)
        learned = run_episode(env, agent, waiver_policy=as_waiver_policy(agent))
        credits = grade_claims(env)
        field = run_episode(arena.environment(task), SeasonPolicy(table=table))
        ceiling = run_episode(
            arena.environment(task), PerfectPolicy(truth=truths[task.season])
        )
        rows.append(
            {
                "split": label,
                "season": task.season,
                "seed": task.seed,
                "agent_wins": learned.wins,
                "agent_points": learned.total_points,
                "agent_reward": learned.total_reward,
                "field_wins": field.wins,
                "field_points": field.total_points,
                "field_reward": field.total_reward,
                "oracle_wins": ceiling.wins,
                "oracle_reward": ceiling.total_reward,
                "claims": len(credits),
                "credit": float(sum(c.marginal for c in credits)),
            }
        )
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame, agent: str, other: str) -> tuple[float, float, float]:
    diff = frame[agent] - frame[other]
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return float(diff.mean()), float(se), float(diff.mean() / se) if se else float("nan")


def report(frame: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: {len(frame)} episodes ===")
    for metric in ("wins", "points" if "agent_points" in frame else "reward"):
        agent_column, field_column = f"agent_{metric}", f"field_{metric}"
        if agent_column not in frame:
            continue
        mean, se, tstat = paired(frame, agent_column, field_column)
        print(
            f"  {metric:7s}  agent {frame[agent_column].mean():8.2f}   "
            f"field {frame[field_column].mean():8.2f}   "
            f"diff {mean:+7.2f} (SE {se:.2f}, t {tstat:+.2f})"
        )
    mean, se, tstat = paired(frame, "agent_wins", "field_wins")
    ceiling, _, _ = paired(frame, "oracle_wins", "field_wins")
    if ceiling:
        print(
            f"  oracle is {ceiling:+.2f} wins over the field; the agent captured "
            f"{100.0 * mean / ceiling:.0f}% of that band"
        )
    print(
        f"  claims {frame['claims'].mean():.1f}/season, "
        f"marginal credit {frame['credit'].mean():+.1f}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seasons", type=int, nargs="+", default=list(TRAIN_SEASONS))
    parser.add_argument("--eval-seasons", type=int, nargs="+", default=list(EVAL_SEASONS))
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--train-seeds", type=int, default=40)
    parser.add_argument("--eval-seeds", type=int, default=25)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument(
        "--exclude", type=str, nargs="*", default=[],
        help="feature names held at zero, so an ablation differs by the feature "
             "alone rather than by which seasons and seeds each arm drew",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/league_agent.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/league_agent.json"))
    args = parser.parse_args(argv)

    seasons = sorted(set(args.train_seasons) | set(args.eval_seasons))
    started = time.time()
    pool = build_player_pool(seasons)
    tables = build_feature_tables(pool, seasons)
    # Standardisation is fitted on the training seasons only. Fitting it on the
    # holdout too would be a small leak, and a pointless one.
    scaler = Scaler.fit({s: tables[s] for s in args.train_seasons})
    arena = Arena(pool=pool, tables=tables, config=LeagueConfig())
    print(f"setup {time.time() - started:.1f}s; {len(FEATURE_COLUMNS)} features")

    mask = np.ones(PARAMETER_COUNT, bool)
    for name in args.exclude:
        if name not in FEATURE_COLUMNS:
            raise SystemExit(f"unknown feature {name!r}")
        mask[FEATURE_COLUMNS.index(name)] = False
    if args.exclude:
        print(f"holding at zero: {', '.join(args.exclude)}")

    trainer = CrossEntropyTrainer(
        arena,
        scaler,
        mask=mask,
        seasons=args.train_seasons,
        seeds=args.train_seeds,
        population=args.population,
        batch=args.batch,
        sigma=args.sigma,
        workers=args.workers,
        rng=np.random.default_rng(args.seed),
    )
    if not args.cold_start:
        trainer.mu = warm_start() * mask
    start_theta = trainer.mu.copy()

    print(
        f"training on {args.train_seasons} ({len(trainer.tasks)} seats), "
        f"population {args.population}, batch {args.batch}, "
        f"{args.generations} generations"
    )
    began = time.time()
    theta = trainer.run(args.generations)
    print(f"search took {(time.time() - began) / 60:.1f} min")

    print("\n=== learned weights ===")
    for name, value in zip(FEATURE_COLUMNS, theta[:-1]):
        print(f"  {name:14s} {value:+8.3f}   (start {start_theta[FEATURE_COLUMNS.index(name)]:+.3f})")
    print(f"  {'claim gap':14s} {theta[-1]:+8.3f}   (start {start_theta[-1]:+.3f})")

    train_tasks = [Task(s, seed) for s in args.train_seasons for seed in range(args.eval_seeds)]
    eval_tasks = [Task(s, seed) for s in args.eval_seasons for seed in range(args.eval_seeds)]
    on_train = evaluate(arena, theta, scaler, train_tasks, label="train")
    on_eval = evaluate(arena, theta, scaler, eval_tasks, label="holdout")
    report(on_train, "TRAIN seasons (searched over)")
    report(on_eval, "HOLDOUT seasons (never seen)")

    frame = pd.concat([on_train, on_eval], ignore_index=True)
    save_agent(
        args.output,
        theta,
        scaler,
        {
            "features": list(FEATURE_COLUMNS),
            "train_seasons": args.train_seasons,
            "generations": args.generations,
            "cold_start": args.cold_start,
            "excluded": args.exclude,
        },
    )
    print(f"\nwrote {args.output}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "theta": theta.tolist(),
                    "features": list(FEATURE_COLUMNS),
                    "history": [
                        {
                            "generation": g.index,
                            "best": g.best,
                            "elite_mean": g.elite_mean,
                            "mean": g.mean,
                            "sigma": g.sigma,
                        }
                        for g in trainer.history
                    ],
                    "episodes": frame.to_dict("records"),
                },
                indent=2,
                default=str,
            ),
            "utf-8",
        )
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
