"""Training the agent, under a signal-to-noise ratio that decides the method.

Three facts about this problem pick the algorithm, and none of them is a
preference:

**The lineup is an argmax.** The environment turns the policy's scores into a
lineup by exact constrained assignment. That is a discrete operation with no
useful gradient, so a policy-gradient method would have to either relax it or
sample lineups and eat the variance. A search that only ever needs the *return*
of a parameter vector never touches it.

**The reward is buried in noise.** A season swings about plus or minus three wins
on the draft slot and schedule alone, against a total prize of 2.58. Any method
that reads a single episode as evidence will chase seeds.

**There are sixteen parameters.** At that size a population-based search covers
the space perfectly well, and the sample efficiency a gradient would buy is not
worth the machinery.

So: the cross-entropy method, with three variance reductions that matter more
than the optimiser does.

*Common random numbers.* Every candidate in a generation is evaluated on the same
(season, seed) pairs. Two candidates then differ by their parameters and nothing
else, which is the same reason every comparison in this repo is paired.

*A control variate.* Fitness is the candidate's return **minus the standard
opponent's return in the same seat, same season, same seed**. The baseline is
computed once and reused. This removes the "was this a good draft slot" component
of the return, which is most of its variance and none of its signal.

*A fresh batch each generation.* The seeds are resampled between generations, so
a candidate that happens to suit one batch does not survive on that alone. The
batch is common within a generation and different across them, which is the
combination that gives low-variance comparison without letting the search fit the
sample.

**The waiver reward is not added to the return, and that is deliberate.** The
episodic return already contains every point a claim produced, because a claimed
player who starts scores into the lineup. The credit in
:mod:`ffmodel.league.credit` is a *credit-assignment* device -- it answers "which
decision earned this" for a method that needs to attribute a return to individual
actions. A method scored on the whole season does not need that attribution and
adding the credit on top would count the same points twice, rewarding churn
exactly as gross credit would. The credit is reported for the trained agent as a
diagnostic instead, which is what it is good for here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ffmodel.league.agent import PARAMETER_COUNT, LinearAgent, Scaler, as_waiver_policy
from ffmodel.league.config import LeagueConfig
from ffmodel.league.env import FantasyLeagueEnv, run_episode
from ffmodel.league.policies import EwmaPolicy, SeasonPolicy


@dataclass
class Task:
    """One evaluation: a season and a seed, which together fix the whole world."""

    season: int
    seed: int


@dataclass
class Arena:
    """Everything an episode needs, built once and reused for every candidate."""

    pool: pd.DataFrame
    tables: dict
    config: LeagueConfig = field(default_factory=LeagueConfig)

    def environment(self, task: Task) -> FantasyLeagueEnv:
        table = self.tables[task.season]
        return FantasyLeagueEnv(
            self.pool,
            season=task.season,
            config=self.config,
            seed=task.seed,
            opponent=SeasonPolicy(table=table),
            roster_valuation=EwmaPolicy(table=table),
        )

    def baseline(self, task: Task) -> float:
        """The standard opponent's return in the agent's own seat."""
        env = self.environment(task)
        result = run_episode(env, SeasonPolicy(table=self.tables[task.season]))
        return result.total_reward

    def evaluate(self, theta: np.ndarray, task: Task, scaler: Scaler) -> dict:
        env = self.environment(task)
        agent = LinearAgent.from_parameters(theta, self.tables[task.season], scaler)
        result = run_episode(env, agent, waiver_policy=as_waiver_policy(agent))
        standings = result.standings
        return {
            "reward": result.total_reward,
            "wins": result.wins,
            "points": result.total_points,
            "rank": int(standings.loc[standings["is_agent"], "rank"].iloc[0]),
            "claims": sum(1 for week in result.weeks if week.claim is not None),
        }


# The worker holds the arena in a module global so a forked process inherits it
# instead of pickling a pool and ten feature tables for every task.
_ARENA: Arena | None = None
_SCALER: Scaler | None = None


def _init_worker(arena: Arena, scaler: Scaler) -> None:
    global _ARENA, _SCALER
    _ARENA, _SCALER = arena, scaler


def _run_one(job):
    theta, season, seed = job
    return _ARENA.evaluate(np.asarray(theta), Task(season, seed), _SCALER)["reward"]


def _run_baseline(job):
    season, seed = job
    return _ARENA.baseline(Task(season, seed))


@dataclass
class Generation:
    """What one round of the search did, kept so a run can be read afterwards."""

    index: int
    best: float
    mean: float
    elite_mean: float
    sigma: float
    theta: np.ndarray


class CrossEntropyTrainer:
    """Fit the agent's parameters by cross-entropy search."""

    def __init__(
        self,
        arena: Arena,
        scaler: Scaler,
        *,
        seasons,
        seeds: int = 40,
        population: int = 24,
        elite_fraction: float = 0.25,
        batch: int = 12,
        sigma: float = 0.5,
        sigma_floor: float = 0.02,
        smoothing: float = 0.7,
        rng: np.random.Generator | None = None,
        workers: int = 1,
    ) -> None:
        self.arena = arena
        self.scaler = scaler
        self.tasks = [Task(int(s), seed) for s in seasons for seed in range(seeds)]
        self.population = population
        self.elites = max(2, int(round(population * elite_fraction)))
        self.batch = min(batch, len(self.tasks))
        self.sigma_floor = sigma_floor
        self.smoothing = smoothing
        self.rng = rng or np.random.default_rng(0)
        self.workers = max(1, int(workers))

        self.mu = np.zeros(PARAMETER_COUNT)
        self.sigma = np.full(PARAMETER_COUNT, float(sigma))
        self._baselines: dict[tuple[int, int], float] = {}
        self.history: list[Generation] = []
        self._pool = None

    # ------------------------------------------------------------ machinery

    def _start_pool(self):
        if self.workers == 1 or self._pool is not None:
            return
        import multiprocessing as mp

        context = mp.get_context("fork")
        self._pool = context.Pool(
            self.workers, initializer=_init_worker, initargs=(self.arena, self.scaler)
        )

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def _map(self, function, jobs):
        if self.workers == 1:
            _init_worker(self.arena, self.scaler)
            return [function(job) for job in jobs]
        self._start_pool()
        return self._pool.map(function, jobs, chunksize=1)

    def baselines(self, tasks) -> np.ndarray:
        """The control variate, computed once per task and cached."""
        missing = [t for t in tasks if (t.season, t.seed) not in self._baselines]
        if missing:
            values = self._map(_run_baseline, [(t.season, t.seed) for t in missing])
            for task, value in zip(missing, values):
                self._baselines[(task.season, task.seed)] = value
        return np.array([self._baselines[(t.season, t.seed)] for t in tasks])

    # -------------------------------------------------------------- search

    def fitness(self, population: np.ndarray, tasks) -> np.ndarray:
        """Mean return above the standard opponent, on the same seats."""
        base = self.baselines(tasks)
        jobs = [
            (theta, task.season, task.seed) for theta in population for task in tasks
        ]
        rewards = np.asarray(self._map(_run_one, jobs), float)
        rewards = rewards.reshape(len(population), len(tasks))
        return (rewards - base[None, :]).mean(axis=1)

    def step(self, index: int) -> Generation:
        # A fresh batch every generation, common to every candidate within it.
        chosen = self.rng.choice(len(self.tasks), size=self.batch, replace=False)
        tasks = [self.tasks[i] for i in chosen]

        population = self.rng.normal(
            self.mu, self.sigma, size=(self.population, PARAMETER_COUNT)
        )
        # The incumbent competes in its own population. Without it a generation
        # of unlucky draws can walk the mean somewhere worse and there is nothing
        # holding the line.
        population[0] = self.mu
        scores = self.fitness(population, tasks)

        elite = population[np.argsort(scores)[-self.elites :]]
        # Smoothed rather than replaced: the fitness of a batch is itself noisy,
        # and jumping the mean onto one batch's elites is how a search starts
        # tracking the sample instead of the objective.
        self.mu = self.smoothing * elite.mean(axis=0) + (1 - self.smoothing) * self.mu
        self.sigma = np.maximum(
            self.smoothing * elite.std(axis=0) + (1 - self.smoothing) * self.sigma,
            self.sigma_floor,
        )
        record = Generation(
            index=index,
            best=float(scores.max()),
            mean=float(scores.mean()),
            elite_mean=float(scores[np.argsort(scores)[-self.elites :]].mean()),
            sigma=float(self.sigma.mean()),
            theta=self.mu.copy(),
        )
        self.history.append(record)
        return record

    def run(self, generations: int, *, log=print) -> np.ndarray:
        try:
            for index in range(generations):
                record = self.step(index)
                log(
                    f"gen {record.index:3d}  best {record.best:+8.1f}  "
                    f"elite {record.elite_mean:+8.1f}  mean {record.mean:+8.1f}  "
                    f"sigma {record.sigma:.3f}"
                )
        finally:
            self.close()
        return self.mu


def save_agent(path: Path, theta: np.ndarray, scaler: Scaler, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"theta": np.asarray(theta).tolist(), "scaler": scaler.to_dict(), **meta},
            indent=2,
            default=str,
        ),
        "utf-8",
    )


def load_agent(path: Path, table: pd.DataFrame) -> LinearAgent:
    data = json.loads(Path(path).read_text("utf-8"))
    return LinearAgent.from_parameters(
        np.asarray(data["theta"], float), table, Scaler.from_dict(data["scaler"])
    )
