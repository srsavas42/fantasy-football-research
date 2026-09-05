"""The league as a step-able environment.

One episode is one historical season. Twelve teams are drafted from that
season's consensus board, play a head-to-head schedule, and score the points the
players really scored. Eleven of them are run by a fixed policy; the twelfth is
whatever is being evaluated -- a heuristic, the shipped weekly model, or a
learned agent.

**The environment holds the future and must not leak it.** It has to, because it
scores the week. So the observation handed out at week `w` is built from a frame
truncated to weeks strictly before `w`, and that truncation happens in one place
(:meth:`FantasyLeagueEnv._history_before`) rather than being each policy's
responsibility. :class:`ffmodel.league.policies.PerfectPolicy` is the single
deliberate exception, and it exists to measure headroom rather than to compete.

The action is a score per rostered player plus an optional waiver claim, not a
lineup. The environment does the constrained assignment itself, so a policy is
judged on ranking its players rather than on satisfying the roster rules -- see
:mod:`ffmodel.league.lineup`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ffmodel.league.config import LeagueConfig
from ffmodel.league.draft import run_draft
from ffmodel.league.lineup import Lineup, optimal_lineup, round_robin, score_lineup
from ffmodel.league.policies import Policy, SeasonPolicy


@dataclass
class WaiverClaim:
    """Swap one rostered player for one free agent. ``None`` is standing pat."""

    add_key: str
    drop_key: str


@dataclass
class WeekResult:
    """What happened to the agent's team in one week."""

    week: int
    points: float
    opponent_points: float
    opponent_id: int
    won: bool
    tied: bool
    reward: float
    lineup: Lineup
    claim: WaiverClaim | None = None


@dataclass
class SeasonResult:
    """A whole episode, from the agent's point of view."""

    weeks: list[WeekResult] = field(default_factory=list)
    standings: pd.DataFrame | None = None

    @property
    def total_points(self) -> float:
        return float(sum(week.points for week in self.weeks))

    @property
    def wins(self) -> int:
        return int(sum(week.won for week in self.weeks))

    @property
    def losses(self) -> int:
        return int(sum(not week.won and not week.tied for week in self.weeks))

    @property
    def ties(self) -> int:
        return int(sum(week.tied for week in self.weeks))

    @property
    def total_reward(self) -> float:
        return float(sum(week.reward for week in self.weeks))

    def summary(self) -> dict:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "points": round(self.total_points, 2),
            "points_per_week": round(
                self.total_points / max(len(self.weeks), 1), 2
            ),
            "reward": round(self.total_reward, 2),
        }


class FantasyLeagueEnv:
    """A season of head-to-head fantasy, replayed from real scores.

    Usage mirrors the Gym convention without taking the dependency::

        env = FantasyLeagueEnv(pool, season=2024, config=LeagueConfig())
        obs = env.reset()
        while not env.done:
            scores = my_policy(obs)
            obs, reward, done, info = env.step(scores)
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        season: int,
        config: LeagueConfig | None = None,
        *,
        opponent: Policy | None = None,
        agent_team: int = 0,
        seed: int = 0,
    ) -> None:
        self.config = config or LeagueConfig()
        self.season = int(season)
        self.seed = int(seed)
        self.agent_team = int(agent_team)
        self.opponent = opponent or SeasonPolicy()

        block = pool[pool["season"] == self.season].copy()
        if block.empty:
            raise ValueError(f"no pool rows for season {season}")
        weeks = self.config.weeks
        self.frame = block[block["week"].isin(weeks)].reset_index(drop=True)

        # Static per-player facts, looked up constantly during a season.
        players = self.frame.groupby("player_key", as_index=False).agg(
            player_name=("player_name", "first"),
            position=("position", "first"),
            adp_rank=("adp_rank", "first"),
        )
        self.players = players
        self.positions = dict(zip(players["player_key"], players["position"]))
        self.names = dict(zip(players["player_key"], players["player_name"]))
        self.board = players[["player_key", "adp_rank"]].copy()

        # points[(player_key, week)] -> what he actually scored.
        self._points = self.frame.set_index(["player_key", "week"])["points"]

        self.reset()

    # ---------------------------------------------------------------- setup

    def reset(self) -> dict:
        """Draft the league and return the week-1 observation."""
        self.draft = run_draft(self.frame, self.season, self.config, seed=self.seed)
        self.rosters = {team: list(keys) for team, keys in self.draft.rosters.items()}
        self.schedule = round_robin(
            self.config.teams, len(self.config.weeks), seed=self.seed
        )
        drafted = {key for keys in self.rosters.values() for key in keys}
        self.free_agents = [
            key for key in self.players["player_key"] if key not in drafted
        ]
        self.week_index = 0
        self.done = False
        self.result = SeasonResult()
        self._records = {
            team: {"wins": 0, "losses": 0, "ties": 0, "points": 0.0}
            for team in range(self.config.teams)
        }
        return self.observe()

    # ------------------------------------------------------------ observing

    @property
    def week(self) -> int:
        return self.config.weeks[self.week_index]

    def _history_before(self, week: int) -> pd.DataFrame:
        """Every played week strictly before ``week``.

        The single chokepoint for the environment's one hard rule. Everything a
        policy is shown passes through here, so "could a manager have known
        this?" is answered in one place rather than in each policy.
        """
        return self.frame[self.frame["week"] < week]

    def observe(self) -> dict:
        """What the agent is allowed to see this week."""
        if self.done:
            return {}
        week = self.week
        history = self._history_before(week)
        roster = self.rosters[self.agent_team]

        # The waiver shortlist: free agents ranked by recent scoring, because a
        # policy that must consider every one of several hundred is solving a
        # harder problem than a manager reading a sorted waiver page.
        shortlist = self._waiver_shortlist(history)

        opponent_id = self._opponent_for(week, self.agent_team)
        return {
            "season": self.season,
            "week": week,
            "roster": list(roster),
            "positions": {key: self.positions.get(key) for key in roster},
            "names": {key: self.names.get(key) for key in roster},
            "history": history,
            "board": self.board,
            "free_agents": shortlist,
            "opponent_id": opponent_id,
            "opponent_roster": list(self.rosters.get(opponent_id, []))
            if opponent_id is not None
            else [],
            "record": dict(self._records[self.agent_team]),
        }

    def _waiver_shortlist(self, history: pd.DataFrame) -> list[str]:
        """Free agents worth showing, best recent scorers first."""
        if not self.free_agents:
            return []
        if history.empty:
            ranked = (
                self.players[self.players["player_key"].isin(self.free_agents)]
                .sort_values("adp_rank", na_position="last")
            )
            return ranked["player_key"].head(self.config.waiver_shortlist).tolist()
        recent = (
            history[history["player_key"].isin(self.free_agents)]
            .groupby("player_key")["points"]
            .mean()
            .sort_values(ascending=False)
        )
        return recent.head(self.config.waiver_shortlist).index.tolist()

    def _opponent_for(self, week: int, team: int) -> int | None:
        pairs = self.schedule[self.week_index]
        for home, away in pairs:
            if home == team:
                return away
            if away == team:
                return home
        return None

    # --------------------------------------------------------------- acting

    def step(
        self, scores: dict[str, float], claim: WaiverClaim | None = None
    ) -> tuple[dict, float, bool, dict]:
        """Play one week with the agent's player scores and optional claim."""
        if self.done:
            raise RuntimeError("season is over; call reset()")
        week = self.week
        history = self._history_before(week)

        if claim is not None:
            self._apply_claim(self.agent_team, claim)

        # Everybody else decides with the same information the agent had.
        lineups: dict[int, Lineup] = {}
        for team in range(self.config.teams):
            roster = self.rosters[team]
            if team == self.agent_team:
                team_scores = scores
            else:
                team_scores = self.opponent.score(roster, history, week, self.board)
            lineups[team] = optimal_lineup(
                roster, self.positions, team_scores, self.config.slots
            )

        actual = self._actual_points(week)
        totals = {
            team: score_lineup(lineup, actual) for team, lineup in lineups.items()
        }

        for team, points in totals.items():
            self._records[team]["points"] += points

        opponent_id = self._opponent_for(week, self.agent_team)
        agent_points = totals[self.agent_team]
        opponent_points = totals.get(opponent_id, 0.0) if opponent_id is not None else 0.0

        won = opponent_id is not None and agent_points > opponent_points
        tied = opponent_id is not None and agent_points == opponent_points

        # Record every team's result, not just the agent's, so the standings the
        # environment reports are a real league table rather than one row.
        for home, away in self.schedule[self.week_index]:
            self._settle(home, away, totals[home], totals[away])

        reward = self.config.points_weight * agent_points
        if won:
            reward += self.config.win_bonus
        elif tied:
            reward += self.config.tie_bonus

        outcome = WeekResult(
            week=week,
            points=agent_points,
            opponent_points=opponent_points,
            opponent_id=opponent_id if opponent_id is not None else -1,
            won=won,
            tied=tied,
            reward=reward,
            lineup=lineups[self.agent_team],
            claim=claim,
        )
        self.result.weeks.append(outcome)

        self.week_index += 1
        self.done = self.week_index >= len(self.config.weeks)
        if self.done:
            self.result.standings = self.standings()

        info = {
            "week": week,
            "points": agent_points,
            "opponent_points": opponent_points,
            "won": won,
            "tied": tied,
            "lineup": lineups[self.agent_team],
        }
        return (self.observe(), reward, self.done, info)

    def _settle(self, home: int, away: int, home_points: float, away_points: float):
        if home_points > away_points:
            self._records[home]["wins"] += 1
            self._records[away]["losses"] += 1
        elif away_points > home_points:
            self._records[away]["wins"] += 1
            self._records[home]["losses"] += 1
        else:
            self._records[home]["ties"] += 1
            self._records[away]["ties"] += 1

    def _apply_claim(self, team: int, claim: WaiverClaim) -> None:
        roster = self.rosters[team]
        if claim.add_key not in self.free_agents:
            raise ValueError(f"{claim.add_key} is not a free agent")
        if claim.drop_key not in roster:
            raise ValueError(f"{claim.drop_key} is not on team {team}")
        roster.remove(claim.drop_key)
        roster.append(claim.add_key)
        self.free_agents.remove(claim.add_key)
        self.free_agents.append(claim.drop_key)

    def _actual_points(self, week: int) -> dict[str, float]:
        """What every player really scored in ``week``.

        Only ever called to *score* a week that has already been decided, never
        to build an observation.
        """
        try:
            block = self._points.xs(week, level="week")
        except KeyError:
            return {}
        return block.to_dict()

    # ------------------------------------------------------------- reporting

    def standings(self) -> pd.DataFrame:
        rows = [
            {
                "team_id": team,
                "wins": record["wins"],
                "losses": record["losses"],
                "ties": record["ties"],
                "points": round(record["points"], 2),
                "is_agent": team == self.agent_team,
            }
            for team, record in self._records.items()
        ]
        table = pd.DataFrame(rows).sort_values(
            ["wins", "points"], ascending=False, kind="mergesort"
        )
        table["rank"] = range(1, len(table) + 1)
        return table.reset_index(drop=True)


def run_episode(
    env: FantasyLeagueEnv,
    policy: Policy,
    *,
    waiver_policy=None,
) -> SeasonResult:
    """Play a whole season with one policy in the agent's seat."""
    observation = env.reset()
    while not env.done:
        scores = policy.score(
            observation["roster"],
            observation["history"],
            observation["week"],
            observation["board"],
        )
        claim = None
        if waiver_policy is not None:
            claim = waiver_policy(env, observation, scores)
        observation, _reward, _done, _info = env.step(scores, claim)
    return env.result
