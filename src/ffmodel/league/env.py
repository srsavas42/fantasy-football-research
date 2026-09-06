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

from ffmodel.league.availability import Availability, build_availability
from ffmodel.league.config import LeagueConfig
from ffmodel.league.draft import run_draft
from ffmodel.league.lineup import Lineup, optimal_lineup, round_robin, score_lineup
from ffmodel.league.policies import EwmaPolicy, Policy, SeasonPolicy
from ffmodel.league.roster import Transaction, availability_scores, manage_roster


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
    moves: list[Transaction] = field(default_factory=list)


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
        roster_valuation: Policy | None = None,
        agent_team: int = 0,
        seed: int = 0,
    ) -> None:
        self.config = config or LeagueConfig()
        self.season = int(season)
        self.seed = int(seed)
        self.agent_team = int(agent_team)
        self.opponent = opponent or SeasonPolicy()
        # How every team values players for add/drop/IR decisions. Deliberately
        # separate from the lineup policy -- see :meth:`_manage`.
        self.roster_valuation = roster_valuation or EwmaPolicy()

        block = pool[pool["season"] == self.season].copy()
        if block.empty:
            raise ValueError(f"no pool rows for season {season}")

        # Built from the whole season, before the frame is cut down to the
        # league's weeks. A bye is inferred from the one week a club is absent,
        # and truncating first would hide the bye of every club whose bye falls
        # outside the window -- or, worse, invent one for every club at the edge.
        self.availability = build_availability(block, self.season)

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
        self.ir = {team: [] for team in range(self.config.teams)}
        drafted = {key for keys in self.rosters.values() for key in keys}
        self.free_agents = [
            key for key in self.players["player_key"] if key not in drafted
        ]
        self.week_index = 0
        self.done = False
        self.result = SeasonResult()
        # One entry per played week: the agent's roster and the scores it set
        # its card with. Public because grading a claim after the season needs
        # it -- see :mod:`ffmodel.league.credit` -- and it is a record of weeks
        # already played, so nothing in it could leak the future.
        self.ledger: list[dict] = []
        self._pending_claim: WaiverClaim | None = None
        self._claims_this_week = 0
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
            "ir": list(self.ir[self.agent_team]),
            "positions": {key: self.positions.get(key) for key in roster},
            "names": {key: self.names.get(key) for key in roster},
            "history": history,
            "board": self.board,
            "free_agents": shortlist,
            # Known before kickoff, so stating it leaks nothing: a bye is public
            # in August and a game-status report lands a day early. What it does
            # not cover is the absence nobody sees coming, which stays the
            # manager's risk -- see :mod:`ffmodel.league.availability`.
            "unavailable": self.availability.unavailable(roster, week),
            "availability": self.availability,
            "opponent_id": opponent_id,
            "opponent_roster": list(self.rosters.get(opponent_id, []))
            if opponent_id is not None
            else [],
            "record": dict(self._records[self.agent_team]),
        }

    def _waiver_shortlist(self, history: pd.DataFrame) -> list[str]:
        """Free agents worth showing, best recent scorers first.

        Ranked on the weeks they were active rather than every week on a roster.
        A free agent is usually somebody who has missed time, and averaging in
        the weeks he was hurt is what buries the returning starter who is the
        single most valuable thing on a waiver wire.
        """
        if not self.free_agents:
            return []
        if history.empty:
            ranked = (
                self.players[self.players["player_key"].isin(self.free_agents)]
                .sort_values("adp_rank", na_position="last")
            )
            return ranked["player_key"].head(self.config.waiver_shortlist).tolist()
        block = history[history["player_key"].isin(self.free_agents)]
        if "played" in block.columns:
            active = block[block["played"] == 1]
            if len(active):
                block = active
        recent = (
            block.groupby("player_key")["points"].mean().sort_values(ascending=False)
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
        claim = claim or self._pending_claim
        # Set either way, so a claim passed straight to `step` is shielded from
        # the housekeeping exactly like one made through `submit_claim`.
        self._pending_claim = claim

        # Housekeeping first, for everybody. Activating a returning player and
        # covering a hole change who is even on the roster, so they have to
        # happen before the card is set rather than after it.
        #
        # In waiver priority, worst record first. The order is not a detail: the
        # teams share one free-agent pool, so whoever runs first gets the best
        # replacement, and running them in team order would hand seat 0 -- the
        # agent's -- the top of the wire every week of every season. That is an
        # edge worth roughly the thing being measured. Reverse standings is both
        # the fix and what real leagues do.
        moves = {
            team: self._manage(team, history, week)
            for team in self._waiver_order()
        }

        # Everybody else decides with the same information the agent had.
        lineups: dict[int, Lineup] = {}
        for team in range(self.config.teams):
            roster = self.rosters[team]
            if team == self.agent_team:
                team_scores = self._fill_unscored(dict(scores), roster)
            else:
                team_scores = self.opponent.score(roster, history, week, self.board)
            # Whoever is known not to be playing goes to the bottom. Stated by
            # the environment because it is a fact rather than an opinion, and
            # applied to the scores rather than inside the assignment so a
            # policy that insists on starting an absent player still can.
            team_scores = availability_scores(
                team_scores, roster, self.availability, week
            )
            if team == self.agent_team:
                agent_scores = team_scores
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

        self.ledger.append(
            {
                "week": week,
                "roster": list(self.rosters[self.agent_team]),
                "scores": dict(agent_scores),
                "lineup": lineups[self.agent_team],
                "claim": claim,
            }
        )

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
            moves=moves[self.agent_team],
        )
        self.result.weeks.append(outcome)

        self._pending_claim = None
        self._claims_this_week = 0
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

    def _fill_unscored(
        self, scores: dict[str, float], roster: list[str]
    ) -> dict[str, float]:
        """Give a value to a rostered player the policy never scored.

        Housekeeping can put somebody on the roster after the policy has spoken.
        Leaving him unscored means a zero, which benches him -- and if he arrived
        because a starting slot was empty, benching him leaves it empty, so the
        move that fetched him accomplished nothing.

        The stand-in is the *median of the scored players at his own position*,
        not another policy's opinion. That matters: a policy's numbers can be
        projected points, inverse draft ranks, or a learned value with no units
        at all, and borrowing a second policy's scale would have silently ranked
        a thirty-point breakout below a replacement-level back. A within-scale
        median says "treat him as ordinary for his position", which is the least
        that can be assumed and is guaranteed to be comparable.
        """
        unscored = [key for key in roster if key not in scores]
        if not unscored:
            return scores
        by_position: dict[str, list[float]] = {}
        for key, value in scores.items():
            position = self.positions.get(key)
            if position is not None:
                by_position.setdefault(position, []).append(float(value))
        overall = sorted(float(value) for value in scores.values())
        for key in unscored:
            pool = sorted(by_position.get(self.positions.get(key), [])) or overall
            scores[key] = pool[len(pool) // 2] if pool else 0.0
        return scores

    def submit_claim(self, claim: WaiverClaim) -> dict:
        """Make a waiver claim, then look at the roster it produced.

        Separate from :meth:`step` on purpose. A claim changes who is on the
        roster, and a policy has to score the roster it will actually field --
        so the claim resolves first and the observation that follows already
        includes the new player. Passing a claim to ``step`` instead still works
        and is the older path, but there the policy has already spoken and the
        arrival is valued by :meth:`_fill_unscored` rather than by the policy
        itself.
        """
        self._apply_claim(self.agent_team, claim)
        self._pending_claim = claim
        return self.observe()

    def _waiver_order(self) -> list[int]:
        """Who claims first this week: worst record, then fewest points.

        Ties break on team id, which is stable and therefore reproducible; in
        week 1 every record is identical and the order is simply team order,
        which is the one week it cannot matter because nobody has a hole yet.
        """
        return sorted(
            range(self.config.teams),
            key=lambda team: (
                self._records[team]["wins"],
                self._records[team]["points"],
                team,
            ),
        )

    def _manage(self, team: int, history: pd.DataFrame, week: int) -> list:
        """Run one team's roster housekeeping for the week.

        Every team gets the same mechanic, the agent included. These are rules
        rather than strategy -- a player who is ruled out cannot be started, and
        a starting slot with nobody in it scores nothing -- so applying them to
        eleven teams and not the twelfth would make the comparison dishonest in
        whichever direction happened to help.

        **The valuation is not the team's lineup policy**, and the reason is
        specific. The standard opponent ranks by the draft board for the first
        three weeks, and the board has nothing to say about a player nobody
        drafted: an undrafted breakout scoring thirty a week ranks below every
        rostered player, so the housekeeping would cut him to cover a hole the
        week after he was claimed. It did exactly that before this was fixed.
        Recent production is the only scale on which a rostered player and a
        waiver pickup are comparable, so roster decisions use it throughout,
        while lineup decisions stay the policy's own.

        The agent's explicit claim is protected for the week it is made. It is a
        deliberate decision, and automatic housekeeping undoing a deliberate
        decision in the same breath is not a rule, it is a bug.
        """

        def valuation(keys):
            keys = list(keys)
            if not keys:
                return {}
            return self.roster_valuation.score(keys, history, week, self.board)

        protected = set()
        if team == self.agent_team and self._pending_claim is not None:
            protected.add(self._pending_claim.add_key)

        return manage_roster(
            roster=self.rosters[team],
            ir=self.ir[team],
            free_agents=self.free_agents,
            positions=self.positions,
            availability=self.availability,
            week=week,
            slots=self.config.slots,
            valuation=valuation,
            protected=protected,
        )

    def _apply_claim(self, team: int, claim: WaiverClaim) -> None:
        roster = self.rosters[team]
        if team == self.agent_team:
            if self._claims_this_week >= self.config.waiver_adds_per_week:
                raise ValueError(
                    f"already made {self._claims_this_week} claim(s) in week "
                    f"{self.week}; the limit is "
                    f"{self.config.waiver_adds_per_week}"
                )
            self._claims_this_week += 1
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
        if waiver_policy is not None:
            claim = waiver_policy(env, observation)
            if claim is not None:
                # Resolve the claim before the lineup, so the policy scores the
                # roster it is actually going to field.
                observation = env.submit_claim(claim)
        scores = policy.score(
            observation["roster"],
            observation["history"],
            observation["week"],
            observation["board"],
        )
        observation, _reward, _done, _info = env.step(scores)
    return env.result
