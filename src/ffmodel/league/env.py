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
from ffmodel.league.kickoff import KickoffSlots, build_kickoff_slots
from ffmodel.league.policies import EwmaPolicy, Policy, SeasonPolicy, WeekState
from ffmodel.league.roster import Transaction, availability_scores, manage_roster
from ffmodel.league.waivers import PHASES, WaiverWire, hour_of


class _FixedScores(Policy):
    """The agent's submitted scores, wrapped so every team looks like a policy.

    ``step`` receives numbers rather than a policy, so when nobody has handed
    the environment a reactive agent this stands in: it repeats what was
    submitted, which is what a single-deadline lineup means. Setting
    :attr:`FantasyLeagueEnv.agent_policy` replaces it.
    """

    name = "submitted"
    reactive = False

    def __init__(self) -> None:
        self._scores: dict[str, float] = {}

    def remember(self, scores: dict[str, float]) -> None:
        self._scores = dict(scores)

    def score(self, player_keys, history, week, board, state=None):
        return {key: self._scores.get(key, 0.0) for key in player_keys}


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
    # The first claim that landed, kept for callers that expect one; `claims`
    # is the full list, because there is no cap on adds any more.
    claim: WaiverClaim | None = None
    claims: list[WaiverClaim] = field(default_factory=list)
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
        kickoffs: KickoffSlots | None = None,
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

        # When each player's game kicks off, which is what makes a week a
        # sequence of decisions rather than one. Omitted, the week collapses to
        # a single deadline before any game -- the old behaviour, and still what
        # every non-reactive policy experiences.
        self.kickoffs = kickoffs
        self._club = dict(
            zip(
                zip(self.frame["player_key"], self.frame["week"].astype(int)),
                self.frame["team"].astype(str),
            )
        )
        # The scores `step` was handed, wrapped so the resolver can treat every
        # team as a policy. `agent_policy` is consulted instead only when it is
        # reactive, so a normal agent is asked once a week exactly as before.
        self._submitted = _FixedScores()
        self._agent_policy: Policy | None = None

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
        # Nobody is on waivers at the start: an undrafted player has never been
        # cut, so he is available from week one.
        self.wire = WaiverWire()
        self.wire.drafted(drafted)
        # Waiver priority starts as the inverse of the draft order -- the team
        # that picked last off the board picks first off the wire -- and rolls
        # from there rather than resetting.
        self.waiver_priority = list(reversed(self.draft.order)) or list(
            range(self.config.teams)
        )
        self._claim_queue: list[WaiverClaim] = []
        self.phase = 0
        # Moves made this week, accumulated across its transaction phases. Kept
        # by the environment rather than returned to the caller, because a phase
        # can be run either by `transact` or by `step` and the week's record
        # must read the same either way.
        self._week_moves: dict[int, list] = {}
        self._applied_claims: list[WaiverClaim] = []
        self.week_index = 0
        self.done = False
        self.result = SeasonResult()
        # One entry per played week: the agent's roster and the scores it set
        # its card with. Public because grading a claim after the season needs
        # it -- see :mod:`ffmodel.league.credit` -- and it is a record of weeks
        # already played, so nothing in it could leak the future.
        self.ledger: list[dict] = []
        self._pending_claim: WaiverClaim | None = None
        self._claims_this_phase = 0
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
            # Which transaction phase this is, and who is locked up. A player
            # cut an hour ago is not addable by anybody, which is what makes a
            # drop a commitment rather than a formality.
            "phase": self.phase,
            "phases": len(PHASES),
            "hour": self.hour,
            "on_waivers": self.wire.on_waivers(self.hour),
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
        addable = self.wire.free_agents(self.free_agents, self.hour)
        if not addable:
            return []
        if history.empty:
            ranked = (
                self.players[self.players["player_key"].isin(addable)]
                .sort_values("adp_rank", na_position="last")
            )
            return ranked["player_key"].head(self.config.waiver_shortlist).tolist()
        block = history[history["player_key"].isin(addable)]
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

        # Every transaction phase not already run, then the games. Housekeeping
        # has to precede the card because activating a returning player or
        # covering a hole changes who is even on the roster.
        #
        # In waiver priority, worst record first. The order is not a detail: the
        # teams share one pool, so whoever runs first gets the best replacement,
        # and running them in team order would hand seat 0 -- the agent's -- the
        # top of the wire every week of every season. That is an edge worth
        # roughly the thing being measured. Reverse standings is both the fix
        # and what real leagues do.
        while self.phase < len(PHASES):
            self.transact()
        moves = {
            team: self._week_moves.get(team, [])
            for team in range(self.config.teams)
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
                self._submitted.remember(team_scores)
            lineups[team] = optimal_lineup(
                roster, self.positions, team_scores, self.config.slots
            )

        actual = self._actual_points(week)
        totals = self._resolve_week(week, lineups, history, actual)

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
                "claims": list(self._applied_claims),
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
            claim=self._applied_claims[0] if self._applied_claims else None,
            claims=list(self._applied_claims),
            moves=moves[self.agent_team],
        )
        self.result.weeks.append(outcome)

        self._pending_claim = None
        self._claims_this_phase = 0
        self.phase = 0
        self._week_moves = {}
        self._applied_claims = []
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

    def _slot_of(self, key: str, week: int) -> int | None:
        """Which decision point commits this player, or ``None`` if he is idle."""
        club = self._club.get((key, int(week)))
        if club is None:
            return None
        if self.kickoffs is None:
            return 0
        return self.kickoffs.slot_of_club(club, week)

    def _resolve_week(
        self,
        week: int,
        lineups: dict[int, Lineup],
        history: pd.DataFrame,
        actual: dict[str, float],
    ) -> dict[int, float]:
        """Play the week one kickoff at a time, revising what is still movable.

        Each pass is a decision point: a reactive policy re-scores the players
        whose games have not started, the card is re-optimised around the ones
        that have, and then that slot's games are played. A policy that is not
        reactive is asked once, at the top, because re-running the same scoring
        function on the same information cannot change its mind -- so the common
        case costs exactly what it did when a week was a single decision.
        """
        totals = {team: 0.0 for team in lineups}
        locked_in = {team: set() for team in lineups}
        locked_out = {team: set() for team in lineups}
        reactive = [
            team
            for team in lineups
            if self._policy_for(team).reactive and self._slots_in(week) > 1
        ]

        for slot in range(self._slots_in(week)):
            for team in reactive:
                lineups[team] = self._revise(
                    team, week, slot, lineups[team], locked_in, locked_out,
                    totals, history, lineups_now=lineups,
                )
            # The slot's games are played. Everyone in them is now committed,
            # whichever side of the card they were on.
            for team, lineup in lineups.items():
                starting = set(lineup.starting_keys())
                for key in self.rosters[team]:
                    if self._slot_of(key, week) != slot:
                        continue
                    if key in starting:
                        locked_in[team].add(key)
                        totals[team] += float(actual.get(key, 0.0))
                    else:
                        locked_out[team].add(key)

        # Anyone whose club never took the field -- a bye, or a game that was
        # not played -- scores nothing and has already been benched, so the
        # totals above are complete. Starters with no slot at all are the one
        # exception: a roster too thin to cover a bye can still seat one, and he
        # is worth exactly the zero he scored.
        return totals

    def _slots_in(self, week: int) -> int:
        return self.kickoffs.slots_in(week) if self.kickoffs is not None else 1

    @property
    def agent_policy(self) -> Policy | None:
        return self._agent_policy

    @agent_policy.setter
    def agent_policy(self, policy: Policy | None) -> None:
        """Hand the environment the agent's policy so it can be re-asked.

        Only reactive policies need this. ``step`` takes scores rather than a
        policy, which is the right interface for a single decision and not
        enough for several -- a later decision point has to be able to ask
        again, with the state of the week in hand.
        """
        self._agent_policy = policy

    def _policy_for(self, team: int) -> Policy:
        if team != self.agent_team:
            return self.opponent
        policy = self._agent_policy
        # A non-reactive policy would return the same numbers it already gave,
        # so replaying the submitted card is both equivalent and cheaper.
        if policy is None or not getattr(policy, "reactive", False):
            return self._submitted
        return policy

    def _yet_to_play(self, team, week, lineup, locked_in, locked_out) -> int:
        """Starters on ``team``'s card whose game has not kicked off."""
        if lineup is None:
            return 0
        return sum(
            1
            for key in lineup.starting_keys()
            if key not in locked_in[team]
            and key not in locked_out[team]
            and self._slot_of(key, week) is not None
        )

    def _revise(
        self, team, week, slot, lineup, locked_in, locked_out, totals, history,
        lineups_now=None,
    ) -> Lineup:
        lineups_now = lineups_now or {}
        roster = self.rosters[team]
        opponent_id = self._opponent_for(week, team)
        playable = frozenset(
            key
            for key in roster
            if key not in locked_in[team] and key not in locked_out[team]
        )
        state = WeekState(
            slot=slot,
            slots=self._slots_in(week),
            points=totals[team],
            opponent_points=totals.get(opponent_id, 0.0) if opponent_id is not None else 0.0,
            locked_in=frozenset(locked_in[team]),
            locked_out=frozenset(locked_out[team]),
            playable=playable,
            remaining=self._yet_to_play(team, week, lineup, locked_in, locked_out),
            opponent_remaining=(
                self._yet_to_play(
                    opponent_id, week, lineups_now.get(opponent_id), locked_in, locked_out
                )
                if opponent_id is not None
                else 0
            ),
        )
        scores = self._policy_for(team).score(roster, history, week, self.board, state)
        scores = availability_scores(scores, roster, self.availability, week)
        return optimal_lineup(
            roster,
            self.positions,
            scores,
            self.config.slots,
            locked_in=locked_in[team],
            locked_out=locked_out[team],
        )

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
        """Queue a waiver claim for this transaction phase.

        Queued rather than applied, because the agent transacts at its place in
        the waiver queue like everybody else. It resolves during
        :meth:`transact`, and it can fail there -- a higher-priority team may
        have taken the player first, which is precisely what the priority system
        is for. The observation returned is the roster as it stands *now*, so a
        policy that needs the post-transaction roster should look again after
        the phase.
        """
        self._apply_claim(self.agent_team, claim)
        self._pending_claim = claim
        return self.observe()

    def _waiver_order(self) -> list[int]:
        """Who transacts first, front of the rolling queue to the back."""
        return list(self.waiver_priority)

    def _rotate_priority(self, claimed) -> None:
        """Anyone who added a player this phase goes to the back of the queue.

        A rolling queue rather than reverse standings, which is what the
        environment used before. The difference matters: reverse standings
        recomputes every week, so a bad team keeps first pick indefinitely and
        never pays for using it. A rolling queue makes priority a *resource* --
        spending it on a marginal pickup means the next player worth having goes
        to somebody else, which is the decision the mechanic exists to create.

        Relative order is preserved among those who moved and among those who
        did not, so a phase where everybody claims leaves the queue as it was
        rather than reshuffling it.
        """
        moved = [team for team in self.waiver_priority if team in claimed]
        stayed = [team for team in self.waiver_priority if team not in claimed]
        self.waiver_priority = stayed + moved

    def transact(self) -> dict:
        """Run one transaction phase for every team, then move to the next.

        A week has two: Wednesday, and Friday two days later. The gap is the
        whole point -- a player cut on Wednesday clears waivers on Friday and can
        be picked up before the games, and one cut on Friday cannot be picked up
        by anybody until the following week. Calling this is optional; `step`
        runs whatever phases are left before playing the week.
        """
        if self.done:
            raise RuntimeError("season is over; call reset()")
        week = self.week
        history = self._history_before(week)
        made: dict[int, list] = {}
        for team in self._waiver_order():
            moves = []
            if team == self.agent_team:
                # The agent's own claims are applied at its turn in the queue,
                # not before everybody else's. Applying them on submission gave
                # the agent first pick of the wire every phase of every season,
                # which is the same systematic edge the housekeeping order had.
                moves.extend(self._drain_claims())
            moves.extend(self._manage(team, history, week))
            made[team] = moves
        for team, moves in made.items():
            self._week_moves.setdefault(team, []).extend(moves)
        self._rotate_priority(
            {
                team
                for team, moves in made.items()
                if any(m.kind == "waiver-add" for m in moves)
            }
        )
        self.phase += 1
        self._claims_this_phase = 0
        return made

    def _drain_claims(self) -> list:
        """Apply the agent's queued claims, in the order it made them.

        A claim that is no longer possible is dropped rather than raised on: by
        the time the queue reaches the agent a higher-priority team may have
        taken the player, and losing him is the outcome the priority system
        exists to produce.
        """
        moves = []
        queued, self._claim_queue = self._claim_queue, []
        for claim in queued:
            roster = self.rosters[self.agent_team]
            if claim.drop_key not in roster:
                continue
            if claim.add_key not in self.free_agents:
                continue
            if not self.wire.is_free(claim.add_key, self.hour):
                continue
            self._swap(self.agent_team, claim)
            # Recorded only once it has actually landed. A claim that lost the
            # player to a higher-priority team did not happen, and grading it as
            # though it had would credit the agent for a swap it never made.
            self._applied_claims.append(claim)
            moves.append(
                Transaction(self.week, "waiver-add", claim.add_key, hour=self.hour)
            )
            moves.append(
                Transaction(self.week, "drop", claim.drop_key, hour=self.hour)
            )
        return moves

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
            wire=self.wire,
            hour=self.hour,
        )

    @property
    def hour(self) -> int:
        """Hours from the season's start, at the current transaction phase."""
        return hour_of(self.week_index, PHASES[min(self.phase, len(PHASES) - 1)])

    def _apply_claim(self, team: int, claim: WaiverClaim) -> None:
        """Queue the agent's claim; other teams transact directly."""
        limit = self.config.waiver_adds_per_phase
        if team == self.agent_team:
            if limit is not None and self._claims_this_phase >= limit:
                raise ValueError(
                    f"already made {self._claims_this_phase} claim(s) at this "
                    f"transaction phase; the limit is {limit}"
                )
            # Checked now so an impossible claim is a bug the caller hears
            # about, and checked again when the queue reaches the agent, where
            # failing is a legitimate outcome rather than an error.
            #
            # Against the roster the queue *will* produce, not the one standing
            # now: with no cap on adds a policy can make several claims in one
            # phase, and the second is naturally about the roster the first
            # leaves behind.
            roster, pool = self._projected(team)
            if claim.add_key not in pool:
                raise ValueError(f"{claim.add_key} is not a free agent")
            if not self.wire.is_free(claim.add_key, self.hour):
                raise ValueError(
                    f"{claim.add_key} is on waivers until hour "
                    f"{self.wire.clears_at(claim.add_key)}; it is {self.hour}"
                )
            if claim.drop_key not in roster:
                raise ValueError(f"{claim.drop_key} is not on team {team}")
            self._claims_this_phase += 1
            self._claim_queue.append(claim)
            return
        self._swap(team, claim)

    def _projected(self, team: int) -> tuple[set[str], set[str]]:
        """The roster and free-agent pool once the queued claims have landed."""
        roster = set(self.rosters[team])
        pool = set(self.free_agents)
        for queued in self._claim_queue:
            if queued.drop_key in roster:
                roster.discard(queued.drop_key)
                pool.add(queued.drop_key)
            if queued.add_key in pool:
                pool.discard(queued.add_key)
                roster.add(queued.add_key)
        return roster, pool

    def _swap(self, team: int, claim: WaiverClaim) -> None:
        """Move one player onto a roster and one off it, updating the wire."""
        roster = self.rosters[team]
        roster.remove(claim.drop_key)
        roster.append(claim.add_key)
        self.free_agents.remove(claim.add_key)
        self.free_agents.append(claim.drop_key)
        # Order matters: the player joining is recorded as added before the one
        # leaving is dropped, so a swap made and unmade inside the same phase
        # reads as the 24-hour undo it is rather than as two waiver events.
        self.wire.added(claim.add_key, self.hour)
        self.wire.dropped(claim.drop_key, self.hour)

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
    env.agent_policy = policy
    while not env.done:
        # One shot per transaction phase, because the pool is different at each:
        # a player cut on Wednesday clears waivers by Friday, so a claim that was
        # impossible at the first phase can be the obvious move at the second.
        while env.phase < len(PHASES):
            if waiver_policy is not None:
                claims = waiver_policy(env, observation)
                if claims is not None:
                    if isinstance(claims, WaiverClaim):
                        claims = [claims]
                    for claim in claims:
                        observation = env.submit_claim(claim)
            # Every phase transacts here, so the lineup is set on the roster the
            # transactions actually produced rather than on the one the agent
            # asked for and may not have got.
            env.transact()
            observation = env.observe()
        scores = policy.score(
            observation["roster"],
            observation["history"],
            observation["week"],
            observation["board"],
        )
        observation, _reward, _done, _info = env.step(scores)
    return env.result
