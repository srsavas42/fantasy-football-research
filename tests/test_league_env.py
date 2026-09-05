"""The league environment's load-bearing properties.

Four things have to hold or every number the environment produces is worthless:
it must be fair (a policy playing the field's own strategy finishes average), it
must never show a policy a week that has not happened, it must seat every team a
legal roster, and it must assign lineups optimally rather than greedily.

Built on a synthetic pool rather than the real panels so the suite stays fast
and offline. The shapes are what matter here, not the football.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.league.config import FLEX_POSITIONS, LeagueConfig, RosterSlots
from ffmodel.league.draft import run_draft
from ffmodel.league.env import FantasyLeagueEnv
from ffmodel.league.lineup import optimal_lineup, round_robin
from ffmodel.league.policies import EwmaPolicy


def _pool(seasons=(2024,), weeks=8, seed=0) -> pd.DataFrame:
    """A synthetic league's worth of players, deep enough to draft twelve teams."""
    rng = np.random.default_rng(seed)
    counts = {"QB": 30, "RB": 60, "WR": 70, "TE": 30, "K": 20, "DST": 20}
    rows = []
    for season in seasons:
        rank = 1
        for position, total in counts.items():
            for index in range(total):
                key = f"{position}{index}"
                level = 20.0 - 0.2 * index
                for week in range(1, weeks + 1):
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "team": f"T{index % 32}",
                            "player_key": key,
                            "player_name": f"{position} Player {index}",
                            "position": position,
                            "points": max(0.0, level + rng.normal(0, 3)),
                            "played": 1,
                            "adp_rank": float(rank),
                            "adp_drafted": True,
                        }
                    )
                rank += 1
    return pd.DataFrame(rows)


def test_optimal_lineup_beats_the_greedy_flex_fill():
    """Filling dedicated slots first and flexing the leftovers is not optimal.

    Two backs at 20 and a receiver at 19: greedy seats both backs at RB and
    hands the flex a 5-point receiver. The right answer starts one back, the
    19-point receiver at WR, and flexes the other back.
    """
    slots = RosterSlots(qb=0, rb=1, wr=1, te=0, flex=1, k=0, dst=0, bench=0)
    positions = {"rb1": "RB", "rb2": "RB", "wr1": "WR", "wr2": "WR"}
    scores = {"rb1": 20.0, "rb2": 20.0, "wr1": 19.0, "wr2": 5.0}

    lineup = optimal_lineup(list(positions), positions, scores, slots)
    # Best legal card is 20 + 20 + 19 = 59, using both backs and the good receiver.
    assert lineup.projected == pytest.approx(59.0)
    assert "wr2" not in lineup.starting_keys()


def test_every_team_leaves_the_draft_able_to_field_a_lineup():
    pool = _pool()
    config = LeagueConfig(teams=12, seasons=(2024,))
    result = run_draft(pool, 2024, config, seed=3)

    positions = pool.drop_duplicates("player_key").set_index("player_key")["position"]
    slots = config.slots
    for team, roster in result.rosters.items():
        assert len(roster) == slots.size, f"team {team} roster is short"
        held = positions.reindex(roster).value_counts()
        # Every dedicated starting slot must be fillable, or it sits empty all
        # season -- the failure the draft's forced-pick rule exists to prevent.
        for position, count in slots.dedicated().items():
            assert held.get(position, 0) >= count, (
                f"team {team} cannot fill {count} {position} slot(s)"
            )
        # And the flex needs a body beyond the dedicated ones. This is the part
        # a per-position minimum misses: every individual requirement can be
        # satisfied while the roster is still one player short of a legal card.
        flex_eligible = sum(held.get(position, 0) for position in FLEX_POSITIONS)
        required = sum(slots.dedicated()[p] for p in FLEX_POSITIONS) + slots.flex
        assert flex_eligible >= required, (
            f"team {team} has {flex_eligible} flex-eligible, needs {required}"
        )

    # Nobody is on two rosters.
    everyone = [key for roster in result.rosters.values() for key in roster]
    assert len(everyone) == len(set(everyone))


def test_the_caps_bind_and_the_uncapped_positions_do_not():
    """Two apiece at quarterback, kicker and defense; unlimited elsewhere.

    A third of any of those three cannot be started in the same week as the
    first two, so the cap is what stops the board handing a team dead weight.
    Backs, receivers and tight ends are deliberately uncapped -- how much depth
    to carry at the flex-eligible positions is a decision a policy should own.
    """
    pool = _pool()
    config = LeagueConfig(teams=12, seasons=(2024,))
    result = run_draft(pool, 2024, config, seed=5)
    positions = pool.drop_duplicates("player_key").set_index("player_key")["position"]

    for team, roster in result.rosters.items():
        held = positions.reindex(roster).value_counts()
        for position in ("QB", "K", "DST"):
            assert held.get(position, 0) <= 2, (
                f"team {team} holds {held.get(position)} at {position}"
            )

    # And at least one team should exceed what a cap of two would have allowed
    # somewhere flex-eligible, or "uncapped" is not doing anything.
    depth = max(
        sum(
            positions.reindex(roster).value_counts().get(position, 0)
            for position in ("RB",)
        )
        for roster in result.rosters.values()
    )
    assert depth > 2, "no team stockpiled backs; the uncapped rule is inert"


def test_the_environment_never_shows_a_policy_an_unplayed_week():
    """The one rule the whole package is built on."""
    pool = _pool(weeks=8)
    config = LeagueConfig(teams=12, first_week=1, last_week=6)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=1)

    seen_weeks = []
    while not env.done:
        observation = env.observe()
        history = observation["history"]
        if len(history):
            # Strictly before: a policy deciding week w must not see week w.
            assert history["week"].max() < observation["week"]
        seen_weeks.append(observation["week"])
        scores = {key: 1.0 for key in observation["roster"]}
        env.step(scores)
    assert seen_weeks == list(config.weeks)


def test_a_policy_playing_the_field_s_own_strategy_finishes_average():
    """Fairness. If the seat itself carried an edge, every result would be one.

    Averaged over seeds, because a single fourteen-week season swings roughly
    three wins either way on luck alone -- which is itself worth knowing, and is
    why policy comparisons in this environment need many episodes.
    """
    pool = _pool(weeks=10)
    config = LeagueConfig(teams=12, first_week=1, last_week=10)
    records = []
    for seed in range(8):
        env = FantasyLeagueEnv(pool, season=2024, config=config, seed=seed)
        policy = EwmaPolicy()
        while not env.done:
            observation = env.observe()
            scores = policy.score(
                observation["roster"],
                observation["history"],
                observation["week"],
                observation["board"],
            )
            env.step(scores)
        records.append(env.result.wins)
    # Ten weeks, so a fair seat averages five.
    assert 3.0 <= float(np.mean(records)) <= 7.0, records


def test_round_robin_gives_each_team_exactly_one_opponent_a_week():
    schedule = round_robin(teams=12, weeks=14, seed=0)
    assert len(schedule) == 14
    for pairs in schedule:
        seats = [team for pair in pairs for team in pair]
        assert len(seats) == len(set(seats)), "a team was scheduled twice"
        assert len(seats) == 12, "a team was left without an opponent"


def test_waiver_claim_moves_a_player_both_ways():
    pool = _pool(weeks=6)
    config = LeagueConfig(teams=12, first_week=1, last_week=4)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=2)

    from ffmodel.league.env import WaiverClaim

    roster = list(env.rosters[env.agent_team])
    add = env.free_agents[0]
    drop = roster[-1]
    env.step({key: 1.0 for key in roster}, WaiverClaim(add_key=add, drop_key=drop))

    assert add in env.rosters[env.agent_team]
    assert drop not in env.rosters[env.agent_team]
    assert drop in env.free_agents
    assert add not in env.free_agents


def test_an_unknown_player_is_benched_rather_than_crashing():
    """A policy that forgets somebody should lose points, not raise."""
    slots = RosterSlots(qb=1, rb=0, wr=0, te=0, flex=0, k=0, dst=0, bench=1)
    positions = {"qb1": "QB", "qb2": "QB"}
    lineup = optimal_lineup(list(positions), positions, {"qb1": 5.0}, slots)
    assert lineup.starters["QB"] == ["qb1"]
