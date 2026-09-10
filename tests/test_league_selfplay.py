"""Opponents that transact: the hook, the rules that bind it, and the mirror."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.league.agent import PARAMETER_COUNT, LinearAgent, Scaler, as_opponent_claims
from ffmodel.league.config import LeagueConfig
from ffmodel.league.env import FantasyLeagueEnv, WaiverClaim, run_episode
from ffmodel.league.features import FEATURE_COLUMNS, build_feature_tables
from ffmodel.league.policies import SeasonPolicy
from ffmodel.league.pool import build_player_pool

SEASON = 2024


@pytest.fixture(scope="module")
def league():
    pool = build_player_pool([SEASON])
    tables = build_feature_tables(pool, [SEASON])
    return pool, tables


def _environment(league, claims=None, seed=0):
    pool, tables = league
    return FantasyLeagueEnv(
        pool, season=SEASON, config=LeagueConfig(), seed=seed,
        opponent=SeasonPolicy(table=tables[SEASON]),
        opponent_claims=claims,
    )


def test_the_standard_field_is_unchanged_by_the_hook(league):
    """No hook is the old environment exactly, not merely close to it."""
    pool, tables = league
    policy = SeasonPolicy(table=tables[SEASON])
    plain = run_episode(_environment(league), policy)
    explicit = run_episode(_environment(league, claims=None), policy)
    assert plain.wins == explicit.wins
    assert plain.total_points == pytest.approx(explicit.total_points)


def test_a_declining_hook_changes_nothing(league):
    """A field that is asked and always says no is the standard field."""
    pool, tables = league
    policy = SeasonPolicy(table=tables[SEASON])
    plain = run_episode(_environment(league), policy)
    asked = run_episode(
        _environment(league, claims=lambda env, team, **kw: None), policy
    )
    assert plain.wins == asked.wins
    assert plain.total_points == pytest.approx(asked.total_points)


def test_opponents_actually_move_players(league):
    """The hook has to reach the rosters, or the field is not competing."""
    pool, tables = league
    seen = []

    def grab_first(env, team, *, roster, free_agents, week):
        if not free_agents or not roster:
            return None
        claim = WaiverClaim(add_key=free_agents[0], drop_key=roster[-1])
        seen.append((team, claim.add_key))
        return claim

    env = _environment(league, claims=grab_first)
    before = {team: list(keys) for team, keys in env.rosters.items()}
    run_episode(env, SeasonPolicy(table=tables[SEASON]))
    assert seen, "the hook was never consulted"
    moved = [t for t in before if t != env.agent_team and set(before[t]) != set(env.rosters[t])]
    assert moved, "no opponent roster changed"


def test_an_opponent_cannot_take_a_player_off_waivers(league):
    """The 48-hour lock binds the field exactly as it binds the agent."""
    pool, tables = league
    env = _environment(league, claims=lambda env, team, **kw: None)
    env.reset()
    # Cut somebody, which puts him on the wire, then try to claim him at once.
    victim = env.rosters[1][-1]
    env._swap(1, WaiverClaim(add_key=env.free_agents[0], drop_key=victim))
    assert not env.wire.is_free(victim, env.hour + 1)

    taken = []

    def take_victim(e, team, *, roster, free_agents, week):
        # Asked for regardless of whether he is offered: the point is that the
        # environment refuses him, not that the policy declines to want him.
        if team == 2:
            taken.append(victim)
            return WaiverClaim(add_key=victim, drop_key=roster[-1])
        return None

    env.opponent_claims = take_victim
    env.transact()
    assert victim not in env.rosters[2]


def test_an_opponents_claim_survives_its_own_housekeeping(league):
    """Claim and cut in the same breath is a bug, not a rule.

    The agent's pending claim has always been protected. An opponent's had to
    be too, or every seat but the agent's would be strictly worse for a reason
    that has nothing to do with the policy being measured.
    """
    pool, tables = league
    env = _environment(league, claims=lambda env, team, **kw: None)
    env.reset()
    # From the shortlist the environment actually offers, not the raw pool.
    target = env._waiver_shortlist(env._history_before(env.week))[0]

    def claim_once(e, team, *, roster, free_agents, week):
        if team == 3 and target in free_agents:
            return WaiverClaim(add_key=target, drop_key=roster[-1])
        return None

    env.opponent_claims = claim_once
    env.transact()
    assert target in env.rosters[3]


def test_as_opponent_claims_runs_the_agents_own_rule(league):
    """Self-play must use the same decision, not a re-implementation of it."""
    pool, tables = league
    theta = np.zeros(PARAMETER_COUNT)
    theta[FEATURE_COLUMNS.index("ewma2")] = 1.0
    theta[-1] = 0.5
    scaler = Scaler.fit(tables)
    agent = LinearAgent.from_parameters(theta, tables[SEASON], scaler)

    env = _environment(league, claims=as_opponent_claims(agent))
    result = run_episode(env, SeasonPolicy(table=tables[SEASON]))
    assert 0 <= result.wins <= len(LeagueConfig().weeks)
