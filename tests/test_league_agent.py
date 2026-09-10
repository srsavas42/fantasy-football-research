"""The learned agent, and the shortcut that makes training it possible.

Two things here are load-bearing and neither is obvious from reading the code.

The feature table is a shortcut around the environment's one hard rule -- it
answers "what had this player averaged" without going through the truncated
history frame that makes leakage impossible. So the first thing tested is that
the shortcut lands in exactly the same place, which is checkable to the float
rather than arguable.

The second is that the agent is a policy like any other: it ranks its own
players, the environment assigns the lineup, and nothing it is handed contains a
week it has not played.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.league.agent import (
    PARAMETER_COUNT,
    LinearAgent,
    Scaler,
    as_waiver_policy,
)
from ffmodel.league.config import LeagueConfig
from ffmodel.league.env import FantasyLeagueEnv, run_episode
from ffmodel.league.features import (
    FEATURE_COLUMNS,
    as_matrix,
    build_feature_table,
    build_feature_tables,
    verify_against_history,
)
from ffmodel.league.policies import EwmaPolicy, SeasonPolicy
from ffmodel.league.train import Arena, CrossEntropyTrainer, Task
from tests.test_league_env import _pool


def test_the_feature_table_says_what_reading_the_history_frame_says():
    """The shortcut has to be exact, not close.

    Every average in the table is a number the environment would otherwise have
    derived from a frame truncated to weeks already played. If the two ever
    disagree, the table is either leaking a week or losing one, and both are
    silent failures that would show up only as an agent that cannot be
    reproduced.
    """
    pool = _pool(weeks=10, stars=2)
    for halflife in (1.0, 2.0, 4.0):
        worst = verify_against_history(pool, 2024, halflife)
        assert worst == pytest.approx(0.0, abs=1e-12), (
            f"half-life {halflife} disagrees by {worst}"
        )


def test_a_player_keeps_his_average_through_a_bye():
    """The lag has to happen after the grid is filled, not before.

    A club's bye leaves the player without a row. Lagging first and filling
    afterwards carries "everything before week 6" into week 7, when the truth for
    week 7 is "everything up to week 6" -- so the player looks worse for a week
    after every gap, and the roster mechanic cuts him for it.
    """
    pool = _pool(weeks=10)
    table = build_feature_table(pool, 2024)
    # T1's bye is week 3, so week 4 must reflect weeks 1-2, not week 1 alone.
    key = pool[pool["team"] == "T1"]["player_key"].iloc[0]
    played = pool[(pool["player_key"] == key) & (pool["week"] < 4)]
    assert len(played) == 2, "fixture assumption: two weeks played before week 4"

    row = table.loc[(key, 4)]
    assert row["experience"] == pytest.approx(np.log1p(2))
    assert row["season_mean"] == pytest.approx(played["points"].mean())
    # The bye week itself carries his history rather than a hole -- and it is
    # the *same* history week 4 sees, because no football happened in between.
    assert table.loc[(key, 3), "experience"] == pytest.approx(np.log1p(2))
    assert table.loc[(key, 3), "season_mean"] == pytest.approx(played["points"].mean())


def test_the_table_never_contains_the_week_it_describes():
    """Row `w` is built from weeks strictly before `w`. The whole rule."""
    pool = _pool(weeks=10)
    table = build_feature_table(pool, 2024)
    points = pool.set_index(["player_key", "week"])["points"]
    # A player whose first week is a big score: his own week-1 row must not
    # know about it, and his week-2 row must.
    key = pool["player_key"].iloc[0]
    assert table.loc[(key, 1), "last_points"] == 0.0
    assert table.loc[(key, 2), "last_points"] == pytest.approx(points.loc[(key, 1)])


def test_the_fast_policy_and_the_slow_one_agree_exactly():
    """Same numbers with the table or without it, or the shortcut is a fork."""
    pool = _pool(weeks=10, stars=2)
    table = build_feature_table(pool, 2024)
    block = pool[pool["season"] == 2024]
    keys = sorted(block["player_key"].unique())
    board = block.drop_duplicates("player_key")[["player_key", "adp_rank"]]

    for week in sorted(block["week"].unique()):
        history = block[block["week"] < week]
        slow = EwmaPolicy().score(keys, history, int(week), board)
        fast = EwmaPolicy(table=table).score(keys, history, int(week), board)
        for key in keys:
            assert slow[key] == pytest.approx(fast[key], abs=1e-12), (
                f"{key} disagrees in week {week}"
            )


def test_a_table_policy_refuses_a_history_mode_it_does_not_hold():
    """The table carries one reading of history. Silently serving another is worse
    than refusing, because the numbers would look perfectly reasonable."""
    pool = _pool(weeks=6)
    table = build_feature_table(pool, 2024)
    policy = EwmaPolicy(table=table, history_mode="active")
    with pytest.raises(ValueError, match="history mode"):
        policy.score(["QB0"], pd.DataFrame(), 3, pd.DataFrame())


def test_the_agent_ranks_by_its_weights():
    """A weight on one feature has to order players by that feature."""
    pool = _pool(weeks=8)
    table = build_feature_table(pool, 2024)
    scaler = Scaler.fit({2024: table})
    theta = np.zeros(PARAMETER_COUNT)
    theta[FEATURE_COLUMNS.index("ewma2")] = 1.0
    agent = LinearAgent.from_parameters(theta, table, scaler)

    keys = [f"RB{index}" for index in range(6)]
    scores = agent.score(keys, pd.DataFrame(), 5, pd.DataFrame())
    averages = table.loc[[(key, 5) for key in keys], "ewma2"]
    assert [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])] == [
        key for key, _ in sorted(zip(keys, averages), key=lambda kv: -kv[1])
    ]


def test_a_wrong_sized_parameter_vector_is_refused():
    pool = _pool(weeks=6)
    table = build_feature_table(pool, 2024)
    scaler = Scaler.fit({2024: table})
    with pytest.raises(ValueError, match="parameters"):
        LinearAgent.from_parameters(np.zeros(PARAMETER_COUNT - 1), table, scaler)


def test_the_agent_never_cuts_a_player_it_would_start():
    """A threshold must not be able to authorise trading away a starter."""
    pool = _pool(weeks=8, stars=3)
    table = build_feature_table(pool, 2024)
    scaler = Scaler.fit({2024: table})
    theta = np.zeros(PARAMETER_COUNT)
    theta[FEATURE_COLUMNS.index("ewma2")] = 1.0
    theta[-1] = -1e6  # claim at every opportunity
    agent = LinearAgent.from_parameters(theta, table, scaler)

    config = LeagueConfig(teams=12, first_week=1, last_week=6)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=1)
    observation = env.reset()
    while not env.done:
        claim = agent.claim(env, observation)
        if claim is not None:
            from ffmodel.league.lineup import optimal_lineup

            values = agent.values(observation["roster"], observation["week"])
            lineup = optimal_lineup(
                observation["roster"], env.positions, values, env.config.slots
            )
            assert claim.drop_key not in lineup.starting_keys(), (
                f"dropped {claim.drop_key}, who it would have started"
            )
            observation = env.submit_claim(claim)
        scores = agent.score(
            observation["roster"], observation["history"], observation["week"], observation["board"]
        )
        observation, _, _, _ = env.step(scores)


def test_the_agent_plays_a_whole_season_without_leaking_a_future_week():
    """The same guarantee every other policy gets, checked on the learned one."""
    pool = _pool(weeks=10, stars=2)
    table = build_feature_table(pool, 2024)
    scaler = Scaler.fit({2024: table})
    theta = np.zeros(PARAMETER_COUNT)
    theta[FEATURE_COLUMNS.index("ewma1")] = 1.0
    agent = LinearAgent.from_parameters(theta, table, scaler)

    config = LeagueConfig(teams=12, first_week=1, last_week=8)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=2)
    observation = env.reset()
    while not env.done:
        history = observation["history"]
        if len(history):
            assert history["week"].max() < observation["week"]
        claim = agent.claim(env, observation)
        if claim is not None:
            observation = env.submit_claim(claim)
        scores = agent.score(
            observation["roster"], observation["history"], observation["week"], observation["board"]
        )
        observation, _, _, _ = env.step(scores)
    assert env.result.wins >= 0


def test_missing_feature_rows_come_back_as_zeros_not_as_a_crash():
    pool = _pool(weeks=6)
    table = build_feature_table(pool, 2024)
    values = as_matrix(table, ["QB0", "nobody-at-all"], 3)
    assert values.shape == (2, len(FEATURE_COLUMNS))
    assert np.all(values[1] == 0.0)


# ------------------------------------------------------------------ training


def test_the_search_improves_a_deliberately_bad_starting_point():
    """The one property a trainer has to have: it goes uphill.

    Started at zeros -- which ranks every player identically and is measurably
    terrible -- a handful of generations must beat where they began, scored on
    the same seats so the comparison is not a draw of the dice.
    """
    pool = _pool(weeks=8, stars=2)
    tables = build_feature_tables(pool, [2024])
    scaler = Scaler.fit(tables)
    arena = Arena(pool=pool, tables=tables, config=LeagueConfig(first_week=1, last_week=8))

    trainer = CrossEntropyTrainer(
        arena, scaler, seasons=[2024], seeds=6, population=10, batch=4,
        sigma=0.6, rng=np.random.default_rng(0), workers=1,
    )
    tasks = [Task(2024, seed) for seed in range(6)]
    before = trainer.fitness(np.zeros((1, PARAMETER_COUNT)), tasks)[0]
    theta = trainer.run(4, log=lambda *_: None)
    after = trainer.fitness(theta[None, :], tasks)[0]
    assert after > before, f"search went downhill: {before:.1f} -> {after:.1f}"


def test_the_control_variate_is_paired_and_cached():
    """Fitness is measured against the same seat, not against an average.

    If the baseline were a constant, a candidate that drew easy seeds would
    outscore a better one that drew hard ones, and the search would be fitting
    the draw.
    """
    pool = _pool(weeks=8)
    tables = build_feature_tables(pool, [2024])
    arena = Arena(pool=pool, tables=tables, config=LeagueConfig(first_week=1, last_week=8))
    trainer = CrossEntropyTrainer(
        arena, Scaler.fit(tables), seasons=[2024], seeds=4, workers=1
    )
    tasks = [Task(2024, seed) for seed in range(4)]
    first = trainer.baselines(tasks)
    assert len(set(first)) > 1, "every seat scoring the same is not a seat"
    # Cached, and identical on a second call -- the control variate must not
    # itself be a source of noise between generations.
    assert np.array_equal(first, trainer.baselines(tasks))


def test_a_saved_agent_reloads_to_the_same_policy(tmp_path):
    from ffmodel.league.train import load_agent, save_agent

    pool = _pool(weeks=6)
    table = build_feature_table(pool, 2024)
    scaler = Scaler.fit({2024: table})
    theta = np.linspace(-1.0, 1.0, PARAMETER_COUNT)
    path = tmp_path / "agent.json"
    save_agent(path, theta, scaler, {"note": "test"})

    reloaded = load_agent(path, table)
    assert np.allclose(reloaded.parameters(), theta)
    keys = ["QB0", "RB0", "WR0"]
    original = LinearAgent.from_parameters(theta, table, scaler)
    assert reloaded.values(keys, 3) == original.values(keys, 3)


# --------------------------------------------------------------- projections


def _projections(pool, season=2024):
    """A projection cache shaped like the real one, valued so it is testable."""
    block = pool[pool["season"] == season]
    rows = block[["season", "week", "player_key"]].drop_duplicates().copy()
    # Deliberately a *different* number from anything derivable from history, so
    # a table that quietly ignored the cache would be caught.
    rows["projection"] = 100.0 + rows["week"]
    rows["ros_projection"] = 1000.0 + rows["week"]
    return rows


def test_a_projection_is_not_lagged_the_way_a_running_statistic_is():
    """The distinction the whole join turns on.

    Every other column is a statistic computed through week `w` and shifted,
    because otherwise it contains the week it decides. A projection is already a
    statement about week `w` made without week `w`. Lagging it would hand the
    agent last week's projection to set this week's lineup -- which is not
    caution, it is a bug that looks like the model being useless.
    """
    pool = _pool(weeks=8)
    table = build_feature_table(pool, 2024, _projections(pool))
    # A player whose club's bye avoids the weeks checked: T2 sits out week 4,
    # and a bye week is legitimately zero, which the next test covers.
    key = pool[pool["team"] == "T2"]["player_key"].iloc[0]
    for week in (1, 2, 5):
        assert table.loc[(key, week), "projection"] == pytest.approx(100.0 + week)


def test_the_projection_survives_a_week_the_player_does_not_play():
    """A bye is worth nothing on Sunday and does not touch the rest of the year.

    So the two horizons fill differently across the gap, and the difference is
    the reason rest-of-season exists as its own feature: a waiver decision about
    a player idle this week is still a decision about the eight weeks after it.
    """
    pool = _pool(weeks=8)
    table = build_feature_table(pool, 2024, _projections(pool))
    key = pool[pool["team"] == "T1"]["player_key"].iloc[0]  # bye in week 3
    assert (key, 3) not in set(zip(pool["player_key"], pool["week"])), (
        "fixture assumption: a club on bye contributes no row"
    )
    assert table.loc[(key, 3), "projection"] == 0.0, "he cannot score on a bye"
    assert table.loc[(key, 3), "ros_projection"] == pytest.approx(1000.0 + 2), (
        "his rest-of-season value should carry across the bye"
    )


def test_without_a_projection_cache_the_features_are_zero_not_missing():
    """The agent has to remain runnable before the cache is built."""
    pool = _pool(weeks=6)
    table = build_feature_table(pool, 2024, None)
    assert (table["projection"] == 0.0).all()
    assert (table["ros_projection"] == 0.0).all()
    assert not table.isna().any().any()


def test_an_agent_weighting_the_projection_ranks_by_it():
    pool = _pool(weeks=8)
    projections = _projections(pool)
    # Give one player a standout projection and check he rises. RB1's club takes
    # its bye in week 3, so week 5 is an ordinary week for him.
    star = "RB1"
    projections.loc[projections["player_key"] == star, "projection"] = 999.0
    table = build_feature_table(pool, 2024, projections)
    scaler = Scaler.fit({2024: table})
    theta = np.zeros(PARAMETER_COUNT)
    theta[FEATURE_COLUMNS.index("projection")] = 1.0
    agent = LinearAgent.from_parameters(theta, table, scaler)

    keys = [f"RB{index}" for index in range(6)]
    scores = agent.score(keys, pd.DataFrame(), 5, pd.DataFrame())
    assert max(scores, key=scores.get) == star


def test_the_search_can_hold_a_feature_at_zero():
    """How an ablation is run inside the search rather than after it.

    Both arms then differ by the feature alone, not by which seasons or seeds
    each happened to draw.
    """
    pool = _pool(weeks=8)
    tables = build_feature_tables(pool, [2024], _projections(pool))
    scaler = Scaler.fit(tables)
    arena = Arena(pool=pool, tables=tables, config=LeagueConfig(first_week=1, last_week=8))

    mask = np.ones(PARAMETER_COUNT, bool)
    held = [FEATURE_COLUMNS.index("projection"), FEATURE_COLUMNS.index("ros_projection")]
    mask[held] = False
    trainer = CrossEntropyTrainer(
        arena, scaler, seasons=[2024], seeds=4, population=6, batch=2,
        rng=np.random.default_rng(0), workers=1, mask=mask,
    )
    theta = trainer.run(3, log=lambda *_: None)
    assert np.allclose(theta[held], 0.0), theta[held]
    assert not np.allclose(np.delete(theta, held), 0.0), "the search moved nothing"


# ------------------------------------------------------ acquisition context


def test_context_reads_the_roster_rather_than_the_player():
    """The whole point: the same player scores differently on different rosters.

    Every other feature is a fact about a player and would be identical here.
    Depth, and the upgrade over the man he displaces, are facts about a team.
    """
    from ffmodel.league.config import LeagueConfig
    from ffmodel.league.context import CONTEXT_COLUMNS, build_context, team_shortfalls

    slots = LeagueConfig().slots
    positions = {"rb1": "RB", "rb2": "RB", "rb3": "RB", "wr1": "WR", "free": "RB"}
    values = {"rb1": 90.0, "rb2": 80.0, "rb3": 20.0, "wr1": 70.0, "free": 60.0}

    deep = ["rb1", "rb2", "rb3", "wr1"]
    thin = ["rb3", "wr1"]
    shortfalls = team_shortfalls({0: deep, 1: thin}, positions, slots)

    def context_for(roster, starters):
        return build_context(
            ["free"], values=values, roster=roster, starters=starters,
            positions=positions, slots=slots, free_agents=["free"],
            shortfalls=shortfalls, agent_team=0,
        )[0]

    depth = CONTEXT_COLUMNS.index("ctx_depth")
    upgrade = CONTEXT_COLUMNS.index("ctx_upgrade")

    rich = context_for(deep, {"rb1", "rb2", "wr1"})
    poor = context_for(thin, {"rb3", "wr1"})

    assert rich[depth] > poor[depth], "a deeper roster should read as deeper"
    # On the thin roster he displaces a 20-point back; on the deep one an
    # 80-point back. Same player, opposite decisions.
    assert poor[upgrade] > rich[upgrade]
    assert poor[upgrade] > 0 > rich[upgrade]


def test_a_replaceable_player_reads_as_replaceable():
    """The wire gap: worth is relative to the next man on the page."""
    from ffmodel.league.config import LeagueConfig
    from ffmodel.league.context import CONTEXT_COLUMNS, build_context, team_shortfalls

    slots = LeagueConfig().slots
    positions = {"mine": "RB", "a": "RB", "b": "RB"}
    roster, starters = ["mine"], {"mine"}
    gap = CONTEXT_COLUMNS.index("ctx_wire_gap")

    def wire_gap(values, free):
        shortfalls = team_shortfalls({0: roster}, positions, slots)
        rows = build_context(
            ["a"], values=values, roster=roster, starters=starters,
            positions=positions, slots=slots, free_agents=free,
            shortfalls=shortfalls, agent_team=0,
        )
        return rows[0][gap]

    unique = wire_gap({"mine": 10.0, "a": 90.0, "b": 10.0}, ["a", "b"])
    crowded = wire_gap({"mine": 10.0, "a": 90.0, "b": 88.0}, ["a", "b"])
    assert unique > crowded, "a player with no equal behind him should stand out"


def test_the_context_block_only_touches_the_acquisition():
    """A lineup has no alternative to weigh; the roster is already what it is."""
    pool = _pool(weeks=8)
    table = build_feature_table(pool, 2024, _projections(pool))
    scaler = Scaler.fit({2024: table})

    from ffmodel.league.agent import parameter_count
    from ffmodel.league.context import CONTEXT_COLUMNS

    theta = np.zeros(parameter_count(context=True))
    theta[FEATURE_COLUMNS.index("ewma2")] = 1.0
    plain = LinearAgent.from_parameters(theta[: parameter_count()], table, scaler)
    # Wild context weights, which must not move a single lineup score.
    theta[-len(CONTEXT_COLUMNS) :] = 5.0
    with_context = LinearAgent.from_parameters(theta, table, scaler, context=True)

    keys = [f"RB{index}" for index in range(6)]
    assert plain.score(keys, pd.DataFrame(), 5, pd.DataFrame()) == with_context.score(
        keys, pd.DataFrame(), 5, pd.DataFrame()
    )


def test_the_mlp_carries_its_own_parameter_count_and_ranks():
    from ffmodel.league.agent import MLPAgent, mlp_parameter_count

    pool = _pool(weeks=8)
    table = build_feature_table(pool, 2024, _projections(pool))
    scaler = Scaler.fit({2024: table})

    size = mlp_parameter_count(hidden=4)
    rng = np.random.default_rng(0)
    agent = MLPAgent.from_parameters(rng.normal(0, 0.5, size), table, scaler, hidden=4)

    keys = [f"RB{index}" for index in range(6)]
    scores = agent.score(keys, pd.DataFrame(), 5, pd.DataFrame())
    assert len(scores) == len(keys)
    assert len(set(scores.values())) > 1, "every player scored the same"
    assert np.allclose(agent.parameters().shape, (size,))

    with pytest.raises(ValueError, match="parameters"):
        MLPAgent.from_parameters(np.zeros(size - 1), table, scaler, hidden=4)
