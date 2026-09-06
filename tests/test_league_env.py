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

from ffmodel.league.availability import ACTIVE, BYE, build_availability
from ffmodel.league.config import FLEX_POSITIONS, LeagueConfig, RosterSlots
from ffmodel.league.draft import run_draft
from ffmodel.league.credit import grade_claims
from ffmodel.league.env import FantasyLeagueEnv, WaiverClaim, run_episode
from ffmodel.league.lineup import optimal_lineup, round_robin
from ffmodel.league.policies import EwmaPolicy
from ffmodel.league.roster import manage_roster


# Club T{n} takes its bye in week 2 + n % 4, so every fixture wide enough to
# have four clubs exercises the bye path and no two clubs share one.
def _bye_for(club: str, weeks: int) -> int | None:
    week = 2 + int(club[1:]) % 4
    return week if week <= weeks else None


def _pool(seasons=(2024,), weeks=8, seed=0, byes=True, outs=(), stars=0) -> pd.DataFrame:
    """A synthetic league's worth of players, deep enough to draft twelve teams.

    ``byes`` drops each club's bye week, the way the real panel does -- a club on
    bye contributes no rows at all, which is how the availability layer infers
    the bye in the first place. ``outs`` is an explicit list of
    ``(player_key, week)`` to flag on the game-status report.

    ``stars`` adds that many undrafted players per flex position who outscore
    everybody. Without them the fixture has no waiver wire worth claiming from --
    every free agent is a leftover by construction -- and a test of what a good
    claim is worth has nothing good to claim.
    """
    rng = np.random.default_rng(seed)
    counts = {"QB": 30, "RB": 60, "WR": 70, "TE": 30, "K": 20, "DST": 20}
    ruled_out = {(key, int(week)) for key, week in outs}
    rows = []
    for season in seasons:
        rank = 1
        for position, total in counts.items():
            for index in range(total):
                key = f"{position}{index}"
                club = f"T{index % 32}"
                level = 20.0 - 0.2 * index
                bye = _bye_for(club, weeks) if byes else None
                for week in range(1, weeks + 1):
                    if week == bye:
                        continue
                    out = (key, week) in ruled_out
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "team": club,
                            "player_key": key,
                            "player_name": f"{position} Player {index}",
                            "position": position,
                            "points": 0.0 if out else max(0.0, level + rng.normal(0, 3)),
                            "played": 0 if out else 1,
                            "is_out": int(out),
                            "adp_rank": float(rank),
                            "adp_drafted": True,
                        }
                    )
                rank += 1

        # Breakouts: better than anyone drafted, and ranked far enough down the
        # board that nobody drafts them. This is the player a waiver wire exists
        # to find.
        for position in ("RB", "WR", "TE"):
            for index in range(stars):
                key = f"{position}X{index}"
                club = f"T{(index + 7) % 32}"
                bye = _bye_for(club, weeks) if byes else None
                for week in range(1, weeks + 1):
                    if week == bye:
                        continue
                    rows.append(
                        {
                            "season": season,
                            "week": week,
                            "team": club,
                            "player_key": key,
                            "player_name": f"{position} Breakout {index}",
                            "position": position,
                            "points": max(0.0, 30.0 + rng.normal(0, 2)),
                            "played": 1,
                            "is_out": 0,
                            "adp_rank": float(900 + index),
                            "adp_drafted": True,
                        }
                    )
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


# --------------------------------------------------------------- availability


def test_a_bye_is_inferred_from_the_week_a_club_is_absent():
    pool = _pool(weeks=8)
    availability = build_availability(pool, 2024)
    # T1's bye is week 3 under the fixture's rule; T2's is week 4.
    on_t1 = pool[pool["team"] == "T1"]["player_key"].iloc[0]
    assert availability.status(on_t1, 3) == BYE
    assert availability.status(on_t1, 2) == ACTIVE
    assert availability.status(on_t1, 4) == ACTIVE
    assert not availability.is_out(on_t1, 3), "a bye is not an injury"


def test_a_second_idle_week_is_accepted_because_one_really_happens():
    """Buffalo and Cincinnati are each idle twice in 2022.

    Their week 17 game was abandoned and never replayed, so neither club's
    players could score -- which is the only thing a lineup decision asks. An
    earlier version refused to answer whenever a club was absent twice, which
    turned a real event into a crash.
    """
    pool = _pool(weeks=8)
    cancelled = pool[~((pool["team"] == "T1") & (pool["week"] == 6))]
    availability = build_availability(cancelled, 2024)
    on_t1 = pool[pool["team"] == "T1"]["player_key"].iloc[0]
    assert availability.status(on_t1, 3) == BYE, "the real bye"
    assert availability.status(on_t1, 6) == BYE, "the game that was not played"
    assert availability.status(on_t1, 5) == ACTIVE


def test_a_club_idle_more_often_than_a_schedule_allows_still_raises():
    """Three missing weeks is a broken panel, not a season."""
    pool = _pool(weeks=10)
    broken = pool[~((pool["team"] == "T1") & (pool["week"].isin([5, 6, 7])))]
    with pytest.raises(ValueError, match="played no game in 4 weeks"):
        build_availability(broken, 2024)


def test_a_club_with_no_missing_week_simply_has_no_bye():
    """Silence is the safe failure: nobody gets marked absent."""
    pool = _pool(weeks=8, byes=False)
    availability = build_availability(pool, 2024)
    key = pool["player_key"].iloc[0]
    assert all(availability.status(key, week) == ACTIVE for week in range(1, 9))


def test_the_environment_benches_a_player_who_cannot_play():
    """The one thing the whole availability layer exists to do."""
    pool = _pool(weeks=8)
    config = LeagueConfig(teams=12, first_week=1, last_week=8)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=0)

    present = set(zip(pool["player_key"], pool["week"]))
    policy = EwmaPolicy()
    started_on_bye = 0
    while not env.done:
        observation = env.observe()
        scores = policy.score(
            observation["roster"],
            observation["history"],
            observation["week"],
            observation["board"],
        )
        _, _, _, info = env.step(scores)
        for key in info["lineup"].starting_keys():
            if (key, info["week"]) not in present:
                started_on_bye += 1
    # Not zero by construction -- a roster with nobody else at the position has
    # to start somebody -- but it must be rare, and it was 6.3% of slots before
    # the availability layer existed.
    assert started_on_bye <= 2, f"{started_on_bye} starters were on bye"


# ------------------------------------------------------------------- roster


def _manager_case(out_key, out_week, roster, free_agents, slots):
    """A hand-built one-team scenario for the roster mechanic."""
    pool = _pool(weeks=8, outs=[(out_key, out_week)])
    availability = build_availability(pool, 2024)
    positions = pool.drop_duplicates("player_key").set_index("player_key")["position"]
    values = {key: 100.0 - index for index, key in enumerate(roster + free_agents)}
    return (
        availability,
        positions.to_dict(),
        lambda keys: {key: values.get(key, 0.0) for key in keys},
        slots,
    )


def test_an_absence_the_bench_can_cover_costs_no_roster_move():
    """The common case, and the reason the mechanic is cheap.

    A team with a second quarterback simply starts him. Sending the injured one
    to IR and claiming a replacement would be strictly worse -- it spends a
    roster spot to solve a problem the bench already solved.
    """
    slots = RosterSlots(qb=1, rb=0, wr=0, te=0, flex=0, k=0, dst=0, bench=1, ir=1)
    # QB4's club takes its bye in week 2, so he is genuinely available in week 3.
    # QB1's is in week 3 itself, which would make this a two-absence week and
    # test something else entirely.
    roster, free = ["QB0", "QB4"], ["QB2"]
    availability, positions, valuation, slots = _manager_case(
        "QB0", 3, roster, free, slots
    )
    assert availability.status("QB4", 3) == ACTIVE
    moves = manage_roster(
        roster=roster, ir=[], free_agents=free, positions=positions,
        availability=availability, week=3, slots=slots, valuation=valuation,
    )
    assert moves == [], f"unnecessary moves: {[str(m) for m in moves]}"
    assert roster == ["QB0", "QB4"]


def test_an_absence_the_bench_cannot_cover_sends_the_injured_player_to_ir():
    """The case IR exists for: the slot would otherwise sit empty."""
    slots = RosterSlots(qb=1, rb=0, wr=0, te=0, flex=0, k=0, dst=0, bench=0, ir=1)
    roster, free, ir = ["QB0"], ["QB1", "QB2"], []
    availability, positions, valuation, slots = _manager_case(
        "QB0", 3, roster, free, slots
    )
    moves = manage_roster(
        roster=roster, ir=ir, free_agents=free, positions=positions,
        availability=availability, week=3, slots=slots, valuation=valuation,
    )
    kinds = [move.kind for move in moves]
    assert kinds == ["ir-place", "waiver-add"], kinds
    assert ir == ["QB0"], "the injured player should be kept, not cut"
    assert len(roster) == 1 and roster[0] != "QB0"
    assert roster[0] not in free, "the claimed player is still on the wire"


def test_a_bye_never_sends_anybody_to_injured_reserve():
    """IR is for injuries. A bye is one week and resolves itself."""
    slots = RosterSlots(qb=1, rb=0, wr=0, te=0, flex=0, k=0, dst=0, bench=0, ir=1)
    pool = _pool(weeks=8)
    availability = build_availability(pool, 2024)
    positions = (
        pool.drop_duplicates("player_key").set_index("player_key")["position"].to_dict()
    )
    # QB1 is on club T1, whose bye is week 3.
    assert availability.status("QB1", 3) == BYE
    roster, ir, free = ["QB1"], [], ["QB0", "QB2"]
    moves = manage_roster(
        roster=roster, ir=ir, free_agents=free, positions=positions,
        availability=availability, week=3, slots=slots,
        valuation=lambda keys: {key: 1.0 for key in keys},
    )
    assert ir == [], "a bye must not reach IR"
    assert "ir-place" not in [move.kind for move in moves]


def test_a_returning_player_costs_the_worst_bench_spot():
    """Coming off IR is not free -- somebody has to be cut to make room."""
    slots = RosterSlots(qb=1, rb=0, wr=0, te=0, flex=0, k=0, dst=0, bench=1, ir=1)
    pool = _pool(weeks=8, outs=[("QB0", 3)])
    availability = build_availability(pool, 2024)
    positions = (
        pool.drop_duplicates("player_key").set_index("player_key")["position"].to_dict()
    )
    # Week 4: QB0 is no longer out, and the active roster is full.
    roster, ir, free = ["QB2", "QB3"], ["QB0"], []
    values = {"QB0": 100.0, "QB2": 50.0, "QB3": 5.0}
    moves = manage_roster(
        roster=roster, ir=ir, free_agents=free, positions=positions,
        availability=availability, week=4, slots=slots,
        valuation=lambda keys: {key: values.get(key, 0.0) for key in keys},
    )
    assert [move.kind for move in moves] == ["drop", "ir-activate"]
    assert ir == [] and "QB0" in roster
    assert "QB3" not in roster, "the worst bench player should have been cut"
    assert "QB2" in roster, "the better player was cut instead"


def test_the_manager_does_not_cut_the_player_it_just_added():
    """Covering two holes in one week must not churn the same roster spot."""
    slots = RosterSlots(qb=1, rb=1, wr=0, te=0, flex=0, k=0, dst=0, bench=0, ir=0)
    pool = _pool(weeks=8)
    availability = build_availability(pool, 2024)
    positions = (
        pool.drop_duplicates("player_key").set_index("player_key")["position"].to_dict()
    )
    # Both starters are on T1, whose bye is week 3, so both slots open at once.
    roster = ["QB1", "RB1"]
    assert {availability.status(key, 3) for key in roster} == {BYE}
    free = ["QB0", "RB0"]
    moves = manage_roster(
        roster=roster, ir=[], free_agents=free, positions=positions,
        availability=availability, week=3, slots=slots,
        valuation=lambda keys: {key: 1.0 for key in keys},
    )
    added = [move.player_key for move in moves if move.kind == "waiver-add"]
    dropped = [move.player_key for move in moves if move.kind == "drop"]
    assert not set(added) & set(dropped), f"churned: added {added}, dropped {dropped}"


# ------------------------------------------------------------------ policies


def test_the_ewma_history_modes_count_different_weeks():
    """Three readings of the same history, and they must actually differ.

    Weeks 3 and 4 are both scoreless, but only week 3 was a ruled-out absence.
    ``all`` counts both, ``active`` counts neither, and ``available`` counts the
    one nobody could have seen coming -- which is the distinction the whole
    setting exists to make.
    """
    history = pd.DataFrame(
        {
            "player_key": ["a"] * 4,
            "week": [1, 2, 3, 4],
            "points": [20.0, 20.0, 0.0, 0.0],
            "played": [1, 1, 0, 0],
            "is_out": [0, 0, 1, 0],
        }
    )
    board = pd.DataFrame({"player_key": ["a"], "adp_rank": [1.0]})

    def read(mode):
        return EwmaPolicy(history_mode=mode).score(["a"], history, 5, board)["a"]

    assert read("active") == pytest.approx(20.0)
    assert read("all") < 5.0, "both absences should drag the average down"
    # One scoreless week counted instead of two: between the other readings.
    assert read("all") < read("available") < read("active")


def test_an_unknown_history_mode_is_refused():
    with pytest.raises(ValueError, match="history_mode"):
        EwmaPolicy(history_mode="played-only")


# -------------------------------------------------------------- waiver credit


def _play_with_claim(pool, claim_week, chooser, seed=0, weeks=8):
    """One season with a single scripted waiver claim, graded afterwards."""
    config = LeagueConfig(teams=12, first_week=1, last_week=weeks)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=seed)
    policy = EwmaPolicy()
    observation = env.reset()
    while not env.done:
        if observation["week"] == claim_week:
            observation = env.submit_claim(chooser(env, observation))
        scores = policy.score(
            observation["roster"],
            observation["history"],
            observation["week"],
            observation["board"],
        )
        observation, _, _, _ = env.step(scores)
    return env, grade_claims(env)


def _rest_of_season(env, week):
    block = env.frame[env.frame["week"] >= week]
    return block.groupby("player_key")["points"].sum()


def test_a_good_claim_scores_positive_and_a_bad_one_scores_negative():
    """The sign is the whole signal. Get it wrong and the agent learns backwards."""
    pool = _pool(weeks=8, stars=3)

    def best_swap(env, observation):
        future = _rest_of_season(env, observation["week"])
        add = max(observation["free_agents"], key=lambda key: future.get(key, 0.0))
        drop = min(observation["roster"], key=lambda key: (future.get(key, 0.0), key))
        return WaiverClaim(add_key=add, drop_key=drop)

    def worst_swap(env, observation):
        """Cut the lead back for somebody who will never start.

        Deliberately a back rather than simply the roster's highest scorer: see
        the test below for why those are not the same thing.
        """
        future = _rest_of_season(env, observation["week"])
        backs = [
            key for key in observation["roster"] if env.positions.get(key) == "RB"
        ]
        add = min(env.free_agents, key=lambda key: (future.get(key, 0.0), key))
        drop = max(backs, key=lambda key: future.get(key, 0.0))
        return WaiverClaim(add_key=add, drop_key=drop)

    _, good = _play_with_claim(pool, 3, best_swap)
    _, bad = _play_with_claim(pool, 3, worst_swap)
    assert len(good) == len(bad) == 1
    assert good[0].marginal > 0, good[0]
    assert bad[0].marginal < 0, bad[0]


def test_dropping_the_highest_scorer_is_not_a_bad_move_if_the_backup_covers():
    """The property that separates a lineup differential from a points one.

    Cutting the best quarterback on a roster that carries two costs only the gap
    between them, because the backup fills the one quarterback slot. Gross reads
    the whole score as lost and calls it a disaster; the marginal counterfactual
    reads what the lineup actually lost. When the two disagree this sharply, the
    marginal number is the one describing what happened.
    """
    pool = _pool(weeks=8, stars=3)

    def cut_the_best_quarterback(env, observation):
        future = _rest_of_season(env, observation["week"])
        week = observation["week"]
        quarterbacks = [
            key
            for key in observation["roster"]
            if env.positions.get(key) == "QB"
            # Available ones only. Cutting the starter in a week the backup is
            # on bye leaves no quarterback at all, and the housekeeping repairs
            # that by re-claiming the man just dropped -- correct behaviour, and
            # a different scenario than the one under test.
            and env.availability.is_available(key, week)
        ]
        if len(quarterbacks) < 2:
            pytest.skip("no available backup quarterback this week")
        # A worthless add, so the credit reflects the drop and nothing else.
        add = min(env.free_agents, key=lambda key: (future.get(key, 0.0), key))
        drop = max(quarterbacks, key=lambda key: future.get(key, 0.0))
        return WaiverClaim(add_key=add, drop_key=drop)

    _, credits = _play_with_claim(pool, 5, cut_the_best_quarterback)
    credit = credits[0]
    assert credit.weeks_dropped_would_start > 0, "the dropped starter must matter"
    assert credit.gross < 0, "gross counts his whole score as lost"
    # A hundred-point disagreement between the two readings of the same claim.
    assert credit.marginal > credit.gross + 50.0, (
        f"the backup covered most of it: gross {credit.gross:.1f}, "
        f"marginal {credit.marginal:.1f}"
    )


def test_gross_over_credits_relative_to_the_marginal_counterfactual():
    """Why the default is marginal.

    A claimed player who starts displaces somebody. Gross pays the agent his
    whole score; marginal pays only what he added over the man he replaced. The
    gap is the reward for churn that gross would be handing out.
    """
    pool = _pool(weeks=8, stars=3)

    def best_swap(env, observation):
        future = _rest_of_season(env, observation["week"])
        add = max(observation["free_agents"], key=lambda key: future.get(key, 0.0))
        drop = min(observation["roster"], key=lambda key: (future.get(key, 0.0), key))
        return WaiverClaim(add_key=add, drop_key=drop)

    _, credits = _play_with_claim(pool, 3, best_swap)
    credit = credits[0]
    assert credit.weeks_added_started > 0, "this test needs a claim that starts"
    assert credit.gross > credit.marginal


def test_the_credit_never_reaches_the_observation():
    """It is computed from weeks the agent had not played. It is training-only."""
    pool = _pool(weeks=8)
    config = LeagueConfig(teams=12, first_week=1, last_week=6)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=0)
    banned = {"credit", "marginal", "gross", "future", "truth"}
    assert not banned & set(env.observe())

    with pytest.raises(RuntimeError, match="after the season"):
        grade_claims(env)


def test_by_default_there_is_no_cap_on_adds():
    """Leagues do not cap transactions; the roster size is the budget.

    What limits a manager instead is the waiver period, which makes a drop a
    commitment rather than a formality.
    """
    pool = _pool(weeks=6)
    config = LeagueConfig(teams=12, first_week=1, last_week=4)
    assert config.waiver_adds_per_phase is None
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=0)

    for _ in range(4):
        roster = list(env.rosters[env.agent_team])
        add = env.observe()["free_agents"][0]
        # Cut somebody added long enough ago that this is not an undo.
        drop = roster[0]
        env.submit_claim(WaiverClaim(add_key=add, drop_key=drop))
    assert len([k for k in env.rosters[env.agent_team]]) == config.slots.size


def test_a_cap_is_enforced_when_one_is_set():
    pool = _pool(weeks=6)
    config = LeagueConfig(
        teams=12, first_week=1, last_week=4, waiver_adds_per_phase=1
    )
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=0)
    roster = list(env.rosters[env.agent_team])
    env.submit_claim(WaiverClaim(add_key=env.observe()["free_agents"][0], drop_key=roster[-1]))
    with pytest.raises(ValueError, match="the limit is 1"):
        env.submit_claim(
            WaiverClaim(add_key=env.observe()["free_agents"][0], drop_key=roster[0])
        )
    # The budget refreshes at the next transaction phase.
    env.transact()
    env.submit_claim(
        WaiverClaim(add_key=env.observe()["free_agents"][0], drop_key=roster[0])
    )


# ---------------------------------------------------------------- waivers


def test_a_dropped_player_is_locked_up_for_two_days():
    """The rule that makes a cut a commitment.

    Before this existed the roster mechanic would cut a player and re-claim him
    in the same transaction, which is not a thing any league permits.
    """
    pool = _pool(weeks=6)
    config = LeagueConfig(teams=12, first_week=1, last_week=4)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=0)

    roster = list(env.rosters[env.agent_team])
    add, drop = env.observe()["free_agents"][0], roster[0]
    env.submit_claim(WaiverClaim(add_key=add, drop_key=drop))

    assert drop in env.free_agents, "he left the roster"
    assert drop in env.observe()["on_waivers"], "but he is not addable yet"
    assert drop not in env.observe()["free_agents"]
    with pytest.raises(ValueError, match="on waivers"):
        env.submit_claim(WaiverClaim(add_key=drop, drop_key=roster[1]))

    # Friday, two days later: he has cleared.
    env.transact()
    assert drop not in env.observe()["on_waivers"]
    assert env.wire.is_free(drop, env.hour)


def test_a_player_dropped_right_after_being_added_skips_waivers():
    """Otherwise a manager could quarantine anybody by adding and cutting him.

    The roster housekeeping would do it by accident, every time it claimed a
    replacement and then needed the spot back in the same phase.
    """
    pool = _pool(weeks=6)
    config = LeagueConfig(teams=12, first_week=1, last_week=4)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=0)

    roster = list(env.rosters[env.agent_team])
    add = env.observe()["free_agents"][0]
    env.submit_claim(WaiverClaim(add_key=add, drop_key=roster[0]))
    # Undo it in the same phase: he was added moments ago, so cutting him now
    # puts him straight back in the pool.
    env.submit_claim(WaiverClaim(add_key=env.observe()["free_agents"][0], drop_key=add))
    assert env.wire.is_free(add, env.hour), "the undo sent him to waivers"
    assert add not in env.observe()["on_waivers"]


def test_a_player_cut_on_friday_is_not_available_until_the_next_week():
    """His 48 hours run out after the games have been played."""
    pool = _pool(weeks=8)
    config = LeagueConfig(teams=12, first_week=1, last_week=4)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=0)

    env.transact()  # to Friday
    roster = list(env.rosters[env.agent_team])
    add, drop = env.observe()["free_agents"][0], roster[0]
    env.submit_claim(WaiverClaim(add_key=add, drop_key=drop))
    assert not env.wire.is_free(drop, env.hour)

    env.step({key: 1.0 for key in env.rosters[env.agent_team]})
    # Next Wednesday.
    assert env.wire.is_free(drop, env.hour), "he never cleared"


def test_nobody_is_reclaimed_before_his_waiver_period_ends():
    """The behaviour the period exists to prevent, checked end to end.

    Not "never in the same week": a player cut on Wednesday clears on Friday and
    may legitimately be claimed before the games. The rule is the 48 hours, so
    that is what is checked.
    """
    from ffmodel.league.waivers import WAIVER_HOURS

    pool = _pool(weeks=8, stars=2)
    config = LeagueConfig(teams=12, first_week=1, last_week=6)
    env = FantasyLeagueEnv(pool, season=2024, config=config, seed=3)
    result = run_episode(env, EwmaPolicy())

    moves = [m for week in result.weeks for m in week.moves]
    dropped_at = {}
    for move in moves:
        if move.kind == "drop":
            dropped_at[move.player_key] = move.hour
        elif move.kind == "waiver-add":
            when = dropped_at.get(move.player_key)
            if when is not None:
                assert move.hour - when >= WAIVER_HOURS, (
                    f"{move.player_key} was cut at {when} and reclaimed at "
                    f"{move.hour}, inside the {WAIVER_HOURS}-hour period"
                )
