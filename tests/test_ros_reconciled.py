"""The reconciled rest-of-season model: the identity, the state it reads, the intervals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly import injury_type
from ffmodel.weekly.injury_type import MECHANISMS, attach_injury_type, mechanism
from ffmodel.weekly.ros_reconciled import (
    ERA_START,
    QUANTILE_LEVELS,
    ReconciledROS,
    ResidualQuantiles,
    _absence_groups,
    add_absence_state,
    summarise,
)

CACHE = Path(".cache/weekly_features_2016_2025.pkl")


# ---------------------------------------------------------------- injury mechanism

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hamstring", "muscle"),
        ("right quadricep", "muscle"),
        ("hip flexor", "muscle"),          # the muscle, not the joint
        ("Knee", "joint_ligament"),
        ("left shoulder", "joint_ligament"),
        ("ribs", "bone_contact"),
        ("Concussion", "head"),
        ("Achilles", "tendon_severe"),
        ("illness", "illness"),
        ("not injury related - ankle", "nonmed"),   # not an ankle
        ("knee, ankle, elbow", "multiple"),
        ("something new", "other"),
        (None, "none"),
        (float("nan"), "none"),
    ],
)
def test_mechanism_groups(text, expected):
    assert mechanism(text) == expected


def test_every_mechanism_group_is_reachable():
    seen = {mechanism(t) for t in ("hamstring", "knee", "concussion", "rib", "achilles", "back",
                                    "illness", "personal", "knee, hip", "zzz")}
    assert seen == set(MECHANISMS)


def test_previous_mechanism_is_the_previous_game_not_the_previous_week(monkeypatch):
    """A bye is not a game: the lag is over the player's own rows."""
    frame = pd.DataFrame({
        "player_key": ["a"] * 3, "season": [2024] * 3, "week": [3, 5, 6],  # bye in week 4
    })
    table = pd.DataFrame({"season": [2024], "week": [3], "player_key": ["a"], "mech_cur": ["muscle"]})
    monkeypatch.setattr(injury_type, "load_injury_mechanisms", lambda seasons: table)
    out = attach_injury_type(frame)
    assert out["mech_prev"].tolist() == ["none", "muscle", "none"]
    assert out["mechp_muscle"].tolist() == [0.0, 1.0, 0.0]


# ---------------------------------------------------------------- absence state

def _rows(played, status=None, injury=None, since=None, season=2024):
    n = len(played)
    return pd.DataFrame({
        "player_key": ["p"] * n, "season": [season] * n, "week": list(range(1, n + 1)),
        "played": played,
        "status": status or ["ACT"] * n,
        "inj_status_lagged": injury or [0.0] * n,
        "weeks_since_played": since or [np.nan] + [1.0] * (n - 1),
    })


def test_absence_state_reads_the_previous_game():
    out = add_absence_state(_rows([1, 0, 0], status=["ACT", "RES", "RES"], since=[np.nan, 1.0, 2.0]))
    assert out["missed_last"].tolist() == [0.0, 0.0, 1.0]
    assert out["res_prev"].tolist() == [0.0, 0.0, 1.0]
    assert out["missed_n"].tolist() == [0.0, 0.0, 1.0]   # weeks_since_played 2 = one game missed


def test_the_reason_flags_are_distinct():
    # `inj_status_lagged` already describes the previous game, so it is read as is;
    # the roster status is lagged here.
    frame = _rows([0, 0, 0, 0, 0], status=["ACT", "INA", "ACT", "ACT", "ACT"],
                  injury=[0.0, 0.0, 3.0, 1.0, 0.0])
    out = add_absence_state(frame)
    assert out["ina_prev"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]   # row 2 follows the INA row
    assert out["out_prev"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]   # lagged status 3 = Out
    assert out["qd_prev"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]    # lagged status 1 = Questionable


def test_the_era_term_starts_where_returns_changed():
    frame = pd.concat([_rows([1, 1], season=ERA_START - 1), _rows([1, 1], season=ERA_START)])
    assert add_absence_state(frame.reset_index(drop=True))["era21"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_groups_separate_why_he_is_out():
    frame = add_absence_state(pd.DataFrame({
        "player_key": list("abcdef"), "season": 2024, "week": 3,
        "played": 0, "status": "ACT", "inj_status_lagged": 0.0, "weeks_since_played": 2.0,
    }))
    frame["missed_last"] = 1.0
    for flag, who in (("res_prev", 1), ("out_prev", 2), ("qd_prev", 3), ("ina_prev", 4)):
        frame.loc[who, flag] = 1.0
    groups = _absence_groups(frame)
    assert len(set(groups[[1, 2, 3, 4]])) == 4                      # four different states
    assert groups[0] == groups[5]                                   # the two with no flag match


# ---------------------------------------------------------------- intervals

def test_residual_quantiles_are_ordered_and_nonnegative():
    rng = np.random.default_rng(0)
    total = rng.gamma(4.0, 20.0, 4000)
    actual = np.maximum(total + rng.normal(0.0, 15.0, 4000), 0.0)
    rq = ResidualQuantiles.build(total, actual, rng.integers(0, 2, 4000), rng.integers(0, 2, 4000))
    grid = rq.quantiles(total[:300], np.zeros(300), np.ones(300))
    assert grid.shape == (300, len(QUANTILE_LEVELS))
    assert (np.diff(grid, axis=1) >= 0).all() and (grid >= 0).all()
    q = summarise(grid)
    assert (q["p10"] <= q["p50"]).all() and (q["p50"] <= q["p90"]).all()


def test_residual_quantiles_cover_about_nominally_on_their_own_data():
    rng = np.random.default_rng(1)
    total = rng.gamma(4.0, 20.0, 6000)
    actual = np.maximum(total + rng.normal(0.0, 15.0, 6000), 0.0)
    flags = np.zeros(6000)
    rq = ResidualQuantiles.build(total, actual, flags, flags)
    q = summarise(rq.quantiles(total, flags, flags))
    assert np.mean((actual >= q["p10"]) & (actual <= q["p90"])) == pytest.approx(0.80, abs=0.04)


def test_a_thin_bin_falls_back_rather_than_failing():
    total = np.linspace(1, 100, 300)
    actual = total + 1.0
    rq = ResidualQuantiles.build(total, actual, np.zeros(300), np.zeros(300))
    # A returner flag and draft status the training data never saw.
    grid = rq.quantiles(np.array([50.0]), np.array([1.0]), np.array([1.0]))
    assert np.isfinite(grid).all()


# ---------------------------------------------------------------- the model itself

@pytest.fixture(scope="module")
def fitted():
    if not CACHE.exists():
        pytest.skip("needs the built feature cache")
    from ffmodel.weekly.features import relevant_population
    from ffmodel.weekly.restofseason import OFFSET, TARGET, add_rest_of_season_target

    frame = add_absence_state(attach_injury_type(add_rest_of_season_target(pd.read_pickle(CACHE))))
    frame = frame[np.isfinite(pd.to_numeric(frame[TARGET], errors="coerce"))].reset_index(drop=True)
    train = frame[frame["season"] < 2024]
    test = frame[frame["season"] == 2024]
    test = test[relevant_population(test).to_numpy(bool)]
    model = ReconciledROS().fit(train).fit_intervals(train)
    return model, train, test, OFFSET, TARGET


def test_the_total_is_expected_games_times_the_rate(fitted):
    """The point of the design: one calculation, so no column can disagree."""
    model, _, test, _, _ = fitted
    out = model.predict(test)
    assert np.allclose(out["total"], out["expected_games"] * out["rate"], rtol=0, atol=1e-9)


def test_expected_games_never_exceed_the_games_left(fitted):
    model, _, test, offset, _ = fitted
    out = model.predict(test)
    assert (out["expected_games"] <= test[offset].to_numpy(float) + 1e-9).all()
    assert (out["expected_games"] >= 0).all()


def test_no_row_has_a_rate_that_is_not_a_plausible_points_per_game(fitted):
    """Nacua read 32 a game when the rate was a ratio of two inconsistent numbers."""
    model, _, test, _, _ = fitted
    assert model.predict(test)["rate"].max() < 40.0


def test_the_weighted_rate_leaves_no_overall_bias_in_sample(fitted):
    """The reason the rate is weighted by games played: rate x games sums to the total.

    Checked where it is a property of the fit, on the training rows. On a single
    held-out season it is only approximately true -- 2024 alone runs 5% low while the
    pooled 2023-2025 bias is 0.0 -- so the holdout bound below is deliberately loose.
    """
    model, train, test, _, target = fitted
    in_sample = model.predict(train)["total"].to_numpy()
    actual = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    assert abs(np.mean(actual - in_sample)) < 0.03 * np.mean(actual)
    held = model.predict(test)["total"].to_numpy()
    held_actual = pd.to_numeric(test[target], errors="coerce").to_numpy(float)
    assert abs(np.mean(held_actual - held)) < 0.10 * np.mean(held_actual)


def test_interval_grid_is_ordered_and_brackets_most_outcomes(fitted):
    model, _, test, _, target = fitted
    out = model.predict(test)
    grid = model.quantile_grid(test, out["total"].to_numpy())
    assert (np.diff(grid, axis=1) >= 0).all()
    q = summarise(grid)
    actual = pd.to_numeric(test[target], errors="coerce").to_numpy(float)
    inside = np.mean((actual >= q["p10"]) & (actual <= q["p90"]))
    assert 0.65 < inside < 0.92   # near the nominal 0.80; the exact number is in the validation


def test_predicting_before_intervals_are_fitted_is_an_error(fitted):
    model, train, test, _, _ = fitted
    bare = ReconciledROS().fit(train)
    with pytest.raises(RuntimeError, match="fit_intervals"):
        bare.quantile_grid(test, np.ones(len(test)))
