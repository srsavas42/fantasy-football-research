"""Joint volume-efficiency season scoring invariants."""

import numpy as np
import pandas as pd

from ffmodel.models.efficiency_season_average import (
    SeasonAverageEfficiencyPrediction,
)
from ffmodel.models.volume_season_average import SeasonAveragePrediction
from ffmodel.simulation.season_scoring import (
    REQUIRED_EFFICIENCY_TARGETS,
    apply_efficiency_copulas,
    fantasy_points_samples,
    scale_efficiency_dispersion,
    scale_fantasy_point_dispersion,
    score_volume_prediction,
    simulate_season_scoring,
    volume_efficiency_exposures,
    volume_efficiency_feature_samples,
)


def test_joint_scoring_enforces_opportunity_and_touchdown_identities():
    volume = _volume_prediction(draws=80)
    efficiency = _efficiency_prediction(volume)

    prediction = simulate_season_scoring(volume, efficiency, seed=19)

    assert (prediction.pass_cmp <= volume.pass_attempts).all()
    assert (prediction.pass_td <= prediction.pass_cmp).all()
    assert (prediction.pass_cmp + prediction.pass_int <= volume.pass_attempts).all()
    assert (prediction.receptions <= volume.targets).all()
    assert (prediction.rec_td <= prediction.receptions).all()
    assert (prediction.rush_td <= volume.carries).all()
    assert (prediction.fumbles_lost <= volume.pass_attempts + volume.targets + volume.carries).all()
    assert (prediction.pass_yds[prediction.pass_cmp == 0] == 0).all()
    assert (prediction.rec_yds[prediction.receptions == 0] == 0).all()
    assert set(prediction.fantasy_points) == {"standard", "half_ppr", "ppr"}
    assert prediction.fantasy_points["ppr"].shape == volume.pass_attempts.shape


def test_scoring_is_reproducible_and_ppr_increment_is_receptions():
    volume = _volume_prediction(draws=30)
    efficiency = _efficiency_prediction(volume)

    first = simulate_season_scoring(volume, efficiency, seed=7)
    second = simulate_season_scoring(volume, efficiency, seed=7)

    assert np.array_equal(first.pass_cmp, second.pass_cmp)
    assert np.array_equal(first.fantasy_points["ppr"], second.fantasy_points["ppr"])
    assert np.allclose(
        first.fantasy_points["ppr"] - first.fantasy_points["standard"],
        first.receptions,
    )
    summary = first.summary("ppr")
    assert {"ppr_points_mean", "ppr_points_p10", "ppr_points_p50", "ppr_points_p90"} <= set(summary)


def test_volume_exposure_mapping_uses_the_same_draws():
    volume = _volume_prediction(draws=5)
    exposure = volume_efficiency_exposures(volume)

    assert set(exposure) == set(REQUIRED_EFFICIENCY_TARGETS)
    assert np.array_equal(exposure["pass_td_rate"], volume.pass_attempts)
    assert np.array_equal(exposure["rec_catch_rate"], volume.targets)
    assert np.array_equal(exposure["rush_yards_per_carry"], volume.carries)
    assert np.array_equal(
        exposure["fumble_lost_rate"],
        volume.pass_attempts + volume.targets + volume.carries,
    )


def test_draw_conditioned_handoff_uses_per_team_game_volume_draws():
    volume = _volume_prediction(draws=5)

    class CapturingEfficiency:
        def __init__(self):
            self.feature_samples = "not-called"

        def predict_samples(self, rows, **kwargs):
            self.feature_samples = kwargs.get("volume_feature_samples")
            return _efficiency_prediction(volume)

    candidate = CapturingEfficiency()
    score_volume_prediction(
        volume,
        candidate,
        draw_conditioned_efficiency=True,
        seed=2,
    )

    assert set(candidate.feature_samples) == {
        "oof_pass_attempts_per_team_game",
        "oof_targets_per_team_game",
        "oof_carries_per_team_game",
        "oof_fumble_opportunities_per_team_game",
    }
    assert np.array_equal(
        candidate.feature_samples["oof_targets_per_team_game"],
        volume.targets_per_team_game,
    )
    assert np.array_equal(
        candidate.feature_samples["oof_fumble_opportunities_per_team_game"],
        volume.pass_attempts_per_team_game
        + volume.targets_per_team_game
        + volume.carries_per_team_game,
    )

    baseline = CapturingEfficiency()
    score_volume_prediction(volume, baseline, seed=2)
    assert baseline.feature_samples is None


def test_efficiency_dispersion_scale_preserves_locations_and_bounds():
    volume = _volume_prediction(draws=5)
    efficiency = _efficiency_prediction(volume)
    efficiency.rates["pass_td_rate"] = (
        efficiency.rates["pass_td_rate"] + 0.02
    )

    collapsed = scale_efficiency_dispersion(efficiency, 0.0)
    expanded = scale_efficiency_dispersion(efficiency, 2.0)

    assert np.array_equal(
        collapsed.rates["pass_td_rate"], efficiency.means["pass_td_rate"]
    )
    assert np.allclose(
        expanded.rates["pass_td_rate"],
        efficiency.means["pass_td_rate"] + 0.04,
    )
    assert (expanded.rates["pass_td_rate"] <= 1.0).all()


def test_point_dispersion_scale_preserves_means_and_stat_lines():
    volume = _volume_prediction(draws=30)
    prediction = simulate_season_scoring(
        volume, _efficiency_prediction(volume), seed=4
    )

    expanded = scale_fantasy_point_dispersion(prediction, 1.3)

    assert np.allclose(
        expanded.fantasy_points["ppr"].mean(axis=1),
        prediction.fantasy_points["ppr"].mean(axis=1),
    )
    assert np.array_equal(expanded.pass_yds, prediction.pass_yds)
    assert np.array_equal(expanded.receptions, prediction.receptions)


def test_copula_reorders_draws_without_changing_marginals():
    volume = _volume_prediction(draws=500)
    efficiency = _efficiency_prediction(volume)
    rng = np.random.default_rng(10)
    pass_targets = (
        "pass_completion_rate",
        "pass_yards_per_attempt",
        "pass_td_rate",
        "pass_int_rate",
    )
    for target in pass_targets:
        efficiency.rates[target][0] = rng.normal(size=500)
    original = {
        target: np.sort(efficiency.rates[target][0].copy())
        for target in pass_targets
    }
    correlation = np.array(
        [
            [1.0, 0.7, 0.5, -0.3],
            [0.7, 1.0, 0.6, -0.2],
            [0.5, 0.6, 1.0, -0.2],
            [-0.3, -0.2, -0.2, 1.0],
        ]
    )

    correlated = apply_efficiency_copulas(
        efficiency, {"pass": correlation}, seed=8
    )

    for target in pass_targets:
        assert np.array_equal(np.sort(correlated.rates[target][0]), original[target])
    rank_correlation = pd.DataFrame(
        {target: correlated.rates[target][0] for target in pass_targets}
    ).corr(method="spearman")
    assert rank_correlation.loc[pass_targets[0], pass_targets[1]] > 0.5
    assert rank_correlation.loc[pass_targets[0], pass_targets[3]] < -0.15


def test_array_scoring_matches_manual_weights():
    statistics = {
        "pass_yds": np.array([[300.0]]),
        "pass_td": np.array([[2.0]]),
        "pass_int": np.array([[1.0]]),
        "rush_yds": np.array([[40.0]]),
        "rush_td": np.array([[1.0]]),
        "rec_yds": np.array([[20.0]]),
        "rec_td": np.array([[1.0]]),
        "receptions": np.array([[3.0]]),
        "fumbles_lost": np.array([[1.0]]),
    }

    assert np.isclose(fantasy_points_samples(statistics, "ppr")[0, 0], 37.0)


def _volume_prediction(draws):
    rows = pd.DataFrame(
        {
            "season": [2025] * 4,
            "team": ["A"] * 4,
            "player_key": ["qb", "rb", "wr", "te"],
            "position": ["QB", "RB", "WR", "TE"],
        }
    )
    passes = np.repeat(np.array([600, 0, 0, 0])[:, None], draws, axis=1)
    targets = np.repeat(np.array([0, 70, 140, 90])[:, None], draws, axis=1)
    carries = np.repeat(np.array([60, 250, 4, 1])[:, None], draws, axis=1)
    zeros = np.zeros((4, draws), dtype=float)
    ones = np.ones((4, draws), dtype=float)
    team_games = np.full((4, draws), 17.0)
    return SeasonAveragePrediction(
        team={"rows": pd.DataFrame({"season": [2025], "team": ["A"]})},
        player_rows=rows,
        availability_probability=ones,
        games_active=np.full((4, draws), 17),
        availability=ones,
        snap_share=ones,
        qb_workload_share=zeros,
        qb_pass_propensity=zeros,
        target_role_probability=zeros,
        carry_eligibility_probability=ones,
        pass_attempt_share=zeros,
        target_share=zeros,
        carry_share=zeros,
        pass_attempts=passes,
        targets=targets,
        carries=carries,
        pass_attempts_per_team_game=passes / team_games,
        targets_per_team_game=targets / team_games,
        carries_per_team_game=carries / team_games,
        pass_attempts_per_active_game=passes / team_games,
        targets_per_active_game=targets / team_games,
        carries_per_active_game=carries / team_games,
    )


def _efficiency_prediction(volume):
    shape = volume.pass_attempts.shape
    position = volume.player_rows["position"].to_numpy()

    def values(mapping):
        base = np.array([mapping.get(name, 0.0) for name in position], dtype=float)
        return np.repeat(base[:, None], shape[1], axis=1)

    rates = {
        "pass_completion_rate": values({"QB": 0.65}),
        "pass_yards_per_attempt": values({"QB": 7.4}),
        "pass_td_rate": values({"QB": 0.05}),
        "pass_int_rate": values({"QB": 0.025}),
        "rec_catch_rate": values({"RB": 0.75, "WR": 0.64, "TE": 0.70}),
        "rec_yards_per_target": values({"RB": 6.2, "WR": 8.5, "TE": 7.4}),
        "rec_td_rate": values({"RB": 0.035, "WR": 0.065, "TE": 0.06}),
        "rush_yards_per_carry": values({"QB": 5.0, "RB": 4.5, "WR": 6.0, "TE": 2.0}),
        "rush_td_rate": values({"QB": 0.06, "RB": 0.045, "WR": 0.03, "TE": 0.02}),
        "fumble_lost_rate": values({"QB": 0.004, "RB": 0.003, "WR": 0.002, "TE": 0.002}),
    }
    return SeasonAverageEfficiencyPrediction(
        player_rows=volume.player_rows.copy(),
        means={key: value.copy() for key, value in rates.items()},
        rates=rates,
    )
