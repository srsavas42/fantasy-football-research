"""Team-volume model preparation and posterior simulation invariants."""

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from ffmodel.models.volume_team import TeamVolumeModel, prepare_team_weeks


def _team_weeks():
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 2, 2],
            "team": ["A", "B", "A", "B"],
            "team_pass_att": [36, 30, 40, 28],
            "team_rush_att": [24, 31, 22, 34],
            "is_home": [1, 0, 0, 1],
            "rest_days": [7, 7, 7, 7],
        }
    )


def _posterior(model, draws=12):
    n_team = len(model.teams)
    return az.from_dict(
        posterior={
            "play_intercept": np.full((1, draws), np.log(62.0)),
            "play_team": np.zeros((1, draws, n_team)),
            "play_beta": np.zeros((1, draws, 3)),
            "pass_intercept": np.full((1, draws), 0.25),
            "pass_team": np.zeros((1, draws, n_team)),
            "pass_beta": np.zeros((1, draws, 3)),
            "target_intercept": np.full((1, draws), 2.5),
            "target_team": np.zeros((1, draws, n_team)),
            "target_beta": np.zeros((1, draws, 3)),
            "play_alpha": np.full((1, draws), 20.0),
        }
    )


def test_prepare_team_weeks_needs_no_vegas_columns():
    out = prepare_team_weeks(_team_weeks())
    assert len(out) == 4
    assert out["team_plays"].tolist() == [60, 61, 62, 62]
    assert "spread" not in out and "implied_team_total" not in out
    assert (out["team_pass_att"] <= out["team_plays"]).all()


def test_posterior_draws_are_nonnegative_and_coherent():
    model = TeamVolumeModel()
    model._design(_team_weeks(), fit=True)
    model.idata = _posterior(model)
    pred = model.predict_samples(_team_weeks(), seed=8)
    assert pred["plays"].shape == (4, 12)
    assert (pred["plays"] >= 0).all()
    assert (pred["pass_attempts"] >= 0).all()
    assert (pred["pass_attempts"] <= pred["plays"]).all()
    assert (pred["targets"] >= 0).all()
    assert (pred["targets"] <= pred["pass_attempts"]).all()


def test_unseen_team_uses_population_prior():
    model = TeamVolumeModel()
    model._design(_team_weeks(), fit=True)
    model.idata = _posterior(model)
    future = _team_weeks().head(1).assign(team="NEW", season=2025, week=1)
    pred = model.predict_samples(future)
    assert pred["plays"].shape == (1, 12)
    assert np.isfinite(pred["plays"]).all()
