"""Dirichlet-Multinomial active-set construction and reallocation."""

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from ffmodel.models.volume_share import OpportunityShareModel


def _features():
    rows = []
    for week in (1, 2):
        for team in ("A", "B"):
            for name, position, targets, share in (
                (f"{team}-alpha", "WR", 6, 0.55),
                (f"{team}-beta", "WR", 3, 0.30),
                (f"{team}-back", "RB", 1, 0.15),
            ):
                rows.append(
                    {
                        "player_id": name,
                        "player_name": name,
                        "position": position,
                        "team": team,
                        "season": 2024,
                        "week": week,
                        "targets": targets,
                        "rush_att": 0,
                        "ewma_target_share": share,
                        "ewma_opportunity_share": share,
                        "ewma_ypt": 7.0,
                        "ewma_catch_rate": 0.65,
                        "is_active": 1,
                    }
                )
    return pd.DataFrame(rows)


def _posterior(model, draws=40):
    player_effect = np.zeros((1, draws, len(model.players)))
    for index, name in enumerate(model.players):
        if "alpha" in name:
            player_effect[:, :, index] = 1.5
    return az.from_dict(
        posterior={
            "mu_position": np.zeros((1, draws)),
            "position_alpha": np.zeros((1, draws, len(model.positions))),
            "player_effect": player_effect,
            "beta": np.zeros((1, draws, len(model.feature_names))),
        }
    )


def test_design_is_ragged_and_counts_conserve_group_totals():
    model = OpportunityShareModel("target")
    design = model._design(_features(), fit=True)
    assert design.counts.shape == (4, 3)
    assert design.mask.sum(axis=1).tolist() == [3.0] * 4
    assert (design.counts.sum(axis=1) == design.totals).all()
    assert design.totals.tolist() == [10, 10, 10, 10]


def test_starter_removal_reallocates_all_targets_to_remaining_players():
    model = OpportunityShareModel("target")
    model._design(_features(), fit=True)
    model.idata = _posterior(model)
    week = _features().query("week == 2 and team == 'A'")
    full = model.predict_samples(week, team_totals=np.array([10]), seed=3)
    reduced_rows = week[~week["player_name"].str.contains("alpha")]
    reduced = model.predict_samples(reduced_rows, team_totals=np.array([10]), seed=3)

    assert (full.counts.sum(axis=0) == 10).all()
    assert (reduced.counts.sum(axis=0) == 10).all()
    assert np.allclose(full.shares.sum(axis=0), 1.0)
    assert np.allclose(reduced.shares.sum(axis=0), 1.0)
    beta_full = full.counts[full.rows["player_name"].str.contains("beta")].mean()
    beta_reduced = reduced.counts[reduced.rows["player_name"].str.contains("beta")].mean()
    assert beta_reduced > beta_full


def test_zero_opportunity_skill_player_remains_in_legacy_support():
    frame = _features().head(3).copy()
    frame.loc[2, "targets"] = 0
    frame.loc[2, "is_active"] = 0
    model = OpportunityShareModel("target")
    design = model._design(frame, fit=True)
    assert len(design.rows) == 3
    assert design.mask.sum() == 3


def test_unplayed_week_accepts_zero_outcomes_when_team_total_is_supplied():
    model = OpportunityShareModel("target")
    model._design(_features(), fit=True)
    model.idata = _posterior(model)
    future = _features().query("week == 2 and team == 'A'").assign(targets=0)
    pred = model.predict_samples(future, team_totals=np.array([12]), seed=4)
    assert pred.counts.shape == (3, 40)
    assert (pred.counts.sum(axis=0) == 12).all()
    assert np.allclose(pred.shares.sum(axis=0), 1.0)
