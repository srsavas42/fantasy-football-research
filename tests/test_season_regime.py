"""Contracts for the leakage-safe player-season regime screen."""

import numpy as np
import pandas as pd

from ffmodel.models.season_regime import (
    REGIME_NAMES,
    SeasonRegimeModel,
    add_regime_probabilities,
    add_walk_forward_regime_probabilities,
    fit_regime_thresholds,
    realized_regimes,
)


def _rows():
    return pd.DataFrame(
        [
            {"position": "QB", "observed_availability": 1.0, "snap_share": 0.9, "observed_qb_workload_share": 0.9, "prior_pass_role": 0.8, "prior_availability": 1.0, "roster_active": 1, "depth_rank": 1, "qb_listed_starter": 1},
            {"position": "QB", "observed_availability": 0.8, "snap_share": 0.3, "observed_qb_workload_share": 0.3, "prior_pass_role": 0.3, "prior_availability": 0.7, "roster_active": 1, "depth_rank": 2, "qb_listed_starter": 0},
            {"position": "RB", "observed_availability": 1.0, "snap_share": 0.7, "carry_share": 0.5, "prior_carry_role": 0.8, "prior_availability": 1.0, "roster_active": 1, "depth_rank": 1},
            {"position": "RB", "observed_availability": 0.6, "snap_share": 0.2, "carry_share": 0.1, "prior_carry_role": 0.2, "prior_availability": 0.7, "roster_active": 1, "depth_rank": 2},
            {"position": "WR", "observed_availability": 0.1, "snap_share": 0.01, "target_share": 0.0, "prior_target_role": 0.0, "prior_availability": 0.1, "roster_active": 0, "depth_rank": 4},
            {"position": "WR", "observed_availability": 1.0, "snap_share": 0.8, "target_share": 0.25, "prior_target_role": 0.8, "prior_availability": 1.0, "roster_active": 1, "depth_rank": 1},
            {"position": "TE", "observed_availability": 1.0, "snap_share": 0.7, "target_share": 0.15, "prior_target_role": 0.7, "prior_availability": 1.0, "roster_active": 1, "depth_rank": 1},
            {"position": "TE", "observed_availability": 0.5, "snap_share": 0.2, "target_share": 0.03, "prior_target_role": 0.2, "prior_availability": 0.6, "roster_active": 1, "depth_rank": 2},
            {"position": "WR", "is_replacement_player": 1, "observed_availability": 0.7, "snap_share": 0.4, "target_share": 0.1, "roster_active": 0, "depth_rank": 5},
        ]
    )


def test_realized_regime_labels_hold_replacements_out_of_learned_classes():
    rows = _rows()
    labels = realized_regimes(rows, fit_regime_thresholds(rows))

    assert labels[-1] == "replacement"
    assert labels[4] == "inactive"
    assert "lead" in labels
    assert "committee" in labels


def test_regime_prediction_uses_preseason_fields_and_returns_simplex_draws():
    rows = _rows()
    model = SeasonRegimeModel(steps=100).fit(rows)
    future = rows.drop(columns=["observed_availability", "snap_share", "observed_qb_workload_share", "target_share", "carry_share"], errors="ignore")
    prediction = model.predict_samples(future, draws=17, seed=4)

    assert prediction.probability.shape == (len(rows), len(REGIME_NAMES))
    assert np.allclose(prediction.probability.sum(axis=1), 1.0)
    assert prediction.samples.shape == (len(rows), 17)
    assert ((prediction.samples >= 0) & (prediction.samples < len(REGIME_NAMES))).all()
    assert np.all(prediction.probability[-1] == np.array([1.0, 0.0, 0.0, 0.0]))


def test_regime_prediction_state_round_trip_is_exact():
    rows = _rows()
    model = SeasonRegimeModel(steps=100).fit(rows)
    restored = SeasonRegimeModel.from_state(model.state_dict())

    assert np.allclose(restored.predict_proba(rows), model.predict_proba(rows))


def test_walk_forward_regime_probabilities_use_only_earlier_seasons():
    rows = pd.concat([_rows().assign(season=2022), _rows().assign(season=2023)], ignore_index=True)
    featured = add_walk_forward_regime_probabilities(rows, classifier_steps=50)
    model = SeasonRegimeModel(steps=50).fit(rows[rows["season"].eq(2022)])
    expected = add_regime_probabilities(rows[rows["season"].eq(2023)], model)

    columns = ["regime_probability_inactive", "regime_probability_committee", "regime_probability_lead"]
    assert np.allclose(featured.loc[featured.season.eq(2022), columns].sum(axis=1), 1.0)
    assert np.allclose(
        featured.loc[featured.season.eq(2023), columns].to_numpy(),
        expected[columns].to_numpy(),
    )
