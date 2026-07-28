"""Contracts for the team-conserving regime role ablation."""

import numpy as np
import pandas as pd

from ffmodel.models.season_regime import SeasonRegimeModel
from ffmodel.models.season_regime_coupling import SeasonRegimeRoleCoupling


def _rows():
    return pd.DataFrame(
        [
            {"position": "WR", "observed_availability": 1.0, "snap_share": 0.9, "target_share": 0.25, "prior_target_role": 0.8},
            {"position": "WR", "observed_availability": 1.0, "snap_share": 0.5, "target_share": 0.10, "prior_target_role": 0.4},
            {"position": "WR", "observed_availability": 0.2, "snap_share": 0.02, "target_share": 0.01, "prior_target_role": 0.1},
            {"position": "RB", "observed_availability": 1.0, "snap_share": 0.6, "carry_share": 0.35, "prior_carry_role": 0.7},
        ]
    )


def test_role_coupling_preserves_group_totals_and_changes_state_conditionally():
    rows = _rows()
    regime = SeasonRegimeModel(steps=100).fit(rows)
    prediction = regime.predict_samples(rows, draws=8, seed=3)
    coupling = SeasonRegimeRoleCoupling().fit(rows, thresholds=regime.thresholds)
    shares = np.full((len(rows), 8), 0.25)
    samples = prediction.samples.copy()
    samples[0, :] = 3  # lead
    samples[1, :] = 2  # committee
    samples[2, :] = 1  # inactive
    adjusted = coupling.apply(
        rows,
        shares,
        samples,
        stream="target",
        group_index=np.zeros(len(rows), dtype=int),
    )

    assert np.allclose(adjusted.sum(axis=0), shares.sum(axis=0))
    assert np.all(adjusted[0] > adjusted[2])
    assert np.isfinite(adjusted).all()


def test_role_coupling_state_round_trip_preserves_adjustment():
    rows = _rows()
    regime = SeasonRegimeModel(steps=100).fit(rows)
    coupling = SeasonRegimeRoleCoupling().fit(rows, thresholds=regime.thresholds)
    restored = SeasonRegimeRoleCoupling.from_state(coupling.state_dict())
    shares = np.full((len(rows), 4), 0.25)
    samples = regime.predict_samples(rows, draws=4, seed=2).samples

    assert np.allclose(
        restored.apply(rows, shares, samples, stream="target", group_index=np.zeros(len(rows), dtype=int)),
        coupling.apply(rows, shares, samples, stream="target", group_index=np.zeros(len(rows), dtype=int)),
    )
