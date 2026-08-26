"""The fitted-persistence mean, and the two ways it could quietly not be one.

``prior`` mode is an identity map: the conditional mean it hands the simulator
*is* the lagged shrunk feature, so the layer asserts a persistence coefficient
of exactly 1.000 and the whole of its regression to the mean is the ``K``
pseudo-count in the feature builder. ``persistence`` mode replaces that
assertion with an intercept, sum-to-zero position offsets and a fitted slope,
and nothing else.

Two failure modes are worth pinning. The mode could admit covariates by
accident, which would make it the efficiency-v2 posterior arm under a new name
rather than the two-parameter change that was measured. And the flag could fail
to reach the models, leaving a pipeline that reports the new configuration
while fitting the old one.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    PERSISTENCE_MEAN_MODE,
    POSTERIOR_MEAN_MODE,
    PosteriorSeasonEfficiencyModel,
    SeasonAveragePosteriorEfficiencyPipeline,
)

PRIOR_MODE_TARGETS = tuple(
    target for target, mode in POSTERIOR_MEAN_MODE.items() if mode == "prior"
)


def _rows(target: str, n: int = 120) -> pd.DataFrame:
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    rng = np.random.default_rng(11)
    positions = np.resize(np.array(spec.positions), n)
    span = spec.upper - spec.lower
    value = spec.lower + rng.uniform(0.05, 0.35, n) * span
    return pd.DataFrame(
        {
            "position": positions,
            spec.prior_feature: value,
            spec.prior_exposure: rng.integers(30, 200, n),
            spec.exposure: rng.integers(30, 200, n),
            target: value,
        }
    )


@pytest.mark.parametrize("target", sorted(PERSISTENCE_MEAN_MODE))
def test_persistence_admits_no_covariates(target: str) -> None:
    """An empty design is the point of the mode, not an oversight."""
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    model = PosteriorSeasonEfficiencyModel(spec, mean_mode="persistence")
    assert model._candidates() == ()

    design = model._matrix(_rows(target), fit=True)
    assert design.shape[1] == 0
    assert model.feature_names == []

    # The contrast that makes the claim mean something: the posterior arm on
    # the same frame does admit covariates.
    posterior = PosteriorSeasonEfficiencyModel(spec, mean_mode="posterior")
    assert posterior._candidates()


@pytest.mark.parametrize("target", sorted(PERSISTENCE_MEAN_MODE))
def test_persistence_fits_the_mean_and_prior_does_not(target: str) -> None:
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    assert PosteriorSeasonEfficiencyModel(spec, mean_mode="persistence")._fits_the_mean()
    assert not PosteriorSeasonEfficiencyModel(spec, mean_mode="prior")._fits_the_mean()
    assert PosteriorSeasonEfficiencyModel(spec, mean_mode="posterior")._fits_the_mean()


def test_prior_mode_really_is_an_identity_map() -> None:
    """The premise of the change, asserted rather than assumed.

    If this ever stops holding, ``persistence`` is no longer a generalisation
    of what it replaces and its evidence does not transfer.
    """
    target = "rec_td_rate"
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    rows = _rows(target)
    model = PosteriorSeasonEfficiencyModel(spec, mean_mode="prior")
    model._prior_signal(rows, fit=True)
    mean = model._fixed_mean(rows)
    np.testing.assert_allclose(
        mean, rows[spec.prior_feature].to_numpy(float), rtol=1e-6, atol=1e-6
    )


def test_unknown_mode_is_rejected() -> None:
    spec = EFFICIENCY_MODEL_BY_TARGET["rec_td_rate"]
    with pytest.raises(ValueError, match="unsupported posterior mean mode"):
        PosteriorSeasonEfficiencyModel(spec, mean_mode="nonsense")._mean_mode()


def test_flag_off_reproduces_the_shipped_modes_exactly() -> None:
    """The paired arm has to be the old model, not something adjacent to it."""
    for target, mode in POSTERIOR_MEAN_MODE.items():
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        assert PosteriorSeasonEfficiencyModel(spec, mean_mode=None)._mean_mode() == mode


def test_flag_on_moves_only_the_intended_responses() -> None:
    expected = dict(POSTERIOR_MEAN_MODE)
    expected.update(PERSISTENCE_MEAN_MODE)
    changed = {
        target
        for target in POSTERIOR_MEAN_MODE
        if expected[target] != POSTERIOR_MEAN_MODE[target]
    }
    assert changed == set(PERSISTENCE_MEAN_MODE)
    # Every response it moves must have been on ``prior``: the measurement
    # behind the change is about the identity map, and nothing else.
    assert set(PERSISTENCE_MEAN_MODE) <= set(PRIOR_MODE_TARGETS)
    # One is deliberately left behind: fumble_lost_rate is 0.05% of points
    # variance and was never measured, so there is no reason to widen the
    # surface for it. Recorded here so an edit that quietly adds it has to
    # change this test.
    assert "fumble_lost_rate" not in PERSISTENCE_MEAN_MODE


def test_pipeline_default_is_on_and_the_flag_reaches_the_models() -> None:
    assert SeasonAveragePosteriorEfficiencyPipeline().fitted_persistence_means is True
    off = SeasonAveragePosteriorEfficiencyPipeline(fitted_persistence_means=False)
    assert off.fitted_persistence_means is False

    for target in PERSISTENCE_MEAN_MODE:
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        on_mode = PosteriorSeasonEfficiencyModel(
            spec, mean_mode=PERSISTENCE_MEAN_MODE[target]
        )._mean_mode()
        off_mode = PosteriorSeasonEfficiencyModel(spec, mean_mode=None)._mean_mode()
        assert on_mode == "persistence"
        assert off_mode == "prior"
