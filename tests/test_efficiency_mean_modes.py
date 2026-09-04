"""Which mean mode each response resolves to is the promotion itself.

``PERSISTENCE_MEAN_MODE`` is consulted first and ``POSTERIOR_MEAN_MODE`` second,
so removing a response from the former without setting the latter drops it to
``prior`` -- a third behaviour that was never measured, and one that would look
like a successful promotion in a diff.
"""

from __future__ import annotations

import pytest

from ffmodel.models.efficiency_season_average import (
    PERSISTENCE_MEAN_MODE,
    POSTERIOR_MEAN_MODE,
    EFFICIENCY_MODEL_BY_TARGET,
    PosteriorSeasonEfficiencyModel,
    SeasonAveragePosteriorEfficiencyPipeline,
)


def resolved(target: str) -> str:
    """The mode the *pipeline* builds, which is not the model's own default.

    ``PosteriorSeasonEfficiencyModel`` consults ``POSTERIOR_MEAN_MODE`` alone;
    the persistence table is applied a level up, by the pipeline, gated on
    ``fitted_persistence_means``. Asking the bare model reports ``prior`` for a
    persistence response and would have passed this file while testing nothing.
    """
    pipeline = SeasonAveragePosteriorEfficiencyPipeline()
    mode = (
        PERSISTENCE_MEAN_MODE.get(target)
        if pipeline.fitted_persistence_means
        else None
    )
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET[target], mean_mode=mode
    )
    return model._mean_mode()


@pytest.mark.parametrize(
    "target,mode",
    [
        ("rec_catch_rate", "posterior"),
        ("rush_td_rate", "posterior"),
        ("rec_td_rate", "persistence"),
        ("rec_yards_per_target", "posterior"),
        ("rush_yards_per_carry", "ridge"),
        ("fumble_rate", "prior"),
    ],
)
def test_each_response_resolves_to_the_mode_it_was_measured_in(target, mode):
    assert resolved(target) == mode


def test_the_promoted_responses_left_the_persistence_table():
    assert "rec_catch_rate" not in PERSISTENCE_MEAN_MODE
    assert "rush_td_rate" not in PERSISTENCE_MEAN_MODE


def test_they_did_not_fall_back_to_prior():
    """Deleting the persistence entry alone would have done exactly that."""
    assert POSTERIOR_MEAN_MODE["rec_catch_rate"] == "posterior"
    assert POSTERIOR_MEAN_MODE["rush_td_rate"] == "posterior"


def test_the_unpromoted_response_still_fits_an_empty_design():
    model = PosteriorSeasonEfficiencyModel(
        EFFICIENCY_MODEL_BY_TARGET["rec_td_rate"], mean_mode="persistence"
    )
    assert model._candidates() == ()


def test_the_promoted_responses_now_have_a_design():
    for target in ("rec_catch_rate", "rush_td_rate"):
        model = PosteriorSeasonEfficiencyModel(
            EFFICIENCY_MODEL_BY_TARGET[target], mean_mode=resolved(target)
        )
        assert len(model._candidates()) > 1
