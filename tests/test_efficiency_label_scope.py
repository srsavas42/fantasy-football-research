"""A numerator larger than its own exposure must not become a training label.

`player_preseason_rows` merges the efficiency labels on `(season, player_key)`
while the frame is keyed by `(season, team, player_key)`. The numerator columns
(`eff_*`) are therefore the player's season total across every team he played
for, while the exposure stays team-scoped to his Week-1 roster snapshot. A
mid-season move pairs one team's targets with the whole season's receptions.

`fit` clips success to exposure, so such a row does not raise and does not get
dropped -- it trains as a rate of exactly 1.000. That is a fabricated label, and
it is the worst of the three outcomes because nothing about it looks wrong
downstream.

These tests pin the filter that rejects those rows, and the clip that would
otherwise hide them.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    PosteriorSeasonEfficiencyModel,
)

BETA_BINOMIAL_TARGETS = tuple(
    spec.target
    for spec in EFFICIENCY_MODEL_BY_TARGET.values()
    if spec.likelihood == "beta_binomial"
)


def _rows(target: str) -> pd.DataFrame:
    """Three clean rows and one whose numerator exceeds its exposure."""
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    position = spec.positions[0]
    exposure = [80, 90, 100, 30]
    numerator = [8, 9, 10, 45]  # the last one is impossible
    return pd.DataFrame(
        {
            "position": [position] * 4,
            spec.exposure: exposure,
            spec.numerator: numerator,
            spec.prior_exposure: exposure,
            spec.prior_feature: [0.1, 0.1, 0.1, 0.1],
            target: [n / e for n, e in zip(numerator, exposure)],
            "is_replacement_player": [0, 0, 0, 0],
        }
    )


@pytest.mark.parametrize("target", BETA_BINOMIAL_TARGETS)
def test_impossible_rate_is_not_eligible(target: str) -> None:
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    model = PosteriorSeasonEfficiencyModel(spec, mean_mode="prior")
    eligible = model._eligible(_rows(target))

    numerator = pd.to_numeric(eligible[spec.numerator], errors="coerce")
    exposure = pd.to_numeric(eligible[spec.exposure], errors="coerce")
    assert (numerator <= exposure).all()
    # The three clean rows survive, so the filter is not simply emptying the
    # frame for every response.
    assert len(eligible) == 3


@pytest.mark.parametrize("target", BETA_BINOMIAL_TARGETS)
def test_the_clip_would_have_fabricated_a_perfect_rate(target: str) -> None:
    """Why the filter is needed: the clip turns the bad row into rate 1.000.

    This reproduces what `fit` does to a row the filter did not remove, so the
    consequence is asserted rather than described in a comment.
    """
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    rows = _rows(target)
    exposure = rows[spec.exposure].to_numpy(int)
    success = np.clip(rows[spec.numerator].to_numpy(int), 0, exposure)
    implied = success / exposure
    assert implied[-1] == pytest.approx(1.0)
    assert implied[:-1].max() < 0.2


def test_a_clean_frame_is_untouched() -> None:
    """The filter must not cost rows on data that was already coherent."""
    target = "rec_catch_rate"
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    rows = _rows(target).iloc[:3].reset_index(drop=True)
    model = PosteriorSeasonEfficiencyModel(spec, mean_mode="prior")
    assert len(model._eligible(rows)) == 3
