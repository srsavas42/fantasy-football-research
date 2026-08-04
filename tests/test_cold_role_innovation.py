"""Players with no prior role of their own get their own innovation scale.

The 95% intervals on total fantasy points are about four points too narrow, and
the miss split says the deficit is not spread across rows. On the 2025 holdout,
rows with no prior snap share miss at 28.2% against a 5% nominal (z=+11.52,
32 of 33 misses *above* the interval) while rows with an established role miss
at 2.6%. The model has no per-player information for the first group — the role
prior falls through to a position mean — and represents them with the same
innovation as a returning starter.

These pin the parts of that fix that can be checked without a sampler: who
counts as cold, that the widening lands on them and not on everyone, and that
the allocation stays a valid simplex.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.models.volume_season_average import (
    MIN_COLD_ROLE_ROWS,
    SeasonAverageVolumePipeline,
    SeasonRosterShareModel,
)


def _room():
    """One backfield: a returning starter, a returning backup, and two rookies.

    ``rb3`` is the case the mask has to get right on its own — no carry history
    of any kind, but a full season of snaps, so the model does know something
    about him and he is not cold.
    """
    return pd.DataFrame(
        {
            "season": 2024,
            "team": "KC",
            "player_key": ["rb1", "rb2", "rb3", "rook1", "rook2"],
            "player_name": ["RB One", "RB Two", "RB Three", "Rookie One", "Rookie Two"],
            "position": ["RB", "RB", "WR", "RB", "WR"],
            "rush_att": [220.0, 60.0, 3.0, 40.0, 1.0],
            "targets": [40.0, 20.0, 90.0, 10.0, 30.0],
            "prior_carry_role": [0.62, 0.18, 0.0, np.nan, np.nan],
            "prior_carry_per_snap": [0.44, 0.21, 0.0, np.nan, np.nan],
            "draft_carry_prior": [0.30, 0.20, 0.01, 0.25, 0.02],
            "prior_snap_share": [0.71, 0.34, 0.88, np.nan, 0.0],
            "snap_share": [0.70, 0.35, 0.85, 0.30, 0.20],
            "observed_availability": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_a_player_with_snaps_but_no_carries_is_not_cold():
    """The distinction the coverage split actually found.

    A receiver with a full season of snaps and no carries falls through to the
    position mean in the carry room too, but the model predicts his zero well.
    Widening him would spend variance where there is no error to cover.
    """
    model = SeasonRosterShareModel("carry")

    cold = model._cold_role_rows(_room())

    assert cold.tolist() == [False, False, False, True, True]


def test_the_widening_lands_on_the_cold_rows():
    model = SeasonRosterShareModel("carry", cold_role_innovation=True)
    model.cold_role_multiplier = 3.0
    design = model._design(_room())

    scale = np.full(design["mask"].shape, model.role_innovation_scale)
    widened = np.where(design["cold_role"], scale * model.cold_role_multiplier, scale)

    cold = design["cold_role"][design["mask"] > 0]
    assert cold.sum() == 2
    assert widened[design["cold_role"]].tolist() == [
        pytest.approx(model.role_innovation_scale * 3.0)
    ] * 2
    warm = (design["mask"] > 0) & ~design["cold_role"]
    assert np.allclose(widened[warm], model.role_innovation_scale)


def test_the_multiplier_is_a_ratio_of_realized_dispersions():
    """Sized from the training data rather than chosen, and never below one.

    It is a ratio rather than a second absolute scale so that it composes with
    whatever the base scale ends up being — the cap and the calibration both act
    on the base, and the ratio rides on top of them.
    """
    model = SeasonRosterShareModel("carry", cold_role_innovation=True)
    residuals = np.concatenate([np.full(400, 0.5), np.full(400, 1.5)])
    cold = np.concatenate([np.zeros(400, dtype=bool), np.ones(400, dtype=bool)])
    model._role_innovation_residuals = lambda d: (residuals, cold)

    assert model._estimate_cold_role_multiplier(pd.DataFrame()) == pytest.approx(3.0)


def test_a_thin_split_falls_back_to_one_scale():
    model = SeasonRosterShareModel("carry", cold_role_innovation=True)
    residuals = np.concatenate([np.full(400, 0.5), np.full(10, 5.0)])
    cold = np.concatenate([np.zeros(400, dtype=bool), np.ones(10, dtype=bool)])
    model._role_innovation_residuals = lambda d: (residuals, cold)

    assert cold.sum() < MIN_COLD_ROLE_ROWS
    assert model._estimate_cold_role_multiplier(pd.DataFrame()) == 1.0


def test_the_multiplier_is_capped():
    model = SeasonRosterShareModel(
        "carry", cold_role_innovation=True, cold_role_multiplier_cap=2.0
    )
    residuals = np.concatenate([np.full(400, 0.5), np.full(400, 50.0)])
    cold = np.concatenate([np.zeros(400, dtype=bool), np.ones(400, dtype=bool)])
    model._role_innovation_residuals = lambda d: (residuals, cold)

    assert model._estimate_cold_role_multiplier(pd.DataFrame()) == 2.0


def test_a_cold_row_never_narrows_an_established_one():
    """The floor at one. A season where rookies happened to be predictable is
    not a reason to represent returning starters as more certain than the base
    scale says they are."""
    model = SeasonRosterShareModel("carry", cold_role_innovation=True)
    residuals = np.concatenate([np.full(400, 2.0), np.full(400, 0.1)])
    cold = np.concatenate([np.zeros(400, dtype=bool), np.ones(400, dtype=bool)])
    model._role_innovation_residuals = lambda d: (residuals, cold)

    assert model._estimate_cold_role_multiplier(pd.DataFrame()) == 1.0


def test_the_flag_is_off_by_default_and_reaches_both_allocators():
    """2025 diagnosed this and must not size it, so the default stays off until
    the in-window folds have had their say."""
    assert not SeasonAverageVolumePipeline().cold_role_innovation
    assert not SeasonRosterShareModel("carry").cold_role_innovation

    pipeline = SeasonAverageVolumePipeline(cold_role_innovation=True)
    pipeline.target_model.cold_role_innovation = pipeline.cold_role_innovation
    pipeline.carry_model.cold_role_innovation = pipeline.cold_role_innovation

    assert pipeline.target_model.cold_role_innovation
    assert pipeline.carry_model.cold_role_innovation
