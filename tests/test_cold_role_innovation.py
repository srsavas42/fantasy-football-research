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


def test_the_promoted_default_is_measured_mode():
    """Promoted 2026-08-04 on three in-window folds, confirmed once on 2025.

    Pinned because the mode is what the evidence is about: the relative mode
    was measured too and recovered a seventh of the deficit where this one
    closes it.
    """
    pipeline = SeasonAverageVolumePipeline()

    assert pipeline.cold_role_innovation
    assert pipeline.cold_role_scale_mode == "measured"


def test_the_flag_reaches_both_allocators():
    pipeline = SeasonAverageVolumePipeline(cold_role_innovation=True)
    pipeline.target_model.cold_role_innovation = pipeline.cold_role_innovation
    pipeline.carry_model.cold_role_innovation = pipeline.cold_role_innovation

    assert pipeline.target_model.cold_role_innovation
    assert pipeline.carry_model.cold_role_innovation


def test_measured_mode_targets_the_cold_population_rather_than_the_ratio():
    """Why a second mode exists.

    The relative mode preserves the gap between the populations and inherits
    the cap's compression with it: the base is capped from 1.94 to 0.25, so a
    1.38x ratio puts cold rows at 0.35 against a measured 2.68. Measured mode
    targets that 2.68 directly, so the cap bounds the typical row without also
    bounding the row it was never about.
    """
    model = SeasonRosterShareModel(
        "carry", cold_role_innovation=True, cold_role_scale_mode="measured"
    )
    model.role_innovation_scale = 0.25
    model._cold_and_warm_dispersion = lambda d: (2.678, 1.936)

    multiplier = model._fit_cold_role_multiplier(pd.DataFrame(), None, None)

    # 2.678 / 0.25 is over ten, so the multiplier cap is what binds here.
    assert multiplier == pytest.approx(model.cold_role_multiplier_cap)


def test_measured_mode_still_floors_at_one():
    model = SeasonRosterShareModel(
        "carry", cold_role_innovation=True, cold_role_scale_mode="measured"
    )
    model.role_innovation_scale = 0.25
    model._cold_and_warm_dispersion = lambda d: (0.05, 1.936)

    assert model._fit_cold_role_multiplier(pd.DataFrame(), None, None) == 1.0


def test_an_unknown_mode_is_refused_rather_than_silently_relative():
    model = SeasonRosterShareModel(
        "carry", cold_role_innovation=True, cold_role_scale_mode="mesured"
    )

    with pytest.raises(ValueError, match="unknown cold_role_scale_mode"):
        model._fit_cold_role_multiplier(pd.DataFrame(), None, None)


def test_the_mode_reaches_both_allocators_and_survives_a_round_trip():
    pipeline = SeasonAverageVolumePipeline(
        cold_role_innovation=True, cold_role_scale_mode="measured"
    )
    for model in (pipeline.target_model, pipeline.carry_model):
        model.cold_role_innovation = True
        model.cold_role_scale_mode = pipeline.cold_role_scale_mode

    assert pipeline.carry_model.cold_role_scale_mode == "measured"
    assert pipeline.target_model.cold_role_scale_mode == "measured"


def test_a_missing_prior_snap_share_is_refused_when_the_widening_is_on():
    """The silent-degradation case.

    Without the column the mask falls back to "no role in this stream", which
    is a deliberately rejected population -- 62% of carry rows rather than 34%,
    including receivers whose zero carries the model already predicts well.
    Nothing validated that configuration and it would otherwise run without a
    word.
    """
    rows = _room().drop(columns=["prior_snap_share"])
    model = SeasonRosterShareModel("carry", cold_role_innovation=True)

    with pytest.raises(ValueError, match="prior_snap_share"):
        model._cold_role_rows(rows)


def test_a_missing_prior_snap_share_is_tolerated_when_the_widening_is_off():
    """``_design`` builds the mask on every prediction, so refusing it
    unconditionally would break frames that never asked for this feature."""
    rows = _room().drop(columns=["prior_snap_share"])
    model = SeasonRosterShareModel("carry", cold_role_innovation=False)

    assert model._cold_role_rows(rows).tolist() == [False, False, True, True, True]
