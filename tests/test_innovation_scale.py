"""The innovation scale is measured on one side of the softmax and used on the other.

``_estimate_role_innovation`` measures the RMS log-share deviation between
realized and allocated shares — the dispersion that actually came out. That
number is then handed to the sampler as the standard deviation of noise added to
``eta``, on the *input* side of the softmax. Renormalization compresses, so the
spread the model produces is smaller than the spread it measured, by a factor
that depends on how many players share the room.

The pipeline therefore realizes 70–93% of the churn it observed, and is most
under-dispersed where rooms are smallest — at quarterback, where workload-share
coverage runs 0.647 / 0.619 / 0.726 against an 80% nominal interval.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.models.base import (
    calibrate_innovation_scale,
    realized_share_dispersion,
)


def _rooms(prior, copies=1):
    allocation = np.tile(np.asarray(prior, dtype=float), (copies, 1))
    return allocation, np.ones(allocation.shape, dtype=bool)


@pytest.mark.parametrize(
    "prior, ratio",
    [
        ([0.65, 0.35], 0.71),
        ([0.90, 0.08, 0.02], 0.82),
        ([0.26, 0.22, 0.18, 0.14, 0.10, 0.06, 0.04], 0.92),
    ],
)
def test_the_softmax_delivers_less_spread_than_it_is_given(prior, ratio):
    allocation, mask = _rooms(prior)

    realized = realized_share_dispersion(allocation, mask, 0.60, draws=4000)

    assert realized == pytest.approx(0.60 * ratio, abs=0.01)
    assert realized < 0.60


def test_the_shortfall_is_worst_in_the_smallest_room():
    """Which is why this shows up at quarterback before anywhere else."""
    two, two_mask = _rooms([0.65, 0.35])
    seven, seven_mask = _rooms([0.26, 0.22, 0.18, 0.14, 0.10, 0.06, 0.04])

    small = realized_share_dispersion(two, two_mask, 0.60, draws=4000)
    large = realized_share_dispersion(seven, seven_mask, 0.60, draws=4000)

    assert small < large < 0.60


@pytest.mark.parametrize("target", [0.30, 0.45, 0.60, 0.75])
def test_calibration_recovers_the_measured_dispersion(target):
    allocation, mask = _rooms([0.90, 0.08, 0.02])

    scale = calibrate_innovation_scale(allocation, mask, target, draws=4000)
    achieved = realized_share_dispersion(allocation, mask, scale, draws=4000, seed=7)

    assert scale > target  # the input has to exceed what should come out
    assert achieved == pytest.approx(target, rel=0.03)


def test_calibration_respects_the_mix_of_room_sizes():
    """A single global factor would be wrong; the correction is per-population.

    A league of two-man rooms needs a larger input scale than a league of
    seven-deep rooms to realize the same churn.
    """
    two, two_mask = _rooms([0.65, 0.35])
    seven, seven_mask = _rooms([0.26, 0.22, 0.18, 0.14, 0.10, 0.06, 0.04])

    small = calibrate_innovation_scale(two, two_mask, 0.60, draws=4000)
    large = calibrate_innovation_scale(seven, seven_mask, 0.60, draws=4000)

    assert small > large


def test_ragged_rooms_only_count_their_live_slots():
    padded = np.array([[0.9, 0.08, 0.02], [0.6, 0.4, 0.0]], dtype=float)
    mask = np.array([[True, True, True], [True, True, False]])

    realized = realized_share_dispersion(padded, mask, 0.60, draws=2000)
    two_only = realized_share_dispersion(
        np.array([[0.6, 0.4]]), np.array([[True, True]]), 0.60, draws=2000
    )
    three_only = realized_share_dispersion(
        np.array([[0.9, 0.08, 0.02]]), np.array([[True, True, True]]), 0.60, draws=2000
    )

    # Between the two, since rows are weighted by their live slot count.
    assert min(two_only, three_only) < realized < max(two_only, three_only)


def test_a_zero_target_calibrates_to_zero():
    allocation, mask = _rooms([0.5, 0.5])

    assert calibrate_innovation_scale(allocation, mask, 0.0, draws=200) == 0.0


def test_a_single_player_room_has_no_dispersion_to_lose():
    allocation = np.array([[1.0]])
    mask = np.array([[True]])

    assert realized_share_dispersion(allocation, mask, 0.60, draws=500) == 0.0


def _target_rows():
    """One team-season with three receivers and a quarterback."""
    return pd.DataFrame(
        {
            "season": 2023,
            "team": "KC",
            "player_key": ["wr1", "wr2", "te1", "qb1"],
            "player_name": ["WR One", "WR Two", "TE One", "QB One"],
            "position": ["WR", "WR", "TE", "QB"],
            "targets": [120.0, 80.0, 60.0, 0.0],
            "rush_att": [2.0, 1.0, 0.0, 30.0],
            "prior_target_role": [0.30, 0.20, 0.15, 0.0],
            "prior_target_per_snap": [0.22, 0.16, 0.12, 0.0],
            "draft_target_prior": [0.20, 0.15, 0.10, 0.0],
            "prior_carry_role": [0.01, 0.01, 0.0, 0.08],
            "prior_carry_per_snap": [0.01, 0.01, 0.0, 0.06],
            "draft_carry_prior": [0.01, 0.01, 0.0, 0.05],
            "snap_share": [0.85, 0.70, 0.60, 0.99],
            "observed_availability": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_the_target_estimator_excludes_quarterbacks_like_the_likelihood_does():
    """C6. ``_design`` masks QBs out of the target support; this did not.

    Leaving them in put a player the model never allocates to into both the
    softmax denominator and the observed-share normalization, so the scale
    described a room that is not the room being fitted.
    """
    from ffmodel.models.volume_season_average import SeasonRosterShareModel

    model = SeasonRosterShareModel("target")
    rows = _target_rows()

    support = model._innovation_support(rows)

    assert support.tolist() == [True, True, True, False]

    allocation, mask = model._innovation_rooms(rows)
    assert mask.sum() == 3  # the quarterback is not a slot in the target room
    assert allocation[mask].sum() == pytest.approx(1.0)


def test_the_carry_estimator_keeps_every_position():
    from ffmodel.models.volume_season_average import SeasonRosterShareModel

    model = SeasonRosterShareModel("carry")

    assert model._innovation_support(_target_rows()).all()


def test_the_pipeline_flag_reaches_the_three_allocation_layers():
    from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

    pipeline = SeasonAverageVolumePipeline(calibrated_innovation=True)
    pipeline._enable_calibrated_innovation()

    assert pipeline.workload_model.calibrated_innovation
    assert pipeline.target_model.calibrated_innovation
    assert pipeline.carry_model.calibrated_innovation
    # Per-player layers never renormalize, so no dispersion is lost there.
    assert not hasattr(pipeline.snap_model, "calibrated_innovation")
