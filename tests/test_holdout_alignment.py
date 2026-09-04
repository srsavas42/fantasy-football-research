"""Holdout alignment: keep the projections that failed, drop the ones that were never people.

An inner join to realized stat rows discards every projected player who never
recorded one — precisely the role-collapse cases a preseason projection most
needs to be graded on.
"""

import pandas as pd
import pytest

from ffmodel.evaluation.holdout_alignment import (
    align_projection_to_outcomes,
    real_player_mask,
)


def _projected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": ["played", "never-played", "__replacement_qb__ARI"],
            "team": ["ARI", "ARI", "ARI"],
            "position": ["WR", "QB", "QB"],
            "is_replacement_player": [0, 0, 1],
        }
    )


def _realized() -> pd.DataFrame:
    return pd.DataFrame({"player_key": ["played"], "team": ["ARI"], "actual": [120.0]})


def test_a_projected_player_who_never_appeared_scores_zero():
    aligned = align_projection_to_outcomes(_projected(), _realized())

    outcomes = dict(zip(aligned["player_key"], aligned["actual"]))
    assert outcomes["played"] == 120.0
    # Produced nothing, which is an outcome rather than a missing observation.
    assert outcomes["never-played"] == 0.0


def test_synthetic_replacement_buckets_are_excluded():
    aligned = align_projection_to_outcomes(_projected(), _realized())

    # A modelling device has no realized counterpart to score.
    assert "__replacement_qb__ARI" not in set(aligned["player_key"])
    assert len(aligned) == 2


def test_imputed_zeros_are_distinguishable_from_observed_ones():
    realized = pd.concat(
        [_realized(), pd.DataFrame({"player_key": ["never-played"], "team": ["ARI"], "actual": [0.0]})]
    )

    both = align_projection_to_outcomes(_projected(), realized)
    imputed = align_projection_to_outcomes(_projected(), _realized())

    # Same outcome, different provenance: one was measured at zero, one absent.
    assert both.set_index("player_key").loc["never-played", "realized_row"]
    assert not imputed.set_index("player_key").loc["never-played", "realized_row"]


def test_sample_index_survives_the_filtering():
    aligned = align_projection_to_outcomes(_projected(), _realized())

    # Indices must still address the original posterior sample matrix.
    assert sorted(aligned["sample_index"]) == [0, 1]


def test_replacement_mask_handles_a_frame_without_flags():
    rows = pd.DataFrame({"player_key": ["a", "b"], "team": ["X", "Y"]})

    assert real_player_mask(rows).all()


def test_missing_join_key_is_rejected():
    with pytest.raises(ValueError, match="join keys"):
        align_projection_to_outcomes(
            pd.DataFrame({"team": ["ARI"]}), _realized()
        )


def test_missing_outcome_column_is_rejected():
    with pytest.raises(ValueError, match="actual"):
        align_projection_to_outcomes(
            _projected(), pd.DataFrame({"player_key": ["played"], "team": ["ARI"]})
        )
