"""Cross-positional teammate quality: who is throwing the ball.

Every efficiency spec has an empty feature list, so receiver efficiency is
modelled from the receiver's own history and nothing about his quarterback.
``prior_rec_team_quality_signal`` does not fill the gap -- it ranks a player's
own quality within his team, which is relative standing, not teammate quality.
"""

import numpy as np
import pandas as pd
import pytest

from ffmodel.features.season_efficiency import add_teammate_quality_features


def _roster():
    """Two teams. KC lists its starter; BUF lists nobody and needs the fallback."""
    return pd.DataFrame(
        {
            "season": 2024,
            "team": ["KC", "KC", "KC", "BUF", "BUF", "BUF"],
            "position": ["QB", "QB", "WR", "QB", "QB", "RB"],
            "player_key": ["kc-qb1", "kc-qb2", "kc-wr1", "buf-qb1", "buf-qb2", "buf-rb1"],
            "qb_listed_starter": [1, 0, np.nan, np.nan, np.nan, np.nan],
            "qb_depth_rank": [1.0, 2.0, np.nan, 2.0, 1.0, np.nan],
            "prior_pass_quality_signal": [0.40, -0.30, np.nan, -0.25, 0.15, np.nan],
        }
    )


def test_skill_players_inherit_the_listed_starter():
    out = add_teammate_quality_features(_roster())

    receiver = out.loc[out.player_key.eq("kc-wr1"), "teammate_qb_quality_signal"]
    assert receiver.iloc[0] == pytest.approx(0.40)


def test_the_fallback_is_the_depth_chart_when_nobody_is_listed():
    """BUF lists no starter, and its shallower entry is qb2, not qb1."""
    out = add_teammate_quality_features(_roster())

    back = out.loc[out.player_key.eq("buf-rb1"), "teammate_qb_quality_signal"]
    assert back.iloc[0] == pytest.approx(0.15)


def test_a_quarterback_does_not_read_his_own_quality_as_a_teammates():
    """Otherwise one signal enters the model twice under two names."""
    out = add_teammate_quality_features(_roster())

    assert out.loc[out.position.eq("QB"), "teammate_qb_quality_signal"].isna().all()


def test_it_refuses_to_run_before_the_quality_composite_exists():
    rows = _roster().drop(columns=["prior_pass_quality_signal"])

    with pytest.raises(ValueError, match="prior_pass_quality_signal"):
        add_teammate_quality_features(rows)


def test_it_ignores_primary_qb_entirely():
    """The listed starter and depth rank are preseason artifacts. ``primary_qb``
    is not -- it is derived from who actually played -- so a feature reading it
    would score well in validation and be unavailable when serving.

    Checked by behaviour rather than by grepping the source: flipping
    ``primary_qb`` to contradict the depth chart must change nothing.
    """
    rows = _roster()
    honest = add_teammate_quality_features(rows.assign(primary_qb=[1, 0, 0, 0, 1, 0]))
    inverted = add_teammate_quality_features(rows.assign(primary_qb=[0, 1, 0, 1, 0, 0]))

    pd.testing.assert_series_equal(
        honest["teammate_qb_quality_signal"], inverted["teammate_qb_quality_signal"]
    )
