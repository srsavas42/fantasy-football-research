"""The cold-start role prior has to be a per-snap rate, on one basis.

``_role_prior`` blends ``prior_*_per_snap`` with the lagged share and then clips
to ``[1e-5, 1.0]``. Any fallback fed into that blend must therefore already be a
per-snap rate. Deriving it by dividing a season count by an availability
*fraction* for the rows without snap observations mixes two units three orders of
magnitude apart into one average: the estimate inflates, it inflates differently
per position (so the roster softmax cannot normalise it away), and the clip then
flattens every position onto the same saturated value — at which point a
cold-start receiver outranks an established back in the carry allocation.
"""

import numpy as np
import pandas as pd

from ffmodel.models.volume_season_average import SeasonRosterShareModel


def _rows() -> pd.DataFrame:
    """Two rooms: players with snap observations, plus snapless reserves."""
    records = []
    for team in ("A", "B"):
        records += [
            # Established players, snaps observed.
            dict(player_key=f"{team}-rb1", position="RB", offense_snaps=600.0,
                 rush_att=240.0, targets=60.0, pass_att=0.0, cold_start=0),
            dict(player_key=f"{team}-wr1", position="WR", offense_snaps=900.0,
                 rush_att=4.0, targets=140.0, pass_att=0.0, cold_start=0),
            dict(player_key=f"{team}-te1", position="TE", offense_snaps=700.0,
                 rush_att=1.0, targets=80.0, pass_att=0.0, cold_start=0),
            dict(player_key=f"{team}-qb1", position="QB", offense_snaps=1000.0,
                 rush_att=50.0, targets=0.0, pass_att=560.0, cold_start=0),
            # Cold-start players who produced volume but are absent from the
            # snap feed. These are the rows that supplied the availability
            # -fraction branch: a real count over a fraction at most one, which
            # is what drove the estimate orders of magnitude past a per-snap
            # rate. They are common in practice — the legacy snap files do not
            # cover every player who recorded a stat line.
            dict(player_key=f"{team}-wr9", position="WR", offense_snaps=0.0,
                 rush_att=3.0, targets=45.0, pass_att=0.0, cold_start=1),
            dict(player_key=f"{team}-te9", position="TE", offense_snaps=0.0,
                 rush_att=1.0, targets=25.0, pass_att=0.0, cold_start=1),
            dict(player_key=f"{team}-rb9", position="RB", offense_snaps=0.0,
                 rush_att=70.0, targets=15.0, pass_att=0.0, cold_start=1),
        ]
    frame = pd.DataFrame(records)
    frame["season"] = 2023
    frame["team"] = [key.split("-")[0] for key in frame["player_key"]]
    frame["player_name"] = frame["player_key"]
    frame["observed_availability"] = 0.5
    frame["snap_share"] = frame["offense_snaps"] / 1000.0
    frame["prior_availability"] = 0.8
    for column in ("prior_target_per_snap", "prior_carry_per_snap",
                   "prior_qb_attempts_per_snap", "prior_target_role",
                   "prior_carry_role", "prior_pass_role", "draft_target_prior",
                   "draft_carry_prior", "draft_pass_prior"):
        frame[column] = np.nan
    return frame


def _fitted(stream: str) -> SeasonRosterShareModel:
    model = SeasonRosterShareModel(stream)
    model._fit_metadata(model._prepare(_rows()))
    return model


def test_cold_role_prior_stays_a_per_snap_rate():
    # A per-snap rate cannot exceed one opportunity per snap. Before the fix
    # these ran to 7.4 (targets) and 38.1 (carries) on real data.
    for stream in ("pass", "target", "carry"):
        for position, value in _fitted(stream).cold_role_prior.items():
            assert 0.0 < value <= 1.0, (stream, position, value)


def test_cold_role_prior_is_not_saturated_by_the_role_prior_clip():
    # Distinct positions must stay distinguishable after ``_role_prior`` clips.
    # Mixed units pushed every position above 1.0, so the clip erased the
    # position ordering entirely and returned the same number for all of them.
    carry = _fitted("carry").cold_role_prior
    assert carry["RB"] > carry["WR"] > carry["TE"]
    assert carry["RB"] < 1.0


def test_cold_start_receiver_does_not_outrank_an_established_back_on_carries():
    model = _fitted("carry")
    rows = model._prepare(_rows())
    prior = model._role_prior(rows)

    back = prior[rows["player_key"].eq("A-rb1").to_numpy()][0]
    cold_receiver = prior[rows["player_key"].eq("A-wr9").to_numpy()][0]

    assert cold_receiver < back
    # No row may sit on the upper clip; that is the signature of the unit bug.
    assert not (prior >= 1.0).any()


def test_availability_prior_is_read_from_prior_availability():
    # The position loop used to rebind the name it had just used to build the
    # exposure, which made this easy to break silently.
    for position, value in _fitted("target").availability_prior.items():
        assert value == 0.8, (position, value)
