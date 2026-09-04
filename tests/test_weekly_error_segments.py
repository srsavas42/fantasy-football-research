"""The diagnostic's segment definitions, which are easy to get quietly wrong.

An error attribution is only worth reading if the segments mean what they say.
Both functions here encode a specific claim -- "this team's established lead back
was out" and "this is the week his role stepped up" -- and a subtly wrong version
of either produces a confident table describing something else entirely.

The lead-back rule is the one with a trap in it. It must identify the lead from
*lagged* share, so the label does not depend on the week being scored, while
reading whether he played *this* week, which is the fact the model is blind to.
Reading the current week's carries to pick the lead would define the segment by
its own outcome: a starter who sat out has no carries, so somebody else becomes
"the lead" and the segment empties itself.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_weekly_errors import (
    lead_back_out,
    previous_team,
    role_change_profile,
)


def _row(week, key, team, share, played, position="RB", **kwargs):
    row = {
        "season": 2023,
        "week": week,
        "team": team,
        "player_key": key,
        "position": position,
        "played": played,
        "prior_rush_share_recent": share,
        "prior_target_share_recent": 0.0,
        "rush_att": kwargs.pop("rush_att", 0.0),
        "targets": kwargs.pop("targets", 0.0),
        "team_rush_att": kwargs.pop("team_rush_att", 30.0),
        "team_targets": kwargs.pop("team_targets", 30.0),
        "points": kwargs.pop("points", 0.0),
    }
    row.update(kwargs)
    return row


def test_the_backup_is_flagged_when_the_lead_sits() -> None:
    frame = pd.DataFrame(
        [
            _row(1, "starter", "ATL", 0.70, 0),
            _row(1, "backup", "ATL", 0.20, 1),
        ]
    )
    got = lead_back_out(frame).to_numpy()
    # The backup is flagged; the absent starter is not flagged against himself.
    assert list(got) == [False, True]


def test_nobody_is_flagged_when_the_lead_plays() -> None:
    frame = pd.DataFrame(
        [
            _row(1, "starter", "ATL", 0.70, 1),
            _row(1, "backup", "ATL", 0.20, 1),
        ]
    )
    assert not lead_back_out(frame).any()


def test_a_committee_has_no_lead_to_lose() -> None:
    """Below the share threshold nobody is established, so nobody inherits."""
    frame = pd.DataFrame(
        [
            _row(1, "a", "ATL", 0.30, 0),
            _row(1, "b", "ATL", 0.28, 1),
        ]
    )
    assert not lead_back_out(frame).any()


def test_the_lead_is_taken_from_history_not_from_this_week() -> None:
    """The trap: picking the lead by current carries empties the segment.

    The starter sat out, so he has zero carries this week. A rule that read
    current volume would crown the backup as the lead and flag nobody.
    """
    frame = pd.DataFrame(
        [
            _row(1, "starter", "ATL", 0.70, 0, rush_att=0.0),
            _row(1, "backup", "ATL", 0.20, 1, rush_att=22.0),
        ]
    )
    assert lead_back_out(frame).sum() == 1


def test_only_the_same_team_is_affected() -> None:
    frame = pd.DataFrame(
        [
            _row(1, "starter", "ATL", 0.70, 0),
            _row(1, "backup", "ATL", 0.20, 1),
            _row(1, "other", "BUF", 0.15, 1),
        ]
    )
    got = lead_back_out(frame)
    assert got.to_numpy()[2] == False  # noqa: E712 - explicit about the value


def test_previous_team_looks_back_exactly_one_season() -> None:
    frame = pd.DataFrame(
        [
            {"player_key": "p", "season": 2022, "week": 17, "team": "ATL"},
            {"player_key": "p", "season": 2023, "week": 1, "team": "BUF"},
        ]
    )
    got = previous_team(frame)
    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == "ATL"


def test_role_change_profile_starts_at_the_promotion() -> None:
    """Offsets are counted from the first qualifying step-up, not from week 1."""
    rows = []
    for week in range(1, 7):
        # Backup through week 3, then a real role from week 4 on.
        promoted = week >= 4
        rows.append(
            _row(
                week,
                "p",
                "ATL",
                0.05,
                1,
                rush_att=18.0 if promoted else 1.0,
                team_rush_att=30.0,
                points=18.0 if promoted else 2.0,
            )
        )
    frame = pd.DataFrame(rows)
    observed = frame["points"].to_numpy(float)
    # A model that always says 5 points: badly low once the role opens.
    model = np.full((len(frame), 8), 5.0)

    got = role_change_profile(frame, observed, model).set_index("weeks_since")
    assert got.loc[0, "n"] == 1
    assert got.loc[0, "observed"] == pytest.approx(18.0)
    # Projected 5 against an observed 18 is a bias of -13.
    assert got.loc[0, "bias"] == pytest.approx(-13.0)
    # Three qualifying weeks follow (4, 5, 6) -> offsets 0, 1, 2.
    assert sorted(got.index.tolist()) == [0, 1, 2]


def test_a_one_week_spike_in_no_role_is_not_a_promotion() -> None:
    """Twenty percent of a team's carries is the floor for calling it a role."""
    rows = [
        _row(week, "p", "ATL", 0.02, 1, rush_att=1.0, team_rush_att=30.0, points=1.0)
        for week in range(1, 5)
    ]
    # One week at 4 of 30 carries: above the +10pt delta but under the 20% floor.
    rows[2]["rush_att"] = 4.0
    frame = pd.DataFrame(rows)
    got = role_change_profile(
        frame, frame["points"].to_numpy(float), np.full((len(frame), 4), 3.0)
    )
    assert got.empty
