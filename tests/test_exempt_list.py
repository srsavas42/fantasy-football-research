"""The exempt-list model is small, and its failure modes are all quiet ones.

Censored episodes dropped or counted as complete both bias the hazard upward
and raise nothing. A one-week holdout pooled with a conduct placement halves
the estimate. A predictive draw that ignores the games actually left in the
season projects a player as missing more football than exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.features.exempt_list import ExemptListModel, exempt_episodes


def _rosters(rows: list[dict]) -> pd.DataFrame:
    # Season length is read off the weeks present in the frame, as it is in the
    # real feed. Without a player who lasts the season a synthetic frame would
    # silently describe a three-week season, and every "games missed" assertion
    # would be measured against the wrong horizon.
    seasons = {r.get("season", 2023) for r in rows}
    filler = [
        {"season": s, "week": w, "full_name": f"Filler{s}", "status": "ACT"}
        for s in seasons
        for w in range(1, 19)
    ]
    return pd.DataFrame(
        [
            {
                "season": r.get("season", 2023),
                "week": r["week"],
                "team": r.get("team", "AAA"),
                "position": r.get("position", "RB"),
                "full_name": r["full_name"],
                "status": r.get("status", "ACT"),
                "status_description_abbr": r.get("code", "A01"),
                "game_type": "REG",
            }
            for r in rows + filler
        ]
    )


def _episode(name: str, weeks: range, *, after: str | None = "ACT", season=2023, code="E02"):
    rows = [
        {"season": season, "week": w, "full_name": name, "status": "EXE", "code": code}
        for w in weeks
    ]
    if after is not None:
        rows += [
            {"season": season, "week": w, "full_name": name, "status": after}
            for w in range(weeks.stop, 19)
        ]
    return rows


def test_a_one_week_placement_is_filtered_out_by_default():
    """One-week placements are dominated by holdouts and un-retirements."""
    frame = _rosters(_episode("Blip", range(5, 6)) + _episode("Real", range(5, 9)))
    assert list(exempt_episodes(frame)["player_name"]) == ["Real"]
    assert set(exempt_episodes(frame, min_weeks=1)["player_name"]) == {"Blip", "Real"}


def test_the_covid_season_is_dropped_by_default():
    frame = _rosters(
        _episode("Covid", range(5, 9), season=2020)
        + _episode("Real", range(5, 9), season=2023)
    )
    assert list(exempt_episodes(frame)["player_name"]) == ["Real"]


def test_international_pathway_is_not_an_exempt_episode():
    frame = _rosters(_episode("Pathway", range(1, 6), code="E14"))
    assert exempt_episodes(frame).empty


def test_weeks_off_the_roster_count_as_games_missed():
    """An indefinite absence shows up in the feed mostly as absence."""
    frame = _rosters(_episode("Gone", range(1, 4), after=None))
    row = exempt_episodes(frame).iloc[0]
    assert row["exempt_weeks"] == 3
    assert row["games_missed"] == 18


def test_an_episode_running_to_the_final_week_is_censored():
    frame = _rosters(_episode("Open", range(15, 19), after=None))
    assert bool(exempt_episodes(frame).iloc[0]["censored"])


def test_a_conversion_to_suspension_is_recorded():
    frame = _rosters(
        _episode("Converted", range(1, 5), after=None)
        + [
            {"week": w, "full_name": "Converted", "status": "RES", "code": "R40"}
            for w in range(5, 11)
        ]
    )
    assert bool(exempt_episodes(frame).iloc[0]["converted_to_suspension"])


def _episodes(lengths, censored=None):
    censored = [False] * len(lengths) if censored is None else censored
    return pd.DataFrame(
        {
            "games_missed": lengths,
            "censored": censored,
            "weeks_remaining": [18] * len(lengths),
        }
    )


def test_censored_episodes_lower_the_hazard_rather_than_being_dropped():
    """A player still out in week 18 is evidence *against* quick resolution."""
    complete = ExemptListModel().fit(_episodes([4, 4, 4]))
    with_censored = ExemptListModel().fit(
        _episodes([4, 4, 4, 12], censored=[False, False, False, True])
    )
    assert with_censored.hazard_mean < complete.hazard_mean


def test_a_censored_episode_contributes_no_resolution():
    model = ExemptListModel().fit(_episodes([6], censored=[True]))
    assert model.posterior_events == model.prior_events
    assert model.posterior_survived == model.prior_survived + 6


def test_predictions_cannot_exceed_the_games_that_remain():
    model = ExemptListModel().fit(_episodes([4, 6, 8]))
    draws = model.predict_samples(size=5000, weeks_remaining=5, seed=1)
    assert draws.max() <= 5
    assert draws.min() >= 1


def test_the_prior_is_the_conduct_policy_baseline():
    """With no data the model returns the six-game policy anchor."""
    model = ExemptListModel()
    model.posterior_events = model.prior_events
    model.posterior_survived = model.prior_survived
    assert 1.0 / model.hazard_mean == pytest.approx(7.0)


def test_more_data_moves_the_estimate_off_the_prior():
    quick = ExemptListModel().fit(_episodes([1] * 30))
    slow = ExemptListModel().fit(_episodes([12] * 30))
    assert quick.hazard_mean > 0.5
    assert slow.hazard_mean < 0.15


def test_fitting_nothing_is_an_error_not_a_prior_dressed_as_a_fit():
    with pytest.raises(ValueError, match="no exempt episodes"):
        ExemptListModel().fit(_episodes([]))
