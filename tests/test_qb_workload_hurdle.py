"""The QB workload softmax and its hurdle are one factorisation, not two models.

The room is close to winner-take-all, so the hurdle exists to let a backup's
season be an exact zero. But a softmax fitted over *every* quarterback already
has the marginal share as its mean — the zero seasons are in the likelihood as
observed zero counts. Gating that mean at prediction time and renormalising
therefore moves it, taking expected volume off backups and giving it to
starters. The softmax has to be fitted over the gated-in room so that it means
the share *conditional* on playing, which is the thing a gate multiplies back
into a marginal.
"""

import numpy as np
import pandas as pd
import pytest

az = pytest.importorskip("arviz")

from ffmodel.models.season_availability import QBWorkloadShareModel


def _room(team: str, snaps: tuple[float, ...]) -> list[dict]:
    return [
        {
            "season": 2024,
            "team": team,
            "player_key": f"{team}-qb{index + 1}",
            "player_name": f"{team} QB{index + 1}",
            "position": "QB",
            "offense_snaps": value,
            "snap_counts_observed": 1,
            "pass_att": value * 0.6,
            "team_games": 17,
            "games": 17,
            "observed_availability": min(1.0, value / 1000.0 + 0.05),
            "prior_availability": 0.85,
            "prior_qb_snap_share": [0.85, 0.12, 0.03][index],
            "prior_pass_role": np.nan,
            "draft_pass_prior": 0.0,
            "age": 27,
            "experience": 5,
            "team_change": 0,
            "cold_start": 0,
            "roster_active": 1,
            "roster_reserve": 0,
            "depth_rank": index + 1,
            "qb_depth_rank": index + 1,
            "qb_listed_starter": int(index == 0),
            "is_replacement_qb": 0,
            "is_replacement_player": 0,
        }
        for index, value in enumerate(snaps)
    ]


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        _room("A", (950.0, 12.0, 0.0))       # starter plays; backup below the bar
        + _room("B", (300.0, 700.0, 0.0))    # starter hurt; backup takes the room
        + _room("C", (10.0, 12.0, 0.0))      # nobody clears the bar
    )


def _conditional(model: QBWorkloadShareModel):
    return model._conditional_design(model._design(_rows(), fit=True))


def test_share_fit_excludes_quarterbacks_below_the_hurdle():
    model = QBWorkloadShareModel(hurdle_min_attempts=25)
    conditional = _conditional(model)

    # Every count entering the Multinomial cleared the bar, so the softmax it
    # identifies is the conditional share. A sub-threshold count left in would
    # make the fitted mean marginal and the later gate double-count it.
    counts = conditional["counts"]
    support = conditional["support"]
    assert ((counts >= model.hurdle_min_attempts) | ~support).all()
    assert (counts[~support] == 0).all()


def test_rooms_where_nobody_cleared_the_bar_leave_the_share_fit_but_label_the_hurdle():
    model = QBWorkloadShareModel(hurdle_min_attempts=25)
    conditional = _conditional(model)

    # Rooms A and B each have a passer over the bar; room C has none, and a room
    # with no conditional observation cannot inform a conditional share.
    assert conditional["counts"].shape[0] == 2
    # It is still evidence about *whether* a passer clears the bar, so all nine
    # active quarterbacks label the hurdle.
    assert conditional["hurdle_y"].shape[0] == 9
    assert conditional["hurdle_y"].sum() == 3


def test_an_always_open_gate_reproduces_the_ungated_prediction():
    # The gate is what converts the conditional softmax back to a marginal. When
    # it never closes, the conversion is the identity, which pins the arithmetic
    # independently of whatever the hurdle regression happens to learn.
    rows = _rows()
    draws = 8

    def predict(with_hurdle: bool):
        model = QBWorkloadShareModel(role_innovation_scale=0.0)
        model._design(rows, fit=True)
        model.role_innovation_scale = 0.0
        posterior = {"beta": np.zeros((1, draws, len(model.feature_names)))}
        if with_hurdle:
            posterior["hurdle_intercept"] = np.full((1, draws), 40.0)
            posterior["hurdle_beta"] = np.zeros(
                (1, draws, len(model.feature_names))
            )
        model.idata = az.from_dict(posterior=posterior)
        return model.predict_share_samples(rows, seed=5).shares

    assert np.allclose(predict(True), predict(False))


def test_role_innovation_ignores_passers_outside_the_conditional_room():
    # The dispersion term describes spread *within* the room that plays, so a
    # third-stringer who never cleared the bar must not move it. Leaving him in
    # charges his near-zero share to dispersion as well as to the gate, which is
    # the same double count the conditional fit exists to avoid.
    two = pd.DataFrame(_room("A", (950.0, 700.0)))
    plus_spectator = pd.DataFrame(_room("A", (950.0, 700.0, 4.0)))

    def innovation(rows: pd.DataFrame) -> float:
        model = QBWorkloadShareModel(hurdle_min_attempts=25)
        model._design(rows, fit=True)
        return model.role_innovation_scale

    assert innovation(two) == pytest.approx(innovation(plus_spectator))


def _hand_fitted(coupled: bool, draws: int = 6) -> QBWorkloadShareModel:
    """A model with a known posterior, so the gate's response is exact."""
    model = QBWorkloadShareModel(
        role_innovation_scale=0.0, couple_gate_to_availability=coupled
    )
    model._design(_rows(), fit=True)
    model.role_innovation_scale = 0.0
    model.hurdle_availability_mean = 0.0
    model.hurdle_availability_scale = 1.0
    posterior = {
        "beta": np.zeros((1, draws, len(model.feature_names))),
        "hurdle_intercept": np.zeros((1, draws)),
        "hurdle_beta": np.zeros((1, draws, len(model.feature_names))),
    }
    if coupled:
        # Positive loading: more availability, more likely to clear the hurdle.
        posterior["hurdle_availability_beta"] = np.full((1, draws), 3.0)
    model.idata = az.from_dict(posterior=posterior)
    return model


def _mean_share(model: QBWorkloadShareModel, availability: float) -> np.ndarray:
    rows = _rows()
    draws = model.idata.posterior.sizes["draw"]
    samples = np.full((len(model._design(rows)["rows"]), draws), availability)
    return model.predict_share_samples(
        rows, availability_samples=samples, seed=5
    ).shares.mean(axis=1)


def test_coupling_is_off_by_default_and_leaves_the_posterior_unchanged():
    # The candidate must not alter the accepted architecture until it clears a
    # validation gate, so an un-opted-in fit carries no availability term.
    assert QBWorkloadShareModel().couple_gate_to_availability is False
    assert "hurdle_availability_beta" not in _hand_fitted(False).idata.posterior


def test_an_uncoupled_gate_ignores_the_availability_draw():
    # This is the incoherence: the softmax offset moves with availability while
    # the gate is drawn from the same probability regardless, so within one draw
    # "available all season" can be paired with a closed gate.
    model = _hand_fitted(False)

    low = _mean_share(model, 0.10)
    high = _mean_share(model, 0.95)

    # Shares renormalise within the room, so a uniform availability shift that
    # the gate ignores leaves every share where it was.
    assert np.allclose(low, high)


def test_a_coupled_gate_responds_to_the_availability_draw():
    model = _hand_fitted(True)
    rows = model._design(_rows())["rows"]
    third = rows["player_key"].str.endswith("qb3").to_numpy()

    low = _mean_share(model, 0.10)
    high = _mean_share(model, 0.95)

    # With a positive loading, higher availability opens more gates, so the room
    # is shared more widely and the deepest passer picks up more of it.
    assert high[third].sum() > low[third].sum()
