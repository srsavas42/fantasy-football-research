"""The competition-weighted history: the identity, the algebra, and the switch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ffmodel.weekly.competition import DEFAULT_KAPPA, WEIGHT_COLUMN, attach_competition
from ffmodel.weekly.features import _prior

ALPHA = 1.0 - 0.5 ** (1.0 / 4.0)


def _frame(n: int = 12, players: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_key": [p for p in players for _ in range(n)],
            "week": list(range(n)) * len(players),
        }
    )


def _brute(values: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    """The weighted exponential average, written out one row at a time."""
    out = np.full(len(values), np.nan)
    for t in range(len(values)):
        seen = [i for i in range(t) if not np.isnan(values[i])]
        if not seen:
            continue
        decay = np.array([(1.0 - alpha) ** (t - 1 - i) for i in seen])
        w = decay * weights[seen]
        out[t] = float((w * values[seen]).sum() / w.sum())
    return out


@pytest.mark.parametrize("constant", [1.0, 0.25, 6.0])
def test_a_uniform_weight_is_the_identity(constant):
    """Only relative weights survive the ratio, so a flat weight changes nothing.

    This is what leaves a pure handcuff -- every one of whose weeks came with the
    starter hurt -- exactly where he was, and it is why the arm can be switched
    off by a value rather than by a branch.
    """
    frame = _frame()
    values = pd.Series(np.random.default_rng(0).normal(10.0, 4.0, len(frame)))
    plain = _prior(frame, ["player_key"], values, how="ewm", alpha=ALPHA)
    weighted = _prior(
        frame, ["player_key"], values, how="ewm", alpha=ALPHA,
        weights=pd.Series(constant, index=values.index),
    )
    assert np.allclose(plain.dropna(), weighted.dropna(), atol=1e-12)


def test_the_weighted_average_matches_the_definition():
    rng = np.random.default_rng(7)
    frame = _frame()
    values = pd.Series(rng.normal(10.0, 4.0, len(frame)))
    values[[2, 9, 15]] = np.nan  # weeks he did not play
    weights = pd.Series(rng.choice([1.0, 0.5], len(frame)), index=values.index)

    got = _prior(
        frame, ["player_key"], values, how="ewm", alpha=ALPHA, weights=weights
    ).to_numpy(float)
    half = len(frame) // 2
    want = np.concatenate(
        [
            _brute(values.to_numpy()[:half], weights.to_numpy()[:half], ALPHA),
            _brute(values.to_numpy()[half:], weights.to_numpy()[half:], ALPHA),
        ]
    )
    assert np.allclose(got[~np.isnan(want)], want[~np.isnan(want)], atol=1e-10)


def test_a_missing_week_spends_no_denominator():
    """The mask comes from the values, not the weights.

    Were it otherwise a week with no observation would still consume weight and
    pull every average toward zero, which is the one way this identity can be
    got wrong without failing loudly.
    """
    frame = _frame(n=6, players=("a",))
    values = pd.Series([np.nan, 5.0, np.nan, 5.0, 5.0, np.nan])
    weights = pd.Series([9.0, 1.0, 9.0, 1.0, 1.0, 9.0])
    got = _prior(
        frame, ["player_key"], values, how="ewm", alpha=ALPHA, weights=weights
    )
    # Every observed value is 5, so every average of them must be 5 regardless of
    # what weight the unobserved weeks carry.
    assert np.allclose(got.dropna(), 5.0)


def test_weights_are_refused_on_the_unweighted_statistics():
    frame = _frame(n=4, players=("a",))
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="only defined for an ewm"):
        _prior(
            frame, ["player_key"], values, how="mean",
            weights=pd.Series(1.0, index=values.index),
        )


def test_attach_competition_writes_a_weight_per_row():
    panel = pd.DataFrame(
        {
            "season": [2022] * 4,
            "week": [1, 2, 1, 2],
            "team": ["KC"] * 4,
            "position": ["RB"] * 4,
            "player_key": ["p1", "p1", "p2", "p2"],
        }
    )
    out = attach_competition(panel, kappa=0.5)
    assert WEIGHT_COLUMN in out.columns
    assert len(out) == len(panel)
    assert out[WEIGHT_COLUMN].isin([0.5, 1.0]).all()


def test_kappa_outside_the_unit_interval_is_refused():
    panel = pd.DataFrame(
        {
            "season": [2022], "week": [1], "team": ["KC"],
            "position": ["RB"], "player_key": ["p1"],
        }
    )
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="fraction of the usual weight"):
            attach_competition(panel, kappa=bad)


def test_the_shipped_default_changes_no_average():
    """The arm is off. Turning it on is a deliberate act with a measured cost."""
    assert DEFAULT_KAPPA == 1.0
