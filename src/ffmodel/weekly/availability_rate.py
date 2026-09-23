"""How many of a club's remaining games a player will actually be on the field for.

The rest-of-season total already prices availability: a player expected to miss
time is projected lower for it. Dividing that total by the *scheduled* games
left therefore double-counts the absence -- the numerator is discounted and the
denominator is not -- and the resulting per-game rate describes a player who
plays every week, which is not the player the total was about.

Across 2022-2025 the average relevant player is on the field for **71.2%** of his
club's remaining games, so the naive rate understates per-active-game scoring by
about a factor of 1.4. For anyone carrying an injury it is far worse.

The obvious denominator is a play-rate feature the panel already carries, and it
does not survive contact:

===================  ====================  ============
``prior_play_rate``  realized share played  n
===================  ====================  ============
0.983                0.864                 2,827
0.899                0.802                 5,110
0.784                0.730                 5,360
0.606                0.600                 3,125
0.334                0.427                 2,199
===================  ====================  ============

It is compressed toward the middle at both ends -- over-predicting the healthy by
twelve points and under-predicting the marginal by nine -- because it is a
backward-looking average being read as a forward-looking probability. The split
that matters most here is worse still: a player who missed last week has a
``prior_play_rate`` of 0.622 and goes on to play **0.369** of what remains.

So the rate is fitted rather than borrowed, on the same walk-forward discipline
as everything else, and :func:`calibration` is what says whether the fit earned
its place before a column is written from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ffmodel.weekly.fitting import Ridge
from ffmodel.weekly.restofseason import OFFSET

#: Read from the panel. Availability is a question about a body and a role, so
#: the pre-game injury report belongs here alongside the play history -- it is
#: the only column that describes the week being projected.
RATE_FEATURES = (
    "prior_play_rate",
    "recent_play_rate",
    "weeks_since_played",
    "prior_games",
    "inj_out",
    "inj_questionable_or_worse",
    "prior_snap_share_recent",
)

TARGET = "played_rate_rest"


def add_played_rate_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of the club's remaining games this player actually appeared in.

    Counted backwards inside each player-season so the value at week ``w`` spans
    ``w`` to the end, matching the window the rest-of-season total is summed
    over. Rows whose offset is missing cannot define a share and are left NaN.
    """
    out = frame.sort_values(["player_key", "season", "week"], kind="mergesort").copy()
    played = pd.to_numeric(out["played"], errors="coerce").fillna(0.0)
    out["_games_played_rest"] = (
        played.iloc[::-1]
        .groupby([out["player_key"].iloc[::-1], out["season"].iloc[::-1]], sort=False)
        .cumsum()
        .iloc[::-1]
    )
    offset = pd.to_numeric(out[OFFSET], errors="coerce")
    out[TARGET] = (out["_games_played_rest"] / offset.where(offset > 0)).clip(0.0, 1.0)
    return out.drop(columns=["_games_played_rest"])


class PlayedRate:
    """Ridge on the availability block, predicting the share of games played."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.model: Ridge | None = None
        self.medians: pd.Series | None = None
        self.features: tuple[str, ...] = ()

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        block = frame[list(self.features)].apply(pd.to_numeric, errors="coerce")
        return block.fillna(self.medians).to_numpy(float)

    def fit(self, frame: pd.DataFrame) -> "PlayedRate":
        self.features = tuple(f for f in RATE_FEATURES if f in frame.columns)
        usable = frame[frame[TARGET].notna()]
        block = usable[list(self.features)].apply(pd.to_numeric, errors="coerce")
        self.medians = block.median()
        self.model = Ridge.fit(
            block.fillna(self.medians).to_numpy(float),
            usable[TARGET].to_numpy(float),
            penalty=self.alpha,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit first")
        # A share, so the prediction is bounded. The floor is not zero: a player
        # nobody expects to suit up still has to divide by something, and a
        # denominator of zero would report an infinite per-game rate for exactly
        # the players it is least safe to be confident about.
        return np.clip(self.model.predict(self._design(frame)), 0.05, 1.0)


def calibration(predicted: np.ndarray, observed: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Predicted against realized, by bucket. The check before the column ships."""
    frame = pd.DataFrame({"predicted": predicted, "observed": observed}).dropna()
    frame["bucket"] = pd.qcut(frame["predicted"], bins, duplicates="drop")
    out = frame.groupby("bucket", observed=True).agg(
        n=("observed", "size"),
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
    )
    out["gap"] = out["observed"] - out["predicted"]
    return out.reset_index(drop=True)
