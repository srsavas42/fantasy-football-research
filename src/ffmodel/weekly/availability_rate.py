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

**A second correction, about what to do with that denominator.** It was first
used to turn the rest-of-season total into a per-game rate by division, and that
was wrong in a way that showed up as Puka Nacua at 32 points a game. The shipped
total is blended toward the draft-board curve, which does not know he missed two
games, while the expected-games denominator is discounted for exactly that. A
ratio of two separately fitted things inherits both of their errors and
disagrees with itself whenever they disagree with each other. Nacua's model-only
total over the same denominator is 23.5 -- still high -- so the blend was not the
whole story either.

The rate is therefore modelled directly: rest-of-season points divided by games
*played*, fitted with the same ridge and the same features as the total
(:func:`add_points_per_active_game_target`). Walk-forward on 2023-2025, for
players with at least three games over the remainder, MAE in points a game:

================================  ======  ======  =========
population                        direct  ratio   own mean
================================  ======  ======  =========
everyone (n=10,119)               2.99    3.42    3.39
missed last week (n=1,327)        3.26    4.23    3.63
ratio says more than 25 (n=88)    4.12    14.07   4.49
================================  ======  ======  =========

The ratio method's bias on the returning-player split is -1.19 against +0.46, and
when it produces a number like Nacua's it is wrong by fourteen points a game.

That table compared the direct rate with an *unblended* ratio. The version that
actually shipped divided the *blended* total, and it is worse again. Same
walk-forward, every relevant player with at least one game over the remainder
(so noisier than the table above), MAE in points a game with bias in brackets:

==================================  ===========  ===============
population                          direct       shipped ratio
==================================  ===========  ===============
everyone (n=12,332)                 3.39 (+0.03) 4.29 (-1.00)
drafted, where the blend applies    3.50 (-0.01) 4.63 (-1.41)
undrafted, where it is a no-op      3.14 (+0.11) 3.48 (-0.01)
missed last week                    3.90 (+0.07) 5.85 (-3.05)
  ... and drafted (n=847)           3.76 (+0.36) 7.37 (-5.46)
  ... and top-50 ADP (n=209)        4.17 (+0.67) 10.25 (-9.28)
shipped rate above 25 (n=376)       5.52 (-1.04) 19.91 (-19.59)
==================================  ===========  ===============

The undrafted row is the control: there the blend does nothing and the two are
close, so the damage is the blend meeting a denominator that knows about the
absence. It grows with how much the board believes in the player and how long he
has been out -- a ten-point miss for a top-50 player who sat last week.

**Superseded for shipping.** The rate and the games count in this module are still
two models whose product is not the total. ``ffmodel.weekly.ros_reconciled`` replaces
the pair with one calculation -- the rate weighted by games played, the games fitted
per absence state -- and ships the product as the total. This module stays as the
record of why, and for :func:`add_points_per_active_game_target`, which the new model
uses.
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


RATE_TARGET = "ppag_rest"
GAMES_PLAYED_REST = "games_played_rest"


def add_points_per_active_game_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Rest-of-season points divided by the games he actually played in them.

    The response for a per-game rate. It is undefined for a player who never
    takes the field again, and those rows are left NaN rather than set to zero:
    a zero would teach the fit that a player who is hurt is a bad player, which
    is exactly the confusion the rate exists to remove.
    """
    from ffmodel.weekly.restofseason import TARGET

    out = frame.sort_values(["player_key", "season", "week"], kind="mergesort").copy()
    played = pd.to_numeric(out["played"], errors="coerce").fillna(0.0)
    games = (
        played.iloc[::-1]
        .groupby([out["player_key"].iloc[::-1], out["season"].iloc[::-1]], sort=False)
        .cumsum()
        .iloc[::-1]
    )
    out[GAMES_PLAYED_REST] = games
    out[RATE_TARGET] = pd.to_numeric(out[TARGET], errors="coerce") / games.where(games > 0)
    return out


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
