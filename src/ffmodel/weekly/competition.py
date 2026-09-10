"""Who else was on the field when a player produced his history.

Every history feature in :mod:`ffmodel.weekly.features` pools a player's past
weeks unconditionally, and the competitive environment those weeks were produced
in is not constant. Measured on skill players 2016-2025 with at least four prior
games, a week in which the man listed ahead of a player is ruled out is worth
**+1.25 points** to him within his own record (t=+7.99), and that production then
enters his average at full weight and stays there:

===============================  ======  ===============================
week (the man ahead is back)          n  points - prior_points_recent
===============================  ======  ===============================
no absence in the last two       38,263  -0.098
he was absent one week ago        1,451  **-0.602**
he was absent two weeks ago       1,179  **-0.706**
===============================  ======  ===============================

That is a real defect in the *feature*. This module was built to repair it, by
down-weighting such weeks in the averages rather than by adding a covariate --
``docs/target-competition-2026-09.md`` records what happens when a room-structure
covariate collides with something the model already holds, and a weight adds no
degree of freedom at all.

**It does not help, and the reason is worth keeping.** The next-week model
already carries ``ahead_out``, ``ahead_out_lagged`` and ``depth_promoted_lagged``,
and it uses them to undo the contamination before a prediction leaves the
building: against the *fitted* model the post-absence residual is +0.106 one week
later and -0.280 two weeks later, against a +0.135 baseline, all inside one
standard error. Only 4% of scored rows are affected at all, so perfect removal of
their remaining bias is worth at most 0.05% of pooled MAE. The measured sweep
lands inside that -- best -0.10% MAE at ``kappa`` 0.5, on a curve that is not
monotone. ``docs/teammate-competition-2026-09.md`` has the run.

So ``DEFAULT_KAPPA`` is one, the identity, and the machinery stays for the season
layer, where the same weights are used by
``scripts/screen_season_competition.py`` and where nothing plays the part
``ahead_out`` plays here.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ffmodel.weekly.news import _ahead_out, load_depth, load_injury_report

#: Name of the column :func:`attach_competition` writes and
#: :func:`ffmodel.weekly.features.add_features` reads.
WEIGHT_COLUMN = "competition_weight"

#: How much of its usual weight a boosted week keeps. One is the identity.
#: ``scripts/sweep_competition_weight.py`` measured 1.0, 0.75, 0.5, 0.25 and 0.0
#: on 2021/2022 and found nothing outside noise, so the shipped value is the one
#: that changes no average.
DEFAULT_KAPPA = 1.0


def competition_state(panel: pd.DataFrame, *, seasons: Iterable[int] | None = None) -> pd.Series:
    """Was someone ahead of this player at his position ruled out that week?

    Rebuilt here rather than read off :func:`ffmodel.weekly.news.add_news_features`
    because the histories that need it are built first. The two feeds are
    cached, so the second pass over them costs a merge and nothing else.
    """
    seasons = sorted(panel["season"].unique().tolist()) if seasons is None else list(seasons)
    frame = panel[["season", "week", "team", "position", "player_key"]].copy()

    injuries = load_injury_report(seasons)
    if injuries.empty:
        return pd.Series(0.0, index=panel.index)
    frame = frame.merge(
        injuries[["season", "week", "player_key", "inj_status"]],
        on=["season", "week", "player_key"],
        how="left",
    )
    frame["inj_status"] = frame["inj_status"].fillna(0.0)

    depth = load_depth(seasons)
    if depth.empty:
        return pd.Series(0.0, index=panel.index)
    frame = frame.merge(depth, on=["season", "week", "player_key"], how="left")

    # The merges preserve row order but reset the labels; put the result back on
    # the panel's own index so the caller can assign it without realigning.
    state = _ahead_out(frame, "depth_rank", "inj_status")
    return pd.Series(state.to_numpy(float), index=panel.index)


def attach_competition(
    panel: pd.DataFrame,
    *,
    seasons: Iterable[int] | None = None,
    kappa: float = DEFAULT_KAPPA,
) -> pd.DataFrame:
    """Add ``competition_weight`` to ``panel``, ready for ``add_features``.

    ``kappa`` of one is the identity and leaves every average where it was, so
    the arm can be turned off without removing the call.
    """
    if not 0.0 <= float(kappa) <= 1.0:
        raise ValueError(f"kappa is a fraction of the usual weight, got {kappa!r}")
    frame = panel.copy()
    boosted = competition_state(frame, seasons=seasons)
    frame["room_depleted"] = boosted.to_numpy(float)
    frame[WEIGHT_COLUMN] = np.where(boosted.to_numpy(float) > 0, float(kappa), 1.0)
    return frame
