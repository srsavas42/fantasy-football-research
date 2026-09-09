"""Every player-week's decision features, computed once per season.

The environment hands a policy a *history frame* and lets it work out what it
wants to know. That is the right interface -- it makes leakage impossible to get
wrong, because the frame has already been truncated -- and it is far too slow to
train against. Profiling one season put 85% of the runtime in a single line: a
grouped ``ewm().mean()`` re-derived from scratch for every player, every week,
every team, every episode, producing the same numbers each time.

They are the same numbers because they do not depend on the episode. What a
player had averaged going into week 9 of 2021 is a fact about 2021, not about
which team drafted him or which seed shuffled the draft order. So this computes
them once per season into a table indexed by ``(player_key, week)``, and every
episode of that season reads from it. Episodes go from 3.6 seconds to a fraction
of one, which is the difference between training being possible here and not.

**The leakage rule still has to hold, and it now holds here instead.** Every
column is built by taking a running statistic through week ``w`` and shifting it
one row forward within the player, so the value on row ``w`` describes weeks
strictly before ``w`` -- exactly what :meth:`FantasyLeagueEnv._history_before`
would have handed over. A player's rows are ordered by week and a bye contributes
no row, so "shift one row" and "the previous week he was on a roster" are the
same statement. This is the same ``_prior`` discipline the weekly feature layer
uses, and :func:`ffmodel.league.features.verify_against_history` checks the table
reproduces the policy that reads the frame, to the float.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Half-lives carried for the agent to weigh against each other. One matches the
# weekly feature layer's selected decay and the naive opponent; the longer two
# are here because how much recency to trust is exactly the kind of thing a
# policy should be allowed to learn rather than be told.
HALFLIVES = (1.0, 2.0, 4.0)

# Where the walk-forward projections live. Built by
# `scripts/build_projections.py`; absent, the two projection features are zero
# and the agent falls back to what it had before they existed.
PROJECTION_CACHE = Path(".cache/league_projections.parquet")

# The projection columns, in the order the cache writes them. Each horizon
# carries a mean and three quantiles: the level and the shape of the range are
# different facts, and the decisions want them differently -- a lineup behind
# late wants the p90, a claim on a player nobody has seen play is a bet on the
# top of his range, and a hurdle model's p10 of zero is the distinct statement
# "he might not play at all".
PROJECTION_COLUMNS = (
    "projection",
    "projection_p10",
    "projection_p50",
    "projection_p90",
    "ros_projection",
    "ros_projection_p10",
    "ros_projection_p50",
    "ros_projection_p90",
)

FEATURE_COLUMNS = (
    "bias",
    *PROJECTION_COLUMNS,
    "ewma1",
    "ewma2",
    "ewma4",
    "season_mean",
    "last_points",
    "played_rate",
    "experience",
    "adp_value",
    "is_qb",
    "is_rb",
    "is_wr",
    "is_te",
    "is_k",
    "is_dst",
)


def _running(frame: pd.DataFrame, values: pd.Series, how: str, **kwargs) -> pd.Series:
    """A statistic over a player's weeks up to and including each row.

    Deliberately *not* lagged here. The lag has to happen after the table is
    filled out to every week of the season, because a player's bye week carries
    no row: lagging first and filling afterwards propagates "everything before
    week 6" into week 7, when the truth for week 7 is "everything up to week 6".
    That ordering is worth a paragraph because getting it wrong is invisible --
    it misprices players only in the weeks after a gap, and it prices them low.
    """
    grouped = values.groupby(frame["player_key"], sort=False)
    if how == "ewm":
        return grouped.transform(lambda s: s.ewm(**kwargs, adjust=True).mean())
    if how == "mean":
        return grouped.transform(lambda s: s.expanding().mean())
    if how == "count":
        return pd.Series(grouped.cumcount() + 1, index=frame.index, dtype=float)
    if how == "identity":
        return values
    raise ValueError(f"unknown statistic {how!r}")


def load_projections(path: Path | None = None) -> pd.DataFrame | None:
    """The walk-forward projection cache, or ``None`` if it has not been built."""
    path = PROJECTION_CACHE if path is None else Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def attach_projections(
    table: pd.DataFrame, projections: pd.DataFrame | None, season: int
) -> pd.DataFrame:
    """Join the model's own projections onto the feature grid, **unlagged**.

    Every other column here is a running statistic computed through week ``w``
    and then shifted, because otherwise it would contain the week it is used to
    decide. A projection is different in kind: it is already a statement about
    week ``w`` made without week ``w``, produced by a model fitted on strictly
    earlier seasons. Lagging it would hand the agent last week's projection to
    decide this week, which is not a safety measure, it is a bug that would look
    like the model being useless.

    The two horizons are filled differently across a bye, and the difference is
    the point. Next week's projection is zero for a week the player does not
    play, which is exactly right. Rest-of-season is carried forward, because a
    player idle on Sunday is worth no less for the eight weeks after it -- and
    the rest-of-season number exists precisely so a waiver decision can ask what
    a player is worth over the rest of the year rather than over one afternoon.
    """
    out = table.copy()
    if projections is None:
        for column in PROJECTION_COLUMNS:
            out[column] = 0.0
        return out

    have = [c for c in PROJECTION_COLUMNS if c in projections.columns]
    block = projections[projections["season"] == int(season)]
    block = block.set_index(["player_key", "week"])[have]
    joined = block.reindex(out.index)
    for column in PROJECTION_COLUMNS:
        if column not in have:
            out[column] = 0.0
        elif column.startswith("ros_"):
            out[column] = (
                joined[column]
                .groupby(level="player_key")
                .ffill()
                .fillna(0.0)
                .to_numpy(float)
            )
        else:
            out[column] = joined[column].fillna(0.0).to_numpy(float)
    return out


def build_feature_table(
    pool: pd.DataFrame, season: int, projections: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per ``(player_key, week)`` of ``season``, all strictly prior.

    Built from the season's own rows only. A player's history does not reach
    across the season boundary here, which matches what the environment shows a
    policy: an episode is one season and starts with nobody having played.
    """
    block = pool[pool["season"] == int(season)].copy()
    if block.empty:
        raise ValueError(f"no pool rows for season {season}")
    block = block.sort_values(["player_key", "week"], kind="mergesort").reset_index(
        drop=True
    )

    points = pd.to_numeric(block["points"], errors="coerce").fillna(0.0).astype(float)
    played = pd.to_numeric(block["played"], errors="coerce").fillna(0).astype(float)

    out = pd.DataFrame(index=block.index)
    out["player_key"] = block["player_key"].to_numpy()
    out["week"] = block["week"].astype(int).to_numpy()
    out["bias"] = 1.0

    for halflife in HALFLIVES:
        alpha = 1.0 - 0.5 ** (1.0 / halflife)
        out[f"ewma{halflife:g}"] = _running(block, points, "ewm", alpha=alpha)

    out["season_mean"] = _running(block, points, "mean")
    out["last_points"] = _running(block, points, "identity")
    out["played_rate"] = _running(block, played, "mean")
    # Weeks of history, compressed: the difference between one game and four is
    # worth far more than between eleven and fourteen.
    out["experience"] = np.log1p(_running(block, points, "count"))

    # A low pick is a high number, reciprocally, so the gap between the 1st and
    # 10th pick is larger than between the 101st and 110th -- which is how draft
    # value behaves and how `AdpPolicy` already reads the board.
    rank = pd.to_numeric(block["adp_rank"], errors="coerce").to_numpy(float)
    out["adp_value"] = np.where(np.isfinite(rank) & (rank > 0), 1.0 / rank, 0.0)

    position = block["position"].astype(str).to_numpy()
    for name in ("QB", "RB", "WR", "TE", "K", "DST"):
        out[f"is_{name.lower()}"] = (position == name).astype(float)

    out = out.set_index(["player_key", "week"]).sort_index()

    # Fill out every week for every player, then lag. A player's club takes a
    # bye and he has no row that week, but he still has a past and a policy
    # still has to value him: the roster mechanic decides who to cut and who to
    # claim on exactly these numbers, and a player who silently loses his
    # average the week he is idle gets cut for being idle.
    weeks = list(range(int(block["week"].min()), int(block["week"].max()) + 1))
    grid = pd.MultiIndex.from_product(
        [out.index.get_level_values("player_key").unique(), weeks],
        names=["player_key", "week"],
    )
    running = out.reindex(grid).sort_index().groupby(level="player_key").ffill()

    # The lag, on the completed grid, so row `w` describes weeks strictly before
    # `w` whether or not the player has a row in `w`.
    shifted = running.groupby(level="player_key").shift(1)
    # The constants are facts about the player rather than about his past, so
    # they are not lagged: his position does not become unknown in week one.
    constant = ["bias", "adp_value"] + [c for c in out.columns if c.startswith("is_")]
    shifted[constant] = running.groupby(level="player_key")[constant].bfill()

    # A player's first week has no past. Zero is the honest value for every
    # running statistic there, and `experience` being zero is what tells a
    # policy the rest of the row is empty rather than bad.
    shifted = shifted.fillna(0.0).sort_index()
    return attach_projections(shifted, projections, season)


def build_feature_tables(
    pool: pd.DataFrame, seasons, projections: pd.DataFrame | None = "auto"
) -> dict[int, pd.DataFrame]:
    """Feature tables per season, with the projection cache loaded once."""
    if isinstance(projections, str) and projections == "auto":
        projections = load_projections()
    return {
        int(season): build_feature_table(pool, season, projections)
        for season in seasons
    }


def as_matrix(table: pd.DataFrame, keys, week: int) -> np.ndarray:
    """Feature rows for ``keys`` in ``week``, in order, missing rows as zeros.

    A player with no row for the week is on bye. The environment benches him
    regardless, so a zero row costs nothing and avoids a lookup failure being
    the way that is discovered.
    """
    index = pd.MultiIndex.from_product([list(keys), [int(week)]])
    frame = table.reindex(index.set_names(table.index.names))
    values = frame[list(FEATURE_COLUMNS)].to_numpy(float)
    return np.nan_to_num(values, nan=0.0)


def verify_against_history(pool: pd.DataFrame, season: int, halflife: float = 1.0):
    """Check the table says what reading the truncated frame would have said.

    The table is a shortcut around the environment's one hard rule, so it is
    worth proving rather than asserting that the shortcut lands in the same
    place. Returns the largest absolute disagreement across every player-week.
    """
    from ffmodel.league.policies import EwmaPolicy

    table = build_feature_table(pool, season)
    block = pool[pool["season"] == int(season)]
    column = f"ewma{halflife:g}"
    worst = 0.0
    for week in sorted(block["week"].unique()):
        history = block[block["week"] < week]
        keys = block[block["week"] == week]["player_key"].tolist()
        expected = EwmaPolicy(
            halflife=halflife, fallback_to_board=False
        ).score(keys, history, int(week), pd.DataFrame(columns=["player_key", "adp_rank"]))
        got = table[column].reindex(
            pd.MultiIndex.from_arrays([keys, [int(week)] * len(keys)])
        )
        for key, value in zip(keys, got.to_numpy(float)):
            worst = max(worst, abs(float(np.nan_to_num(value)) - expected.get(key, 0.0)))
    return worst


def build_volatility(pool: pd.DataFrame, season: int) -> pd.Series:
    """Standard deviation of a player's own points so far, lagged like the rest.

    Kept out of :data:`FEATURE_COLUMNS` deliberately. It is not a projection --
    it says nothing about whether a player will score more -- so it has no place
    in a ranking that maximises points. It exists for the one decision that is
    not about points: a manager who is thirty behind with one game left wants the
    volatile player and a manager who is thirty ahead wants the steady one, and
    neither of those preferences is expressible without it.

    Indexed like the feature table, so a policy can look up ``(player, week)``
    and get what was knowable going into that week.
    """
    block = pool[pool["season"] == int(season)].copy()
    block = block.sort_values(["player_key", "week"], kind="mergesort").reset_index(
        drop=True
    )
    points = pd.to_numeric(block["points"], errors="coerce").fillna(0.0).astype(float)
    running = points.groupby(block["player_key"], sort=False).transform(
        lambda s: s.expanding().std()
    )
    out = pd.DataFrame(
        {
            "player_key": block["player_key"].to_numpy(),
            "week": block["week"].astype(int).to_numpy(),
            "volatility": running.to_numpy(),
        }
    ).set_index(["player_key", "week"]).sort_index()

    weeks = list(range(int(block["week"].min()), int(block["week"].max()) + 1))
    grid = pd.MultiIndex.from_product(
        [out.index.get_level_values("player_key").unique(), weeks],
        names=["player_key", "week"],
    )
    filled = out.reindex(grid).sort_index().groupby(level="player_key").ffill()
    return filled.groupby(level="player_key").shift(1)["volatility"].fillna(0.0)
