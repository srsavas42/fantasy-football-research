"""The naive draft-board forecast, restated weekly.

Every claim this package makes is measured against average draft position,
because ADP is what a fantasy manager already has for free and a model that does
not beat it is not worth running. The season layer learned that the hard way: a
four-parameter rank curve beat the whole hierarchical pipeline by 8.1% MAE, and
four hypotheses for why had to be tested and eliminated before the honest
comparison emerged.

The weekly restatement is the same curve, prorated. Fit season points on log
draft rank within each position, on seasons strictly before the holdout, then
divide by the number of games in a season to get a per-game level. For a single
week that level *is* the forecast; for the rest of a season it is multiplied by
the games remaining. Spread comes from the curve's own residuals at nearby
ranks, so the intervals are the historical spread of outcomes for players drafted
around there rather than an assumed shape.

Two things about this baseline are worth stating before reading any comparison
against it.

**It is stale by construction, and increasingly so.** ADP is published in
August. In week 1 it carries everything anybody knows; by week 12 the model has
eleven weeks of usage the board has never seen. Beating it in week 12 is close to
free, and beating it in week 1 is the only genuinely hard version of the test.
The comparison is therefore always reported by week, never pooled alone.

**It is silent on players it did not rank.** Undrafted rows get the
replacement-level tail of the curve rather than nothing, because a weekly
decision about an unranked player is still a decision -- but that is an
extrapolation of the curve past its data, and it is the weakest part of this
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.weekly.nextweek import POSITIONS

# Below this many drafted rows a per-position curve is noise; the whole board is
# used instead. Mirrors ``ffmodel.models.market_blend``.
MIN_RESIDUALS = 40

# Residuals are pooled from ranks within this window of the row being projected,
# so the spread deep on the board is not borrowed from the first round.
RANK_WINDOW = 24


def season_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Season totals and games played per (season, player), from the panel."""
    return (
        frame.groupby(["season", "player_key"], as_index=False)
        .agg(
            total=("points", "sum"),
            weeks=("points", "size"),
            adp_rank=("adp_rank", "first"),
            adp_drafted=("adp_drafted", "first"),
            position=("position", "first"),
        )
    )


@dataclass
class WeeklyRankCurve:
    """Points per game from draft rank alone, prorated to a horizon.

    ``per_game`` selects the response: ``True`` predicts one week, ``False``
    multiplies by the panel's ``games_remaining`` offset for a rest-of-season
    total.
    """

    name: str = "adp-curve"
    per_game: bool = True
    offset: str = "games_remaining"
    coefficients: dict = field(default_factory=dict)
    residuals: dict = field(default_factory=dict)
    ranks: dict = field(default_factory=dict)
    weeks_per_season: float = 17.0

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "WeeklyRankCurve":
        """Fitted on season totals of drafted players, not on weekly rows.

        The curve is a statement about a draft board, and a draft board is a
        forecast of a season. Fitting it on weekly rows would let players with
        more weeks in the panel dominate the very relationship being estimated.
        """
        seasons = season_points(frame)
        self.weeks_per_season = float(
            frame.groupby(["season", "team"])["week"].nunique().mean()
        )
        rank = pd.to_numeric(seasons["adp_rank"], errors="coerce").to_numpy(float)
        drafted = pd.to_numeric(seasons["adp_drafted"], errors="coerce").eq(1).to_numpy()
        total = seasons["total"].to_numpy(float)
        usable = drafted & np.isfinite(rank) & (rank > 0) & np.isfinite(total)
        if usable.sum() < MIN_RESIDUALS:
            raise ValueError(
                f"the weekly rank curve needs at least {MIN_RESIDUALS} drafted "
                f"player-seasons, got {int(usable.sum())}"
            )
        position = seasons["position"].astype(str).to_numpy()
        for name in POSITIONS:
            at = usable & (position == name)
            fit = at if at.sum() >= MIN_RESIDUALS else usable
            log_rank = np.log(rank[fit])
            coefficients = np.polyfit(log_rank, total[fit], 1)
            self.coefficients[name] = coefficients
            # Residuals are kept on the season scale and divided down with the
            # centre, so a per-game interval is the season interval prorated
            # rather than a separately invented one.
            self.residuals[name] = total[fit] - np.polyval(coefficients, log_rank)
            self.ranks[name] = rank[fit]
        return self

    def _centre(self, frame: pd.DataFrame) -> np.ndarray:
        rank = pd.to_numeric(frame["adp_rank"], errors="coerce").to_numpy(float)
        # An unranked player is placed at the deepest rank the curve saw, which
        # is the replacement level it implies rather than a missing value.
        deepest = max(
            (float(values.max()) for values in self.ranks.values() if len(values)),
            default=300.0,
        )
        rank = np.where(np.isfinite(rank) & (rank > 0), rank, deepest)
        position = frame["position"].astype(str).to_numpy()
        centre = np.zeros(len(frame), dtype=float)
        for name in POSITIONS:
            want = position == name
            if not want.any():
                continue
            coefficients = self.coefficients.get(name)
            if coefficients is None:
                continue
            centre[want] = np.polyval(coefficients, np.log(rank[want]))
        return centre

    def _scale(self, frame: pd.DataFrame) -> np.ndarray:
        """Games this forecast covers: one week, or the rest of the season."""
        if self.per_game:
            return np.ones(len(frame), dtype=float)
        return (
            pd.to_numeric(frame[self.offset], errors="coerce")
            .fillna(1.0)
            .to_numpy(float)
        )

    def predict_samples(
        self, frame: pd.DataFrame, draws: int, seed: int = 0
    ) -> np.ndarray:
        if not self.coefficients:
            raise RuntimeError("fit the curve before predicting")
        rng = np.random.default_rng(seed)
        rank = pd.to_numeric(frame["adp_rank"], errors="coerce").to_numpy(float)
        deepest = max(
            (float(values.max()) for values in self.ranks.values() if len(values)),
            default=300.0,
        )
        rank = np.where(np.isfinite(rank) & (rank > 0), rank, deepest)
        position = frame["position"].astype(str).to_numpy()
        centre = self._centre(frame)
        scale = self._scale(frame) / self.weeks_per_season

        out = np.zeros((len(frame), draws), dtype=float)
        for name in POSITIONS:
            want = np.flatnonzero(position == name)
            if not want.size:
                continue
            pool = self.residuals.get(name)
            if pool is None:
                continue
            fit_ranks = self.ranks[name]
            for row in want:
                near = np.abs(fit_ranks - rank[row]) <= RANK_WINDOW
                local = pool[near] if near.sum() >= MIN_RESIDUALS else pool
                picks = rng.integers(0, len(local), size=draws)
                out[row] = (centre[row] + local[picks]) * scale[row]
        missing = ~np.isin(position, POSITIONS)
        if missing.any():
            out[missing] = 0.0
        # Neither a week nor a rest of season can be negative in aggregate, and
        # a line linear in log rank goes under zero deep on the board.
        return np.maximum(out, 0.0)


HORIZON_BUCKETS = (("early", 1, 4), ("mid", 5, 10), ("late", 11, 18))


def bucket_labels(week) -> np.ndarray:
    """Which horizon each row's week falls in."""
    week = np.asarray(week, dtype=float)
    labels = np.empty(len(week), dtype=object)
    for name, low, high in HORIZON_BUCKETS:
        labels[(week >= low) & (week <= high)] = name
    return labels


def fit_blend_weights(
    frame: pd.DataFrame,
    build_model,
    target: str,
    *,
    draws: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """Per-horizon weight on the model against the curve, estimated honestly.

    The variance-optimal weight for combining two forecasts is the slope of

        observed - curve = a + b * (model - curve)

    which needs *out-of-sample* model predictions to estimate: fitting the model
    and the weight on the same rows would report how well the model fits its own
    training data and hand back a weight near one.

    So the most recent season is held out from ``frame``, both forecasts are
    fitted on what precedes it and scored on it, and the slope is taken there.
    The caller then refits both on everything for the actual projection. This is
    the same discipline the season layer's blend uses across holdouts, applied
    inside a single training window because a live projection has no later
    season to borrow from.

    Estimated per horizon because the answer genuinely differs by horizon: the
    model earns its weight as the season gives it usage the board never saw.
    Measured weights run about 0.5 early and 1.0 late.
    """
    seasons = sorted(frame["season"].unique().tolist())
    if len(seasons) < 3:
        # Not enough history to hold a season out; trust the model, which is
        # what a weight of one means.
        return {name: 1.0 for name, _, _ in HORIZON_BUCKETS}
    inner_test = frame[frame["season"] == seasons[-1]]
    inner_train = frame[frame["season"] < seasons[-1]]

    model = build_model().fit(inner_train, inner_train[target].to_numpy(float))
    curve = WeeklyRankCurve(per_game=False, offset="games_remaining").fit(
        inner_train, inner_train["points"].to_numpy(float)
    )
    drafted = pd.to_numeric(inner_test["adp_drafted"], errors="coerce").eq(1).to_numpy()
    block = inner_test[drafted]
    if block.empty:
        return {name: 1.0 for name, _, _ in HORIZON_BUCKETS}

    observed = block[target].to_numpy(float)
    model_mean = model.predict_samples(block, draws=draws, seed=seed).mean(axis=1)
    curve_mean = curve.predict_samples(block, draws=draws, seed=seed).mean(axis=1)
    labels = bucket_labels(block["week"].to_numpy(float))

    from ffmodel.models.market_blend import slope_weight

    weights = {}
    for name, _, _ in HORIZON_BUCKETS:
        want = labels == name
        if want.sum() < 50:
            weights[name] = 1.0
            continue
        weights[name] = slope_weight(
            observed[want], model_mean[want], curve_mean[want]
        )
    return weights


def attach_adp(frame: pd.DataFrame, directory=None) -> pd.DataFrame:
    """Join preseason ADP onto the weekly panel, one rank per player-season."""
    from ffmodel.features.market import DEFAULT_ADP_DIR, load_adp, _name_key

    directory = DEFAULT_ADP_DIR if directory is None else directory
    seasons = sorted(frame["season"].unique().tolist())
    adp = load_adp(seasons, directory)
    out = frame.copy()
    out["key"] = _name_key(out["player_name"].astype(str))
    merged = out.merge(
        adp[["season", "key", "adp_rank", "adp_position"]], on=["season", "key"], how="left"
    )
    # A name collision that lands a receiver on a running back's rank is worse
    # than no rank at all, so a disagreeing position drops the join.
    disagrees = merged["adp_position"].notna() & merged["adp_position"].ne(
        merged["position"]
    )
    merged.loc[disagrees, "adp_rank"] = np.nan
    merged["adp_drafted"] = merged["adp_rank"].notna().astype(float)
    return merged.drop(columns=["key", "adp_position"], errors="ignore")
