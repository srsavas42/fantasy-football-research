"""Dirichlet-Multinomial allocation of team targets or carries.

Each team-week has a variable active roster, represented internally as a
padded matrix plus an explicit mask.  Player concentrations depend on trailing
usage, optional snap share, trailing efficiency, position priors, and a
partially pooled player effect.  At prediction time concentrations are
renormalized over the supplied active set, so removing an injured starter
automatically reallocates all opportunities among the remaining players.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.features.trailing import player_key
from ffmodel.models.base import sample_model

GROUP_COLUMNS = ["season", "week", "team"]

STREAMS = {
    "target": {
        "outcome": "targets",
        "positions": ("RB", "WR", "TE"),
        "features": (
            "ewma_snap_share",
            "ewma_target_share",
            "ewma_opportunity_share",
            "ewma_ypt",
            "ewma_catch_rate",
        ),
    },
    "carry": {
        "outcome": "rush_att",
        "positions": ("QB", "RB", "WR", "TE"),
        "features": (
            "ewma_snap_share",
            "ewma_carry_share",
            "ewma_opportunity_share",
            "ewma_ypc",
            "ewma_yds_per_touch",
        ),
    },
}


@dataclass
class GroupDesign:
    rows: pd.DataFrame
    group_keys: pd.DataFrame
    counts: np.ndarray
    totals: np.ndarray
    mask: np.ndarray
    X: np.ndarray
    position_idx: np.ndarray
    player_idx: np.ndarray
    row_idx: np.ndarray


@dataclass
class SharePrediction:
    """Posterior opportunity counts and shares aligned to ``rows``."""

    rows: pd.DataFrame
    group_keys: pd.DataFrame
    counts: np.ndarray
    group_totals: np.ndarray

    @property
    def shares(self) -> np.ndarray:
        group_idx = self.rows["_group_idx"].to_numpy(dtype=int)
        den = self.group_totals[group_idx]
        return np.divide(
            self.counts,
            den,
            out=np.zeros_like(self.counts, dtype=float),
            where=den > 0,
        )


def _stack(posterior, name: str) -> np.ndarray:
    return posterior[name].stack(sample=("chain", "draw")).to_numpy()


def _codes(values: pd.Series, categories: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(categories)}
    return values.astype(str).map(lookup).fillna(-1).to_numpy(dtype=int)


@dataclass
class OpportunityShareModel:
    """Fitted target- or carry-allocation model."""

    stream: str = "target"
    positions: list[str] = field(default_factory=list)
    players: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    feature_fill: dict[str, float] = field(default_factory=dict)
    feature_mean: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    idata: object = None

    def __post_init__(self):
        if self.stream not in STREAMS:
            raise ValueError(f"stream must be one of {sorted(STREAMS)}, got {self.stream!r}")

    @property
    def outcome_col(self) -> str:
        return STREAMS[self.stream]["outcome"]

    def _eligible_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = set(GROUP_COLUMNS + ["player_name", "position", self.outcome_col])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"share model input is missing columns: {sorted(missing)}")
        d = frame[frame["position"].isin(STREAMS[self.stream]["positions"])].copy()
        # When snaps exist, is_active is a genuine active-roster indicator.  On
        # the legacy path it is opportunity-derived, so filtering would erase
        # active zero-opportunity players and bias the likelihood.
        has_snap_activity = "offense_snaps" in d and d["offense_snaps"].notna().any()
        if has_snap_activity and "is_active" in d:
            d = d[d["is_active"] == 1]
        d[self.outcome_col] = (
            pd.to_numeric(d[self.outcome_col], errors="coerce")
            .fillna(0.0)
            .round()
            .clip(lower=0)
            .astype(int)
        )
        d["_player_key"] = player_key(d).astype(str)
        # Defensive aggregation for provider feeds that contain duplicate rows.
        key = GROUP_COLUMNS + ["_player_key"]
        if d.duplicated(key).any():
            aggregations = {self.outcome_col: "sum"}
            for column in d.columns:
                if column not in key and column != self.outcome_col:
                    aggregations[column] = "first"
            d = d.groupby(key, as_index=False, dropna=False).agg(aggregations)
        return d.sort_values(GROUP_COLUMNS + ["_player_key"]).reset_index(drop=True)

    def _fit_scaler(self, d: pd.DataFrame) -> None:
        candidates = [
            c for c in STREAMS[self.stream]["features"]
            if c in d and pd.to_numeric(d[c], errors="coerce").notna().any()
        ]
        if not candidates:
            raise ValueError(
                "no trailing share/efficiency features are available; run build_features first"
            )
        self.feature_names = candidates
        for column in candidates:
            values = pd.to_numeric(d[column], errors="coerce")
            fill = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(fill)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            self.feature_fill[column] = fill
            self.feature_mean[column] = mean
            self.feature_scale[column] = scale if scale > 1e-8 else 1.0

    def _matrix(self, d: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.feature_names:
            values = pd.to_numeric(
                d.get(name, pd.Series(np.nan, index=d.index)), errors="coerce"
            ).fillna(self.feature_fill[name])
            columns.append(
                (values.to_numpy(dtype=float) - self.feature_mean[name])
                / self.feature_scale[name]
            )
        return np.column_stack(columns)

    def _design(self, frame: pd.DataFrame, *, fit: bool = False) -> GroupDesign:
        d = self._eligible_rows(frame)
        totals = d.groupby(GROUP_COLUMNS)[self.outcome_col].transform("sum")
        if fit:
            d = d[totals > 0].copy()
        d = d.reset_index(drop=True)
        if d.empty:
            message = (
                f"no team-weeks have positive {self.outcome_col}"
                if fit
                else "the supplied active set has no eligible players"
            )
            raise ValueError(message)
        if fit:
            self.positions = sorted(d["position"].astype(str).unique())
            self.players = sorted(d["_player_key"].unique())
            self._fit_scaler(d)

        position_codes = _codes(d["position"], self.positions)
        player_codes = _codes(d["_player_key"], self.players)
        row_X = self._matrix(d)
        groups = list(d.groupby(GROUP_COLUMNS, sort=True, dropna=False))
        max_slots = max(len(group) for _, group in groups)
        n_groups, n_features = len(groups), len(self.feature_names)
        counts = np.zeros((n_groups, max_slots), dtype=int)
        mask = np.zeros((n_groups, max_slots), dtype=float)
        X = np.zeros((n_groups, max_slots, n_features), dtype=float)
        pos_idx = np.zeros((n_groups, max_slots), dtype=int)
        ply_idx = np.zeros((n_groups, max_slots), dtype=int)
        row_idx = np.full((n_groups, max_slots), -1, dtype=int)
        group_rows = []

        for group_i, (key, group) in enumerate(groups):
            indices = group.index.to_numpy(dtype=int)
            size = len(indices)
            counts[group_i, :size] = d.loc[indices, self.outcome_col].to_numpy(dtype=int)
            mask[group_i, :size] = 1.0
            X[group_i, :size, :] = row_X[indices]
            pos_idx[group_i, :size] = position_codes[indices]
            ply_idx[group_i, :size] = player_codes[indices]
            row_idx[group_i, :size] = indices
            group_rows.append(dict(zip(GROUP_COLUMNS, key)))
            d.loc[indices, "_group_idx"] = group_i

        d["_group_idx"] = d["_group_idx"].astype(int)
        return GroupDesign(
            rows=d,
            group_keys=pd.DataFrame(group_rows),
            counts=counts,
            totals=counts.sum(axis=1),
            mask=mask,
            X=X,
            position_idx=pos_idx,
            player_idx=ply_idx,
            row_idx=row_idx,
        )

    def fit(self, features: pd.DataFrame, **sample_kwargs) -> "OpportunityShareModel":
        """Fit the ragged active-roster Dirichlet-Multinomial model."""
        import pymc as pm

        design = self._design(features, fit=True)
        with pm.Model() as model:
            mu_position = pm.Normal("mu_position", 0.0, 1.0)
            position_sd = pm.HalfNormal("position_sd", 0.75)
            position_z = pm.Normal(
                "position_z", 0.0, 1.0, shape=len(self.positions)
            )
            position_alpha = pm.Deterministic(
                "position_alpha", mu_position + position_z * position_sd
            )
            beta = pm.Normal("beta", 0.0, 0.7, shape=len(self.feature_names))
            player_sd = pm.HalfNormal("player_sd", 0.5)
            player_z = pm.Normal("player_z", 0.0, 1.0, shape=len(self.players))
            player_effect = pm.Deterministic("player_effect", player_z * player_sd)

            eta = position_alpha[design.position_idx]
            eta = eta + pm.math.sum(design.X * beta, axis=2)
            eta = eta + player_effect[design.player_idx]
            concentration = pm.math.exp(pm.math.clip(eta, -8.0, 8.0))
            concentration = concentration * design.mask + (1.0 - design.mask) * 1e-8
            pm.DirichletMultinomial(
                "obs", n=design.totals, a=concentration, observed=design.counts
            )
            sample_kwargs.setdefault("target_accept", 0.93)
            self.idata = sample_model(model, **sample_kwargs)
        return self

    @staticmethod
    def _known_effect(
        effect: np.ndarray, idx: np.ndarray, fallback: np.ndarray | None = None
    ) -> np.ndarray:
        out = np.zeros((*idx.shape, effect.shape[-1]), dtype=float)
        if fallback is not None:
            out[...] = fallback
        known = idx >= 0
        if known.any():
            out[known] = effect[idx[known], :]
        return out

    def _prediction_totals(self, design: GroupDesign, team_totals, draws: int) -> np.ndarray:
        if team_totals is None:
            return np.repeat(design.totals[:, None], draws, axis=1)
        if isinstance(team_totals, pd.DataFrame):
            total_col = "team_targets" if self.stream == "target" else "team_rush_att"
            if total_col not in team_totals:
                raise ValueError(f"team_totals must include {total_col!r}")
            merged = design.group_keys.merge(
                team_totals[GROUP_COLUMNS + [total_col]], on=GROUP_COLUMNS, how="left"
            )
            values = merged[total_col].to_numpy(dtype=float)
        else:
            values = np.asarray(team_totals)
        if values.ndim == 1:
            values = np.repeat(values[:, None], draws, axis=1)
        if values.shape != (len(design.group_keys), draws):
            raise ValueError(
                "team_totals must have shape (n_groups,) or (n_groups, n_draws)"
            )
        return np.rint(values).clip(min=0).astype(int)

    def predict_samples(
        self, features: pd.DataFrame, *, team_totals=None, seed: int = 0
    ) -> SharePrediction:
        """Allocate supplied or observed team totals over the active players."""
        if self.idata is None:
            raise RuntimeError("fit the opportunity-share model before predicting")
        design = self._design(features, fit=False)
        post = self.idata.posterior
        position = _stack(post, "position_alpha")
        player = _stack(post, "player_effect")
        beta = _stack(post, "beta")
        draws = beta.shape[-1]
        position_fallback = _stack(post, "mu_position")
        eta = self._known_effect(
            position, design.position_idx, fallback=position_fallback
        )
        eta = eta + np.einsum("gkf,fs->gks", design.X, beta)
        eta = eta + self._known_effect(player, design.player_idx)
        concentration = np.exp(np.clip(eta, -8.0, 8.0)) * design.mask[:, :, None]
        totals = self._prediction_totals(design, team_totals, draws)

        rng = np.random.default_rng(seed)
        counts = np.zeros((len(design.rows), draws), dtype=int)
        for group_i in range(len(design.group_keys)):
            active = design.mask[group_i].astype(bool)
            rows = design.row_idx[group_i, active]
            for draw in range(draws):
                alpha = np.clip(concentration[group_i, active, draw], 1e-8, None)
                probability = rng.dirichlet(alpha)
                counts[rows, draw] = rng.multinomial(totals[group_i, draw], probability)
        return SharePrediction(design.rows, design.group_keys, counts, totals)

    def predict_quantiles(
        self, features: pd.DataFrame, *, team_totals=None, qs=(0.1, 0.5, 0.9)
    ) -> pd.DataFrame:
        pred = self.predict_samples(features, team_totals=team_totals)
        keep = GROUP_COLUMNS + ["player_name", "position"]
        out = pred.rows[keep].copy()
        out["pred_count_mean"] = pred.counts.mean(axis=1)
        out["pred_share_mean"] = pred.shares.mean(axis=1)
        for q in qs:
            suffix = f"p{int(q * 100)}"
            out[f"pred_count_{suffix}"] = np.quantile(pred.counts, q, axis=1)
            out[f"pred_share_{suffix}"] = np.quantile(pred.shares, q, axis=1)
        return out.reset_index(drop=True)


def fit_target_share(features: pd.DataFrame, **sample_kwargs) -> OpportunityShareModel:
    return OpportunityShareModel("target").fit(features, **sample_kwargs)


def fit_carry_share(features: pd.DataFrame, **sample_kwargs) -> OpportunityShareModel:
    return OpportunityShareModel("carry").fit(features, **sample_kwargs)
