"""Team-conserving coupling of sampled player-season regimes to role shares.

This is deliberately an isolated ablation primitive. It does not change the
accepted volume pipeline until its walk-forward effect is measured. A sampled
regime tilts a player's existing share, then the original team/group total is
restored exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ffmodel.models.season_regime import REGIME_NAMES, RegimeThresholds, realized_regimes


STREAM_COLUMNS = {
    "pass": "observed_qb_workload_share",
    "target": "target_share",
    "carry": "carry_share",
}


def _number(rows: pd.DataFrame, name: str) -> np.ndarray:
    value = rows.get(name)
    if value is None:
        return np.zeros(len(rows), dtype=float)
    return pd.to_numeric(value, errors="coerce").fillna(0.0).to_numpy(float)


@dataclass
class SeasonRegimeRoleCoupling:
    """Empirical state effects used to tilt already-fitted role shares."""

    positions: tuple[str, ...] = ("QB", "RB", "WR", "TE")
    effect_cap: float = 2.0
    thresholds: RegimeThresholds | None = field(default=None, init=False)
    effects: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def fit(
        self, rows: pd.DataFrame, thresholds: RegimeThresholds | None = None
    ) -> "SeasonRegimeRoleCoupling":
        self.thresholds = thresholds
        if self.thresholds is None:
            from ffmodel.models.season_regime import fit_regime_thresholds

            self.thresholds = fit_regime_thresholds(rows)
        labels = realized_regimes(rows, self.thresholds)
        position = rows.get("position", pd.Series("", index=rows.index)).fillna("").to_numpy()
        replacement = labels == "replacement"
        for stream, column in STREAM_COLUMNS.items():
            value = np.maximum(_number(rows, column), 0.0)
            effect = np.zeros((len(REGIME_NAMES), len(self.positions)), dtype=float)
            for position_index, position_name in enumerate(self.positions):
                position_rows = (position == position_name) & ~replacement
                base_values = value[position_rows]
                base = float(np.mean(base_values)) if base_values.size else 0.0
                if base <= 0.0:
                    continue
                for state_index, state_name in enumerate(REGIME_NAMES[1:], start=1):
                    state_values = value[position_rows & (labels == state_name)]
                    state_mean = float(np.mean(state_values)) if state_values.size else base
                    effect[state_index, position_index] = np.clip(
                        np.log((state_mean + 1e-4) / (base + 1e-4)),
                        -self.effect_cap,
                        self.effect_cap,
                    )
            self.effects[stream] = effect
        return self

    def apply(
        self,
        rows: pd.DataFrame,
        shares: np.ndarray,
        regime_samples: np.ndarray,
        *,
        stream: str,
        group_index: np.ndarray,
    ) -> np.ndarray:
        """Tilt shares and restore each group's original sum for every draw."""

        if stream not in self.effects:
            raise ValueError(f"unknown or unfitted role stream: {stream}")
        shares = np.asarray(shares, dtype=float)
        regime_samples = np.asarray(regime_samples, dtype=int)
        group_index = np.asarray(group_index, dtype=int)
        if shares.ndim != 2 or regime_samples.shape != shares.shape:
            raise ValueError("shares and regime_samples must have the same 2D shape")
        position = rows.get("position", pd.Series("", index=rows.index)).fillna("")
        position_index = position.map({name: index for index, name in enumerate(self.positions)}).fillna(-1).to_numpy(int)
        effects = self.effects[stream]
        adjusted = np.maximum(shares, 0.0).copy()
        for row_index in range(len(rows)):
            if position_index[row_index] >= 0:
                adjusted[row_index] *= np.exp(
                    effects[regime_samples[row_index], position_index[row_index]]
                )
        for group in np.unique(group_index):
            members = np.flatnonzero(group_index == group)
            original_total = shares[members].sum(axis=0)
            adjusted_total = adjusted[members].sum(axis=0)
            for draw in range(shares.shape[1]):
                if adjusted_total[draw] > 0.0:
                    adjusted[members, draw] *= original_total[draw] / adjusted_total[draw]
                elif original_total[draw] > 0.0:
                    adjusted[members, draw] = shares[members, draw]
        if not np.allclose(
            np.add.reduceat(
                adjusted[np.argsort(group_index)],
                np.r_[0, np.flatnonzero(np.diff(np.sort(group_index))) + 1],
                axis=0,
            ),
            np.add.reduceat(
                shares[np.argsort(group_index)],
                np.r_[0, np.flatnonzero(np.diff(np.sort(group_index))) + 1],
                axis=0,
            ),
            atol=1e-10,
        ):
            raise AssertionError("regime coupling violated a group share total")
        return adjusted

    def state_dict(self) -> dict[str, object]:
        if self.thresholds is None or not self.effects:
            raise RuntimeError("fit the regime coupling before serializing it")
        return {
            "positions": list(self.positions),
            "effect_cap": self.effect_cap,
            "thresholds": {
                "lead_role_threshold": self.thresholds.lead_role_threshold,
                "inactive_availability": self.thresholds.inactive_availability,
                "inactive_role": self.thresholds.inactive_role,
            },
            "effects": {name: value.tolist() for name, value in self.effects.items()},
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SeasonRegimeRoleCoupling":
        coupling = cls(
            positions=tuple(state["positions"]), effect_cap=float(state["effect_cap"])
        )
        threshold = state["thresholds"]
        coupling.thresholds = RegimeThresholds(
            lead_role_threshold={
                str(name): float(value)
                for name, value in threshold["lead_role_threshold"].items()
            },
            inactive_availability=float(threshold["inactive_availability"]),
            inactive_role=float(threshold["inactive_role"]),
        )
        coupling.effects = {
            str(name): np.asarray(value, dtype=float)
            for name, value in state["effects"].items()
        }
        return coupling
