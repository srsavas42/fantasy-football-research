"""Coherent posterior pipeline from team opportunity to player opportunity.

The pipeline keeps the modeling layers modular while passing the *same* team
posterior draw through each downstream allocation:

``plays -> passes -> targets`` and ``plays - passes -> carries``.

That is deliberately different from evaluating a share model conditional on a
known final team total.  The latter remains useful for diagnosis; this module
is the projection-time path used for individual-player volume forecasts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ffmodel.models.base import load_idata, sampling_quality, save_idata
from ffmodel.models.volume_share import GROUP_COLUMNS, OpportunityShareModel, SharePrediction
from ffmodel.models.volume_team import TeamVolumeModel, prepare_team_weeks


def _align_team_draws(
    group_keys: pd.DataFrame, team_rows: pd.DataFrame, values: np.ndarray
) -> np.ndarray:
    """Align team posterior draws to an allocation model's sorted groups."""
    lookup = pd.MultiIndex.from_frame(team_rows[GROUP_COLUMNS])
    requested = pd.MultiIndex.from_frame(group_keys[GROUP_COLUMNS])
    positions = lookup.get_indexer(requested)
    if (positions < 0).any():
        raise ValueError("player active set contains a team-week absent from team predictions")
    return values[positions]


@dataclass
class VolumePrediction:
    """Aligned team and player opportunity posterior samples."""

    team: dict[str, object]
    targets: SharePrediction
    carries: SharePrediction

    @property
    def rush_attempts(self) -> np.ndarray:
        return self.team["plays"] - self.team["pass_attempts"]


@dataclass
class VolumePipeline:
    """Fit and project the team-total plus player-allocation volume layers."""

    team_model: TeamVolumeModel = field(default_factory=TeamVolumeModel)
    target_model: OpportunityShareModel = field(
        default_factory=lambda: OpportunityShareModel("target")
    )
    carry_model: OpportunityShareModel = field(
        default_factory=lambda: OpportunityShareModel("carry")
    )
    fit_seconds: dict[str, float] = field(default_factory=dict)

    def fit(self, features: pd.DataFrame, **sample_kwargs) -> "VolumePipeline":
        """Fit team, target-share, and carry-share models on one feature frame."""
        started = perf_counter()
        self.team_model.fit(prepare_team_weeks(features), **sample_kwargs)
        self.fit_seconds["team"] = perf_counter() - started
        started = perf_counter()
        self.target_model.fit(features, **sample_kwargs)
        self.fit_seconds["target"] = perf_counter() - started
        started = perf_counter()
        self.carry_model.fit(features, **sample_kwargs)
        self.fit_seconds["carry"] = perf_counter() - started
        return self

    def predict_samples(self, features: pd.DataFrame, *, seed: int = 0) -> VolumePrediction:
        """Simulate player targets and carries from coherent team-volume draws."""
        team = self.team_model.predict_samples(prepare_team_weeks(features), seed=seed)
        draws = team["plays"].shape[1]

        target_groups = self.target_model.allocation_groups(features)
        target_totals = _align_team_draws(target_groups, team["rows"], team["targets"])
        carry_groups = self.carry_model.allocation_groups(features)
        carry_totals = _align_team_draws(
            carry_groups, team["rows"], team["plays"] - team["pass_attempts"]
        )
        if target_totals.shape[1] != draws or carry_totals.shape[1] != draws:
            raise ValueError("team and allocation posteriors must have the same draw count")

        targets = self.target_model.predict_samples(
            features, team_totals=target_totals, seed=seed + 1
        )
        carries = self.carry_model.predict_samples(
            features, team_totals=carry_totals, seed=seed + 2
        )
        return VolumePrediction(team=team, targets=targets, carries=carries)

    def diagnostics(self, *, min_bulk_ess: float = 100.0) -> dict[str, dict[str, object]]:
        """Return comparable quality gates for the three fitted posteriors."""
        if any(model.idata is None for model in (self.team_model, self.target_model, self.carry_model)):
            raise RuntimeError("fit all volume models before requesting diagnostics")
        team_terms = [
            "play_intercept",
            "pass_intercept",
            "target_intercept",
            "play_alpha",
            "play_team_sd",
            "pass_team_sd",
            "target_team_sd",
        ]
        if self.team_model.use_opponent:
            team_terms.extend(["play_opp_sd", "pass_opp_sd"])
        return {
            "team": sampling_quality(
                self.team_model.idata,
                team_terms,
                min_bulk_ess=min_bulk_ess,
            ),
            "target": sampling_quality(
                self.target_model.idata,
                ["beta", "allocation_concentration"],
                min_bulk_ess=min_bulk_ess,
            ),
            "carry": sampling_quality(
                self.carry_model.idata,
                ["beta", "allocation_concentration"],
                min_bulk_ess=min_bulk_ess,
            ),
        }

    def save(self, directory: str | Path) -> Path:
        """Persist posteriors plus the metadata required for prediction later."""
        if any(model.idata is None for model in (self.team_model, self.target_model, self.carry_model)):
            raise RuntimeError("fit all volume models before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_idata(self.team_model.idata, directory / "team.nc")
        save_idata(self.target_model.idata, directory / "target.nc")
        save_idata(self.carry_model.idata, directory / "carry.nc")
        metadata = {
            "team": {
                "teams": self.team_model.teams,
                "opponents": self.team_model.opponents,
                "season_mean": self.team_model.season_mean,
                "use_opponent": self.team_model.use_opponent,
            },
            "target": self._share_metadata(self.target_model),
            "carry": self._share_metadata(self.carry_model),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return directory

    @staticmethod
    def _share_metadata(model: OpportunityShareModel) -> dict[str, object]:
        return {
            "stream": model.stream,
            "positions": model.positions,
            "players": model.players,
            "feature_names": model.feature_names,
            "feature_fill": model.feature_fill,
            "feature_mean": model.feature_mean,
            "feature_scale": model.feature_scale,
            "position_log_prior": model.position_log_prior,
        }

    @classmethod
    def load(cls, directory: str | Path) -> "VolumePipeline":
        """Restore a saved pipeline for reproducible posterior prediction."""
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        team = TeamVolumeModel(**metadata["team"])
        team.idata = load_idata(directory / "team.nc")
        shares = []
        for name in ("target", "carry"):
            state = metadata[name]
            model = OpportunityShareModel(stream=state["stream"])
            model.positions = list(state["positions"])
            model.players = list(state["players"])
            model.feature_names = list(state["feature_names"])
            model.feature_fill = {k: float(v) for k, v in state["feature_fill"].items()}
            model.feature_mean = {k: float(v) for k, v in state["feature_mean"].items()}
            model.feature_scale = {k: float(v) for k, v in state["feature_scale"].items()}
            model.position_log_prior = {
                k: float(v) for k, v in state.get("position_log_prior", {}).items()
            }
            model.idata = load_idata(directory / f"{name}.nc")
            shares.append(model)
        return cls(team_model=team, target_model=shares[0], carry_model=shares[1])
