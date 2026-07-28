"""End-to-end posterior pipeline for total season fantasy scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from ffmodel.evaluation.efficiency_season_average import (
    add_walk_forward_volume_features,
)
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.efficiency_season_average import (
    SeasonAveragePosteriorEfficiencyPipeline,
)
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline
from ffmodel.simulation.season_scoring import (
    REQUIRED_EFFICIENCY_TARGETS,
    SeasonScoringPrediction,
    score_volume_prediction,
)


@dataclass
class SeasonAverageScoringPipeline:
    """Fit volume-v3 and posterior efficiency, then simulate fantasy points."""

    volume_model: SeasonAverageVolumePipeline = field(
        default_factory=SeasonAverageVolumePipeline
    )
    efficiency_model: SeasonAveragePosteriorEfficiencyPipeline = field(
        default_factory=SeasonAveragePosteriorEfficiencyPipeline
    )
    volume_feature_alpha: float = 300.0
    draw_conditioned_efficiency: bool = False

    def fit_efficiency(
        self,
        data: SeasonAverageData,
        **sample_kwargs,
    ) -> "SeasonAverageScoringPipeline":
        """Fit efficiency with volume projections cross-fitted by response year."""
        rows = add_walk_forward_volume_features(
            data,
            include_efficiency=True,
            alpha=self.volume_feature_alpha,
        )
        self.efficiency_model.fit(rows, **sample_kwargs)
        missing = set(REQUIRED_EFFICIENCY_TARGETS) - set(self.efficiency_model.models)
        if missing:
            raise ValueError(
                f"total scoring requires efficiency models: {sorted(missing)}"
            )
        return self

    def fit(
        self,
        data: SeasonAverageData,
        *,
        volume_sample_kwargs: dict[str, object] | None = None,
        efficiency_sample_kwargs: dict[str, object] | None = None,
    ) -> "SeasonAverageScoringPipeline":
        """Fit both layers while keeping their sampler controls independent."""
        self.volume_model.fit(data, **(volume_sample_kwargs or {}))
        return self.fit_efficiency(data, **(efficiency_sample_kwargs or {}))

    def predict_samples(
        self,
        data: SeasonAverageData,
        *,
        games=None,
        seed: int = 0,
    ) -> SeasonScoringPrediction:
        volume = self.volume_model.predict_samples(data, games=games, seed=seed)
        return score_volume_prediction(
            volume,
            self.efficiency_model,
            draw_conditioned_efficiency=self.draw_conditioned_efficiency,
            seed=seed + 10_000,
        )

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.volume_model.save(directory / "volume")
        self.efficiency_model.save(directory / "efficiency")
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "architecture_version": 2,
                    "volume_feature_alpha": self.volume_feature_alpha,
                    "draw_conditioned_efficiency": self.draw_conditioned_efficiency,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "SeasonAverageScoringPipeline":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            volume_model=SeasonAverageVolumePipeline.load(directory / "volume"),
            efficiency_model=SeasonAveragePosteriorEfficiencyPipeline.load(
                directory / "efficiency"
            ),
            volume_feature_alpha=float(metadata.get("volume_feature_alpha", 300.0)),
            draw_conditioned_efficiency=bool(
                metadata.get("draw_conditioned_efficiency", False)
            ),
        )
