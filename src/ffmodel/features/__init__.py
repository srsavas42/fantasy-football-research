"""Feature engineering: raw player-week stat lines -> model-ready covariates.

The single entry point is `build_features`, which turns the canonical stat
frame from `ffmodel.data.load_player_weeks` into the enriched frame the
Phase 3 volume model consumes.
"""

from ffmodel.features.build import FEATURE_COLUMNS, build_features

__all__ = ["build_features", "FEATURE_COLUMNS"]
