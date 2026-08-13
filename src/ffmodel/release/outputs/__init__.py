"""Deterministic consumer-output construction for annual releases."""

from .canonical import CANONICAL_OUTPUT_SCHEMA_VERSION, CanonicalPredictionSet, OutputPlayer
from .outputs import OutputRenderConfig, RankingBasis, RenderedConsumerOutputs, ranked_players, render_consumer_outputs

__all__ = ["CANONICAL_OUTPUT_SCHEMA_VERSION", "CanonicalPredictionSet", "OutputPlayer", "OutputRenderConfig", "RankingBasis", "RenderedConsumerOutputs", "ranked_players", "render_consumer_outputs"]
