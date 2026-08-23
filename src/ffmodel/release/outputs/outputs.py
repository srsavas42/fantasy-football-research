"""Deterministic CSV and plain-text projections of canonical predictions."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from ..contract import ScoringFormat
from ..errors import ReleaseContractError
from .canonical import CanonicalPredictionSet, OutputPlayer


class RankingBasis(str, Enum):
    MEAN = "mean"
    P10 = "p10"
    P50 = "p50"
    P90 = "p90"


@dataclass(frozen=True)
class OutputRenderConfig:
    """Explicit settings for every consumer ranking projection."""

    ranking_basis: RankingBasis
    decimal_places: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.ranking_basis, RankingBasis):
            raise ReleaseContractError("ranking_basis must be an explicit RankingBasis")
        if isinstance(self.decimal_places, bool) or not isinstance(self.decimal_places, int) or not 0 <= self.decimal_places <= 8:
            raise ReleaseContractError("decimal_places must be an integer from 0 through 8")


@dataclass(frozen=True)
class RenderedConsumerOutputs:
    """In-memory output files. T07 owns all staging and publication writes."""

    files: Mapping[str, bytes]

    def __post_init__(self) -> None:
        files = dict(self.files)
        if "predictions.json" not in files:
            raise ReleaseContractError("rendered outputs must contain predictions.json")
        for name, content in files.items():
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ReleaseContractError("output filenames must be plain deterministic filenames")
            if not isinstance(content, bytes):
                raise ReleaseContractError("output content must be bytes")
        object.__setattr__(self, "files", MappingProxyType(dict(sorted(files.items()))))


def _format_number(value: float, decimal_places: int) -> str:
    return format(value, f".{decimal_places}f")


def ranked_players(canonical: CanonicalPredictionSet, scoring_format: ScoringFormat, basis: RankingBasis) -> tuple[OutputPlayer, ...]:
    """Rank canonical records only, with score-descending/player-id tie ordering."""
    if not isinstance(scoring_format, ScoringFormat):
        raise ReleaseContractError("scoring_format must be an explicit ScoringFormat")
    if not isinstance(basis, RankingBasis):
        raise ReleaseContractError("ranking_basis must be an explicit RankingBasis")
    if scoring_format not in canonical.contract.scoring_formats:
        raise ReleaseContractError("scoring_format is not configured for this release")
    return tuple(sorted(canonical.players, key=lambda player: (
        -getattr(player.prediction.scoring[scoring_format], basis.value), player.prediction.player_id,
    )))


def _rows(canonical: CanonicalPredictionSet, scoring_format: ScoringFormat, config: OutputRenderConfig):
    for rank, player in enumerate(ranked_players(canonical, scoring_format, config.ranking_basis), start=1):
        quantiles = player.prediction.scoring[scoring_format]
        provenance = "|".join(item.digest for item in player.provenance)
        yield (rank, player.prediction.player_id, player.prediction.player_name, player.prediction.position.value, player.team, "true" if player.cold_start else "false", provenance, *(_format_number(getattr(quantiles, field), config.decimal_places) for field in ("mean", "p10", "p50", "p90")))


def render_csv(canonical: CanonicalPredictionSet, scoring_format: ScoringFormat, config: OutputRenderConfig) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("target_season", "attempt", "package_digest", "scoring_format", "ranking_basis", "rank", "player_id", "player_name", "position", "team", "cold_start", "provenance_digests", "mean", "p10", "p50", "p90"))
    prefix = (canonical.contract.identity.target_season, canonical.contract.identity.attempt, canonical.package_digest, scoring_format.value, config.ranking_basis.value)
    for row in _rows(canonical, scoring_format, config):
        writer.writerow((*prefix, *row))
    return output.getvalue().encode("utf-8")


def render_text(canonical: CanonicalPredictionSet, scoring_format: ScoringFormat, config: OutputRenderConfig) -> bytes:
    lines = [f"Target season: {canonical.contract.identity.target_season}", f"Attempt: {canonical.contract.identity.attempt}", f"Package digest: {canonical.package_digest}", f"Scoring format: {scoring_format.value}", f"Ranking basis: {config.ranking_basis.value}", ""]
    for rank, player in enumerate(ranked_players(canonical, scoring_format, config.ranking_basis), start=1):
        quantiles = player.prediction.scoring[scoring_format]
        rendered = ", ".join(f"{field}={_format_number(getattr(quantiles, field), config.decimal_places)}" for field in ("mean", "p10", "p50", "p90"))
        cold_start = "; cold-start" if player.cold_start else ""
        provenance = "|".join(item.digest for item in player.provenance)
        lines.append(f"{rank}. {player.prediction.player_name} [{player.prediction.position.value}, {player.team}; {player.prediction.player_id}{cold_start}; provenance={provenance}] — {rendered}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_consumer_outputs(canonical: CanonicalPredictionSet, config: OutputRenderConfig) -> RenderedConsumerOutputs:
    """Render byte-stable consumer files from the canonical release only."""
    canonical = CanonicalPredictionSet.from_dict(canonical.to_dict(), contract=canonical.contract)
    files: dict[str, bytes] = {"predictions.json": canonical.to_bytes()}
    for scoring_format in canonical.contract.scoring_formats:
        files[f"rankings_{scoring_format.value}.csv"] = render_csv(canonical, scoring_format, config)
        files[f"rankings_{scoring_format.value}.txt"] = render_text(canonical, scoring_format, config)
    return RenderedConsumerOutputs(files)
