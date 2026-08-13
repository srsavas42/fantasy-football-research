"""Canonical, schema-backed consumer prediction records.

This module converts accepted T01 prediction records into the single consumer
contract used by every release renderer. It deliberately does not write files;
release staging and publication remain lifecycle responsibilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..contract import ReleaseContract
from ..errors import ReleaseContractError, SchemaValidationError
from ..schema import canonical_json_bytes, require_exact_fields, require_mapping
from ..schemas import EvidenceRecord, PlayerPredictionRecord

CANONICAL_OUTPUT_SCHEMA_VERSION = "release-consumer-output.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required_text(value: Any, field_name: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ReleaseContractError(f"{field_name} must be a non-empty string of at most {limit} characters")
    return value.strip()


@dataclass(frozen=True)
class OutputPlayer:
    """T01 prediction plus consumer-only player metadata."""

    prediction: PlayerPredictionRecord
    team: str
    cold_start: bool
    provenance: tuple[EvidenceRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, PlayerPredictionRecord):
            raise ReleaseContractError("prediction must be a PlayerPredictionRecord")
        object.__setattr__(self, "team", _required_text(self.team, "team", limit=32).upper())
        if not isinstance(self.cold_start, bool):
            raise ReleaseContractError("cold_start must be a boolean")
        try:
            provenance = tuple(self.provenance)
        except TypeError as exc:
            raise ReleaseContractError("provenance must be an iterable of EvidenceRecord values") from exc
        if not provenance or not all(isinstance(item, EvidenceRecord) for item in provenance):
            raise ReleaseContractError("provenance must contain at least one EvidenceRecord")
        object.__setattr__(self, "provenance", tuple(sorted(
            provenance, key=lambda item: (item.kind, item.digest, item.source, item.captured_at)
        )))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.prediction.to_dict(),
            "team": self.team,
            "cold_start": self.cold_start,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OutputPlayer":
        data = require_mapping(value, field_name="output player")
        require_exact_fields(data, {
            "schema_version", "player_id", "player_name", "position", "scoring",
            "team", "cold_start", "provenance",
        }, schema=CANONICAL_OUTPUT_SCHEMA_VERSION)
        if not isinstance(data["provenance"], list):
            raise SchemaValidationError("provenance must be an array")
        return cls(
            prediction=PlayerPredictionRecord.from_dict({
                key: data[key]
                for key in ("schema_version", "player_id", "player_name", "position", "scoring")
            }),
            team=data["team"], cold_start=data["cold_start"],
            provenance=tuple(EvidenceRecord.from_dict(item) for item in data["provenance"]),
        )


@dataclass(frozen=True)
class CanonicalPredictionSet:
    """One deterministic consumer release derived from a T01 release contract."""

    contract: ReleaseContract
    package_digest: str
    players: tuple[OutputPlayer, ...]
    schema_version: str = CANONICAL_OUTPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANONICAL_OUTPUT_SCHEMA_VERSION:
            raise ReleaseContractError("unsupported consumer output schema_version")
        if not isinstance(self.contract, ReleaseContract):
            raise ReleaseContractError("contract must be a ReleaseContract")
        if not isinstance(self.package_digest, str) or not _SHA256.fullmatch(self.package_digest):
            raise ReleaseContractError("package_digest must be a lowercase SHA-256 digest")
        try:
            players = tuple(self.players)
        except TypeError as exc:
            raise ReleaseContractError("players must be an iterable of OutputPlayer values") from exc
        if not players or not all(isinstance(item, OutputPlayer) for item in players):
            raise ReleaseContractError("players must contain at least one OutputPlayer")
        player_ids = [item.prediction.player_id for item in players]
        if len(player_ids) != len(set(player_ids)):
            raise ReleaseContractError("players must have unique player_id values")
        for player in players:
            player.prediction.validate_scoring_formats(self.contract.scoring_formats)
        object.__setattr__(self, "players", tuple(sorted(players, key=lambda item: (
            item.prediction.position.value, item.prediction.player_name.casefold(), item.prediction.player_id,
        ))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_season": self.contract.identity.target_season,
            "attempt": self.contract.identity.attempt,
            "package_digest": self.package_digest,
            "scoring_formats": [item.value for item in self.contract.scoring_formats],
            "players": [item.to_dict() for item in self.players],
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, contract: ReleaseContract) -> "CanonicalPredictionSet":
        data = require_mapping(value, field_name="canonical prediction set")
        require_exact_fields(data, {"schema_version", "target_season", "attempt", "package_digest", "scoring_formats", "players"}, schema=CANONICAL_OUTPUT_SCHEMA_VERSION)
        if data["schema_version"] != CANONICAL_OUTPUT_SCHEMA_VERSION:
            raise SchemaValidationError("unsupported consumer output schema_version")
        if data["target_season"] != contract.identity.target_season or data["attempt"] != contract.identity.attempt:
            raise SchemaValidationError("canonical release identity must match the release contract")
        if data["scoring_formats"] != [item.value for item in contract.scoring_formats]:
            raise SchemaValidationError("canonical scoring_formats must exactly match the release contract")
        if not isinstance(data["players"], list):
            raise SchemaValidationError("players must be an array")
        return cls(contract=contract, package_digest=data["package_digest"], players=tuple(OutputPlayer.from_dict(item) for item in data["players"]), schema_version=data["schema_version"])
