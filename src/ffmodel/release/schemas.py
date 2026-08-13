"""Strict, versioned schemas for annual prediction-release artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .contract import CONTRACT_VERSION, Position, ReleaseContract, ReleaseIdentity, SamplerMinimum, ScoringFormat, require_utc
from .errors import ReleaseContractError, SchemaValidationError
from .schema import datetime_from_wire, datetime_to_wire, require_exact_fields, require_mapping

APPROVAL_SCHEMA_VERSION = "release-approval.v1"
EVIDENCE_SCHEMA_VERSION = "release-evidence.v1"
MANIFEST_SCHEMA_VERSION = "release-manifest.v1"
PREDICTION_SCHEMA_VERSION = "release-prediction.v1"
RANKING_SCHEMA_VERSION = "release-ranking.v1"
APPROVAL_VALIDITY = timedelta(hours=24)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_PATH = re.compile(r"(?i)(?:^|[^a-z0-9_])[a-z]:[\\/]")
_UNC_PATH = re.compile(r"(?:^|[^\\])\\\\[^\\/\s]+[\\/]")
# A slash starts an absolute POSIX path when it is not part of a URL/double
# slash and is not a relative ``word/path`` component. This catches arbitrary
# punctuation delimiters (for example ``note;/tmp/reason``) fail-closed.
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_./])/(?!/)")
_SECRET = re.compile(r"(?i)(?:akia[0-9a-z]{16}|ghp_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|sk-[a-z0-9_-]{12,}|(?:api[\s_-]?key|secret|password|token)\s*[:=]\s*\S+)")


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


def _string(value: Any, name: str, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ReleaseContractError(f"{name} must be a non-empty string of at most {limit} characters")
    return value.strip()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReleaseContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def sanitize_approval_reason(reason: str) -> str:
    """Fail closed instead of silently redacting unsafe rationale."""
    if not isinstance(reason, str):
        raise ReleaseContractError("reason must be a string")
    result = reason.strip()
    if len(result) > 1_000:
        raise ReleaseContractError("reason must be at most 1000 characters")
    if _WINDOWS_PATH.search(result) or _UNC_PATH.search(result) or _POSIX_PATH.search(result):
        raise ReleaseContractError("reason must not contain an absolute or UNC path")
    if _SECRET.search(result):
        raise ReleaseContractError("reason must not contain secret-bearing text")
    return result


@dataclass(frozen=True)
class EvidenceRecord:
    kind: str
    digest: str
    source: str
    captured_at: datetime
    schema_version: str = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ReleaseContractError("unsupported evidence schema_version")
        _string(self.kind, "kind", 128); _digest(self.digest, "digest"); _string(self.source, "source", 2048)
        require_utc(self.captured_at, field_name="captured_at")

    def to_dict(self) -> dict[str, str]:
        return {"schema_version": self.schema_version, "kind": self.kind, "digest": self.digest, "source": self.source, "captured_at": datetime_to_wire(self.captured_at)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        data = require_mapping(value, field_name="evidence")
        require_exact_fields(data, {"schema_version", "kind", "digest", "source", "captured_at"}, schema=EVIDENCE_SCHEMA_VERSION)
        return cls(schema_version=data["schema_version"], kind=data["kind"], digest=data["digest"], source=data["source"], captured_at=datetime_from_wire(data["captured_at"], field_name="captured_at"))


@dataclass(frozen=True)
class ReleaseManifest:
    identity: ReleaseIdentity
    contract: ReleaseContract
    evidence: tuple[EvidenceRecord, ...]
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION or not isinstance(self.identity, ReleaseIdentity) or not isinstance(self.contract, ReleaseContract):
            raise ReleaseContractError("manifest has an invalid schema, identity, or contract")
        try:
            evidence = tuple(self.evidence)
        except TypeError as exc:
            raise ReleaseContractError("evidence must be an iterable of typed records") from exc
        if self.identity != self.contract.identity or not all(isinstance(item, EvidenceRecord) for item in evidence):
            raise ReleaseContractError("manifest identity must match contract and evidence must be typed")
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "identity": {"target_season": self.identity.target_season, "attempt": self.identity.attempt}, "contract": {"contract_version": self.contract.contract_version, "scoring_formats": [item.value for item in self.contract.scoring_formats], "sampler_minimum": {"draws": self.contract.sampler_minimum.draws, "tune": self.contract.sampler_minimum.tune, "chains": self.contract.sampler_minimum.chains}, "refit_required": self.contract.refit_required, "artifact_reuse_allowed": self.contract.artifact_reuse_allowed}, "evidence": [item.to_dict() for item in self.evidence]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseManifest":
        data = require_mapping(value, field_name="manifest")
        require_exact_fields(data, {"schema_version", "identity", "contract", "evidence"}, schema=MANIFEST_SCHEMA_VERSION)
        identity_data = require_mapping(data["identity"], field_name="identity")
        require_exact_fields(identity_data, {"target_season", "attempt"}, schema="release-identity.v1")
        identity = ReleaseIdentity(**identity_data)
        contract_data = require_mapping(data["contract"], field_name="contract")
        require_exact_fields(contract_data, {"contract_version", "scoring_formats", "sampler_minimum", "refit_required", "artifact_reuse_allowed"}, schema=CONTRACT_VERSION)
        sampler_data = require_mapping(contract_data["sampler_minimum"], field_name="sampler_minimum")
        require_exact_fields(sampler_data, {"draws", "tune", "chains"}, schema="sampler-minimum.v1")
        if not isinstance(contract_data["scoring_formats"], list) or not isinstance(data["evidence"], list):
            raise SchemaValidationError("scoring_formats and evidence must be arrays")
        return cls(identity=identity, contract=ReleaseContract(identity=identity, scoring_formats=tuple(ScoringFormat(item) for item in contract_data["scoring_formats"]), sampler_minimum=SamplerMinimum(**sampler_data), refit_required=contract_data["refit_required"], artifact_reuse_allowed=contract_data["artifact_reuse_allowed"], contract_version=contract_data["contract_version"]), evidence=tuple(EvidenceRecord.from_dict(item) for item in data["evidence"]), schema_version=data["schema_version"])


@dataclass(frozen=True)
class PredictionQuantiles:
    mean: float
    p10: float
    p50: float
    p90: float

    def __post_init__(self) -> None:
        values = (self.mean, self.p10, self.p50, self.p90)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values) or not self.p10 <= self.p50 <= self.p90:
            raise ReleaseContractError("quantiles must be finite and satisfy p10 <= p50 <= p90")

    def to_dict(self) -> dict[str, float]:
        return {"mean": float(self.mean), "p10": float(self.p10), "p50": float(self.p50), "p90": float(self.p90)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredictionQuantiles":
        data = require_mapping(value, field_name="quantiles")
        require_exact_fields(data, {"mean", "p10", "p50", "p90"}, schema="release-quantiles.v1")
        return cls(**data)


@dataclass(frozen=True)
class PlayerPredictionRecord:
    player_id: str
    player_name: str
    position: Position
    scoring: Mapping[ScoringFormat, PredictionQuantiles]
    schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_SCHEMA_VERSION or not isinstance(self.position, Position):
            raise ReleaseContractError("prediction schema_version or position is invalid")
        _string(self.player_id, "player_id"); _string(self.player_name, "player_name")
        try: normalized = {ScoringFormat(key): value for key, value in self.scoring.items()}
        except (TypeError, ValueError) as exc: raise ReleaseContractError("prediction scoring format is invalid") from exc
        if not normalized or not all(isinstance(value, PredictionQuantiles) for value in normalized.values()):
            raise ReleaseContractError("scoring must contain typed quantiles")
        object.__setattr__(self, "scoring", MappingProxyType(dict(sorted(normalized.items(), key=lambda pair: pair[0].value))))

    def validate_scoring_formats(self, configured: tuple[ScoringFormat, ...]) -> None:
        if set(self.scoring) != set(configured):
            raise ReleaseContractError("prediction scoring formats must exactly match the release contract")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "player_id": self.player_id, "player_name": self.player_name, "position": self.position.value, "scoring": {key.value: value.to_dict() for key, value in self.scoring.items()}}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlayerPredictionRecord":
        data = require_mapping(value, field_name="prediction")
        require_exact_fields(data, {"schema_version", "player_id", "player_name", "position", "scoring"}, schema=PREDICTION_SCHEMA_VERSION)
        scoring = require_mapping(data["scoring"], field_name="scoring")
        try:
            return cls(player_id=data["player_id"], player_name=data["player_name"], position=Position(data["position"]), scoring={ScoringFormat(key): PredictionQuantiles.from_dict(item) for key, item in scoring.items()}, schema_version=data["schema_version"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("prediction position or scoring format is invalid") from exc


@dataclass(frozen=True)
class RankingRecord:
    player_id: str
    position: Position
    scoring_format: ScoringFormat
    rank: int
    schema_version: str = RANKING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RANKING_SCHEMA_VERSION or not isinstance(self.position, Position) or not isinstance(self.scoring_format, ScoringFormat):
            raise ReleaseContractError("ranking schema or enum is invalid")
        _string(self.player_id, "player_id")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1: raise ReleaseContractError("rank must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "player_id": self.player_id, "position": self.position.value, "scoring_format": self.scoring_format.value, "rank": self.rank}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RankingRecord":
        data = require_mapping(value, field_name="ranking")
        require_exact_fields(data, {"schema_version", "player_id", "position", "scoring_format", "rank"}, schema=RANKING_SCHEMA_VERSION)
        try:
            return cls(player_id=data["player_id"], position=Position(data["position"]), scoring_format=ScoringFormat(data["scoring_format"]), rank=data["rank"], schema_version=data["schema_version"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("ranking position or scoring format is invalid") from exc


@dataclass(frozen=True)
class ApprovalRecord:
    """Strict v1 approval binding; T07 exclusively owns its state transition."""
    release_root: str
    target_season: int
    attempt: str
    approver: str
    decision: ApprovalDecision
    staged_release_digest: str
    reason: str
    decided_at: datetime
    expires_at: datetime
    schema_version: str = APPROVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != APPROVAL_SCHEMA_VERSION: raise ReleaseContractError("unsupported approval schema_version")
        _string(self.release_root, "release_root", 2048); ReleaseIdentity(self.target_season, self.attempt); _string(self.approver, "approver", 256); _digest(self.staged_release_digest, "staged_release_digest")
        if not isinstance(self.decision, ApprovalDecision): raise ReleaseContractError("decision must be approve or reject")
        object.__setattr__(self, "reason", sanitize_approval_reason(self.reason))
        decided_at, expires_at = require_utc(self.decided_at, field_name="decided_at"), require_utc(self.expires_at, field_name="expires_at")
        if expires_at != decided_at + APPROVAL_VALIDITY: raise ReleaseContractError("expires_at must be exactly 24 hours after decided_at")
        object.__setattr__(self, "decided_at", decided_at); object.__setattr__(self, "expires_at", expires_at)

    @classmethod
    def create(cls, *, release_root: str, target_season: int, attempt: str, approver: str, decision: ApprovalDecision, staged_release_digest: str, reason: str, decided_at: datetime) -> "ApprovalRecord":
        decided_at = require_utc(decided_at, field_name="decided_at")
        return cls(release_root, target_season, attempt, approver, decision, staged_release_digest, reason, decided_at, decided_at + APPROVAL_VALIDITY)

    def is_valid_at(self, now: datetime, *, staged_release_digest: str) -> bool:
        current = require_utc(now, field_name="now")
        return self.decision is ApprovalDecision.APPROVE and self.staged_release_digest == staged_release_digest and self.decided_at <= current < self.expires_at

    def to_dict(self) -> dict[str, str | int]:
        return {"schema_version": self.schema_version, "release_root": self.release_root, "target_season": self.target_season, "attempt": self.attempt, "approver": self.approver, "decision": self.decision.value, "staged_release_digest": self.staged_release_digest, "reason": self.reason, "decided_at": datetime_to_wire(self.decided_at), "expires_at": datetime_to_wire(self.expires_at)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApprovalRecord":
        data = require_mapping(value, field_name="approval")
        fields = {"schema_version", "release_root", "target_season", "attempt", "approver", "decision", "staged_release_digest", "reason", "decided_at", "expires_at"}
        require_exact_fields(data, fields, schema=APPROVAL_SCHEMA_VERSION)
        try: decision = ApprovalDecision(data["decision"])
        except (TypeError, ValueError) as exc: raise SchemaValidationError("approval decision is invalid") from exc
        return cls(release_root=data["release_root"], target_season=data["target_season"], attempt=data["attempt"], approver=data["approver"], decision=decision, staged_release_digest=data["staged_release_digest"], reason=data["reason"], decided_at=datetime_from_wire(data["decided_at"], field_name="decided_at"), expires_at=datetime_from_wire(data["expires_at"], field_name="expires_at"), schema_version=data["schema_version"])
