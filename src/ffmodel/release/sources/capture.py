"""Fail-closed immutable capture and replay of release runtime inputs.

The capture boundary stores bytes before a release transforms them into frames.
Consequently replay never needs, or is permitted, to call a network loader.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from ..contract import ReleaseIdentity, require_utc
from ..errors import ReleaseContractError, SchemaValidationError
from ..schema import canonical_json_bytes, datetime_from_wire, datetime_to_wire, require_exact_fields, require_mapping, sha256_digest

SOURCE_MANIFEST_SCHEMA_VERSION = "release-source-manifest.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_ATTEMPT_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SECRET_NAME = re.compile(r"(?i)(?:api[_-]?key|secret|token|password|credential)")
_SECRET_VALUE = re.compile(r"(?i)(?:akia[0-9a-z]{16}|ghp_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|sk-[a-z0-9_-]{12,})")
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{index}" for index in range(1, 10)), *(f"lpt{index}" for index in range(1, 10))}


class SourceCaptureError(ReleaseContractError):
    """A runtime input cannot be safely captured or replayed."""


def _safe_attempt_component(value: str) -> str:
    """Restrict the contract attempt label before it becomes a filesystem path."""
    if not isinstance(value, str) or not _ATTEMPT_COMPONENT.fullmatch(value):
        raise SourceCaptureError("attempt must be a single Windows-safe path component")
    if value.casefold() in _WINDOWS_RESERVED:
        raise SourceCaptureError("attempt must not be a Windows reserved device name")
    return value


def _safe_source_url(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise SourceCaptureError("source_uri must be a non-empty string of at most 2048 characters")
    parsed = urlsplit(value)
    if parsed.username or parsed.password or any(_SECRET_NAME.search(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
        raise SourceCaptureError("source_uri must not include credentials or secret-bearing query parameters")
    return value


def _safe_metadata(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceCaptureError(f"{field_name} must be a mapping")
    try:
        result = json.loads(canonical_json_bytes(dict(value)))
    except SchemaValidationError as exc:
        raise SourceCaptureError(f"{field_name} must contain canonical JSON values") from exc

    def visit(item: Any, key: str = "") -> None:
        if _SECRET_NAME.search(key):
            raise SourceCaptureError(f"{field_name} must not include a secret-bearing key")
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and _SECRET_VALUE.search(item):
            raise SourceCaptureError(f"{field_name} must not include secret-bearing text")

    visit(result)
    return result


def json_schema_fingerprint(payload: bytes) -> str:
    """Hash only JSON structure, so schema drift is detectable before decoding."""
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceCaptureError("source payload is not valid UTF-8 JSON") from exc

    def shape(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): shape(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            shapes = {canonical_json_bytes(shape(child)).decode("utf-8") for child in item}
            return {"array": [json.loads(item) for item in sorted(shapes)]}
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "boolean"
        if isinstance(item, (int, float)):
            return "number"
        if isinstance(item, str):
            return "string"
        raise SourceCaptureError("source JSON contains an unsupported value")

    return sha256_digest(canonical_json_bytes(shape(value)))


@dataclass(frozen=True)
class SourceSpec:
    """One source acquired exactly once for a release attempt."""

    source_id: str
    source_uri: str
    loader: Callable[[], bytes]
    expected_schema_fingerprint: str
    schema_fingerprint: Callable[[bytes], str] = json_schema_fingerprint
    max_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _SOURCE_ID.fullmatch(self.source_id):
            raise SourceCaptureError("source_id must be a lowercase stable identifier")
        _safe_source_url(self.source_uri)
        if not callable(self.loader) or not callable(self.schema_fingerprint):
            raise SourceCaptureError("source loaders and schema validators must be callable")
        if not isinstance(self.expected_schema_fingerprint, str) or not _SHA256.fullmatch(self.expected_schema_fingerprint):
            raise SourceCaptureError("expected_schema_fingerprint must be a lowercase SHA-256 digest")
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes < 1:
            raise SourceCaptureError("max_bytes must be a positive integer")
        self.filename  # validate the platform-safe deterministic projection

    @property
    def filename(self) -> str:
        # Stable ASCII-only names avoid platform-specific normalization surprises.
        stem = self.source_id.replace(".", "_").replace("-", "_")
        if stem.casefold() in _WINDOWS_RESERVED:
            raise SourceCaptureError("source_id yields a Windows reserved filename")
        return f"{stem}.json"


@dataclass(frozen=True)
class CapturedSource:
    source_id: str
    filename: str
    source_uri: str
    digest: str
    schema_fingerprint: str
    retrieved_at: datetime
    byte_count: int

    def __post_init__(self) -> None:
        if not _SOURCE_ID.fullmatch(self.source_id) or self.filename != SourceSpec(self.source_id, self.source_uri, lambda: b"{}", "a" * 64).filename:
            raise SourceCaptureError("captured source identifier or filename is invalid")
        _safe_source_url(self.source_uri)
        if not _SHA256.fullmatch(self.digest) or not _SHA256.fullmatch(self.schema_fingerprint):
            raise SourceCaptureError("captured source digests must be lowercase SHA-256")
        require_utc(self.retrieved_at, field_name="retrieved_at")
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int) or self.byte_count < 1:
            raise SourceCaptureError("captured source byte_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "filename": self.filename, "source_uri": self.source_uri, "digest": self.digest, "schema_fingerprint": self.schema_fingerprint, "retrieved_at": datetime_to_wire(self.retrieved_at), "byte_count": self.byte_count}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapturedSource":
        data = require_mapping(value, field_name="captured source")
        require_exact_fields(data, {"source_id", "filename", "source_uri", "digest", "schema_fingerprint", "retrieved_at", "byte_count"}, schema=SOURCE_MANIFEST_SCHEMA_VERSION)
        return cls(**{**data, "retrieved_at": datetime_from_wire(data["retrieved_at"], field_name="retrieved_at")})


@dataclass(frozen=True)
class SourceManifest:
    identity: ReleaseIdentity
    cutoff: datetime
    code_identity: str
    runtime_identity: Mapping[str, Any]
    configuration: Mapping[str, Any]
    sources: tuple[CapturedSource, ...]
    schema_version: str = SOURCE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_MANIFEST_SCHEMA_VERSION:
            raise SourceCaptureError("unsupported source manifest schema version")
        if not isinstance(self.identity, ReleaseIdentity):
            raise SourceCaptureError("manifest identity must be a ReleaseIdentity")
        if require_utc(self.cutoff, field_name="cutoff") != self.identity.cutoff:
            raise SourceCaptureError("manifest cutoff must equal the deterministic target-season cutoff")
        if not isinstance(self.code_identity, str) or not self.code_identity.strip() or len(self.code_identity) > 512:
            raise SourceCaptureError("code_identity must be a non-empty string of at most 512 characters")
        object.__setattr__(self, "runtime_identity", _safe_metadata(self.runtime_identity, field_name="runtime_identity"))
        object.__setattr__(self, "configuration", _safe_metadata(self.configuration, field_name="configuration"))
        entries = tuple(self.sources)
        if not entries or not all(isinstance(item, CapturedSource) for item in entries):
            raise SourceCaptureError("manifest sources must be a non-empty collection of captured sources")
        if [item.source_id for item in entries] != sorted(item.source_id for item in entries):
            raise SourceCaptureError("manifest sources must have deterministic source-id order")
        if len({item.source_id for item in entries}) != len(entries):
            raise SourceCaptureError("manifest source identifiers must be unique")
        if len({item.filename for item in entries}) != len(entries):
            raise SourceCaptureError("manifest source filenames must be unique")
        object.__setattr__(self, "sources", entries)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "identity": {"target_season": self.identity.target_season, "attempt": self.identity.attempt}, "cutoff": datetime_to_wire(self.cutoff), "code_identity": self.code_identity, "runtime_identity": dict(self.runtime_identity), "configuration": dict(self.configuration), "sources": [item.to_dict() for item in self.sources]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceManifest":
        data = require_mapping(value, field_name="source manifest")
        require_exact_fields(data, {"schema_version", "identity", "cutoff", "code_identity", "runtime_identity", "configuration", "sources"}, schema=SOURCE_MANIFEST_SCHEMA_VERSION)
        identity_data = require_mapping(data["identity"], field_name="identity")
        require_exact_fields(identity_data, {"target_season", "attempt"}, schema="release-identity.v1")
        if not isinstance(data["sources"], list):
            raise SchemaValidationError("source manifest sources must be an array")
        return cls(identity=ReleaseIdentity(**identity_data), cutoff=datetime_from_wire(data["cutoff"], field_name="cutoff"), code_identity=data["code_identity"], runtime_identity=data["runtime_identity"], configuration=data["configuration"], sources=tuple(CapturedSource.from_dict(item) for item in data["sources"]), schema_version=data["schema_version"])


class ImmutableSourceCapture:
    """Acquire attempt inputs once, then replay only verified captured bytes."""

    def __init__(self, release_root: str | Path, identity: ReleaseIdentity) -> None:
        self.release_root = Path(release_root)
        self.identity = identity
        self._attempt = _safe_attempt_component(identity.attempt)

    @property
    def attempt_directory(self) -> Path:
        return self.release_root / "attempts" / str(self.identity.target_season) / self._attempt

    @property
    def manifest_path(self) -> Path:
        return self.attempt_directory / "source-manifest.json"

    def capture(self, specs: tuple[SourceSpec, ...] | list[SourceSpec], *, code_identity: str, runtime_identity: Mapping[str, Any], configuration: Mapping[str, Any], now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> SourceManifest:
        entries = tuple(specs)
        if not entries or not all(isinstance(item, SourceSpec) for item in entries):
            raise SourceCaptureError("capture requires one or more SourceSpec entries")
        if len({item.source_id for item in entries}) != len(entries):
            raise SourceCaptureError("capture source identifiers must be unique")
        if len({item.filename for item in entries}) != len(entries):
            raise SourceCaptureError("capture source filenames collide after Windows-safe normalization")
        if self.attempt_directory.exists():
            raise SourceCaptureError("attempt source directory already exists; immutable inputs cannot be recaptured")
        self.attempt_directory.mkdir(parents=True, exist_ok=False)
        inputs_directory = self.attempt_directory / "inputs"
        inputs_directory.mkdir()
        captured: list[CapturedSource] = []
        for spec in sorted(entries, key=lambda item: item.source_id):
            try:
                payload = spec.loader()
            except Exception as exc:
                raise SourceCaptureError(f"source {spec.source_id!r} could not be downloaded") from exc
            if not isinstance(payload, bytes) or not payload or len(payload) > spec.max_bytes:
                raise SourceCaptureError(f"source {spec.source_id!r} is incomplete or exceeds its configured size limit")
            try:
                fingerprint = spec.schema_fingerprint(payload)
            except Exception as exc:
                raise SourceCaptureError(f"source {spec.source_id!r} has invalid or drifted schema") from exc
            if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
                raise SourceCaptureError(f"source {spec.source_id!r} schema validator did not return a SHA-256 fingerprint")
            if fingerprint != spec.expected_schema_fingerprint:
                raise SourceCaptureError(f"source {spec.source_id!r} schema fingerprint does not match the configured expected schema")
            path = inputs_directory / spec.filename
            path.write_bytes(payload)
            captured.append(CapturedSource(spec.source_id, spec.filename, spec.source_uri, sha256_digest(payload), fingerprint, require_utc(now(), field_name="retrieved_at"), len(payload)))
        manifest = SourceManifest(self.identity, self.identity.cutoff, code_identity, runtime_identity, configuration, tuple(captured))
        self.manifest_path.write_bytes(canonical_json_bytes(manifest.to_dict()))
        return manifest

    def replay(self, specs: tuple[SourceSpec, ...] | list[SourceSpec]) -> tuple[SourceManifest, dict[str, bytes]]:
        """Read only captured files. No loader is invoked on this path."""
        if not self.manifest_path.is_file():
            raise SourceCaptureError("captured source manifest is missing")
        try:
            manifest = SourceManifest.from_dict(json.loads(self.manifest_path.read_bytes().decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, ReleaseContractError) as exc:
            raise SourceCaptureError("captured source manifest is malformed") from exc
        if manifest.identity != self.identity:
            raise SourceCaptureError("captured source manifest belongs to a different release attempt")
        supplied_specs = tuple(specs)
        if not all(isinstance(item, SourceSpec) for item in supplied_specs):
            raise SourceCaptureError("replay requires typed SourceSpec entries")
        by_id = {item.source_id: item for item in supplied_specs}
        if len(by_id) != len(supplied_specs):
            raise SourceCaptureError("replay source identifiers must be unique")
        if set(by_id) != {item.source_id for item in manifest.sources}:
            raise SourceCaptureError("replay source specification does not exactly match capture manifest")
        replayed: dict[str, bytes] = {}
        for entry in manifest.sources:
            spec = by_id[entry.source_id]
            if spec.filename != entry.filename or spec.source_uri != entry.source_uri:
                raise SourceCaptureError(f"replay source specification changed for {entry.source_id!r}")
            path = self.attempt_directory / "inputs" / entry.filename
            if not path.is_file():
                raise SourceCaptureError(f"captured source {entry.source_id!r} is missing")
            payload = path.read_bytes()
            if len(payload) != entry.byte_count or sha256_digest(payload) != entry.digest:
                raise SourceCaptureError(f"captured source {entry.source_id!r} digest does not match its manifest")
            try:
                fingerprint = spec.schema_fingerprint(payload)
            except Exception as exc:
                raise SourceCaptureError(f"captured source {entry.source_id!r} no longer has a valid schema") from exc
            if fingerprint != entry.schema_fingerprint or fingerprint != spec.expected_schema_fingerprint:
                raise SourceCaptureError(f"captured source {entry.source_id!r} schema fingerprint does not match its manifest")
            replayed[entry.source_id] = payload
        return manifest, replayed
