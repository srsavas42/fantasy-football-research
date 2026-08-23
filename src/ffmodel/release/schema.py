"""Strict deterministic serialization helpers for release schema v1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .errors import SchemaValidationError


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("value is not canonical JSON") from exc


def canonical_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_exact_fields(mapping: Mapping[str, Any], expected: set[str], *, schema: str) -> None:
    actual = set(mapping)
    if actual != expected:
        parts = []
        if expected - actual:
            parts.append(f"missing={sorted(expected - actual)}")
        if actual - expected:
            parts.append(f"unknown={sorted(actual - expected)}")
        raise SchemaValidationError(f"{schema} fields are invalid: {', '.join(parts)}")


def require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{field_name} must be an object")
    return value


def datetime_to_wire(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise SchemaValidationError("timestamp must be timezone-aware UTC")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def datetime_from_wire(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field_name} must be an RFC3339 UTC string")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(f"{field_name} is not a valid timestamp") from exc
    if result.tzinfo is None or result.utcoffset() != timezone.utc.utcoffset(result):
        raise SchemaValidationError(f"{field_name} must be UTC")
    return result.astimezone(timezone.utc)
