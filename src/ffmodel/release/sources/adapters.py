"""Narrow, capture-only adapters for the release's required source systems."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlencode

from ffmodel.data import sleeper
from ffmodel.data.providers import nflverse

from .capture import SourceCaptureError, SourceSpec
from ..schema import canonical_json, sha256_digest


def sleeper_players_schema_fingerprint(payload: bytes) -> str:
    """Fingerprint Sleeper player-record structure, excluding dynamic player IDs.

    Sleeper's top-level object is keyed by player IDs. Those values turn over
    normally, so only the nested player-record shape belongs to the schema.
    """
    try:
        records = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceCaptureError("Sleeper players payload is not valid UTF-8 JSON") from exc
    if not isinstance(records, dict) or not records:
        raise SourceCaptureError("Sleeper players payload must be a non-empty object of player records")

    def shape(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): shape(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            shapes = {canonical_json(shape(child)) for child in item}
            return {"array": [json.loads(value) for value in sorted(shapes)]}
        if item is None:
            return "null"
        if isinstance(item, bool):
            return "boolean"
        if isinstance(item, (int, float)):
            return "number"
        if isinstance(item, str):
            return "string"
        raise SourceCaptureError("Sleeper players payload contains an unsupported value")

    record_shapes: list[str] = []
    for player_id, record in records.items():
        if not isinstance(player_id, str) or not player_id or not isinstance(record, dict):
            raise SourceCaptureError("Sleeper players payload must map non-empty IDs to player objects")
        record_shapes.append(canonical_json(shape(record)))
    return sha256_digest(canonical_json({"dynamic_player_records": [json.loads(value) for value in sorted(set(record_shapes))]}).encode("utf-8"))


def sleeper_players_source(*, expected_schema_fingerprint: str, loader: Callable[[], bytes] | None = None) -> SourceSpec:
    """Return the authoritative current Sleeper roster source specification."""
    return SourceSpec("sleeper.players", "https://api.sleeper.app/v1/players/nfl", loader or sleeper.capture_players_payload, expected_schema_fingerprint, sleeper_players_schema_fingerprint)


def nflverse_dataset_source(
    dataset: str,
    seasons: int | Iterable[int] | None = None,
    *,
    expected_schema_fingerprint: str,
    loader: Callable[[], bytes] | None = None,
    **params: Any,
) -> SourceSpec:
    """Return a replayable source spec for one explicitly configured dataset."""
    if not isinstance(dataset, str) or dataset not in nflverse.DATASETS:
        raise ValueError(f"unknown nflverse dataset {dataset!r}")
    normalized_seasons = (seasons,) if isinstance(seasons, int) else (tuple(seasons) if seasons is not None else None)
    request = {"dataset": dataset, "seasons": normalized_seasons, "params": params}
    request_json = canonical_json(request)
    request_digest = sha256_digest(request_json.encode("utf-8"))
    identifier = f"nflverse.{dataset}.{request_digest[:16]}"
    source_uri = f"nflverse://{dataset}?{urlencode({'request': request_json})}"
    return SourceSpec(identifier, source_uri, loader or (lambda: nflverse.capture_dataset_payload(dataset, normalized_seasons, **params)), expected_schema_fingerprint)
