"""Local parquet cache with immutable, point-in-time source snapshots.

The original API (``get_or_fetch("weekly", ..., season=2024)``) remains
supported so existing callers and caches keep working. New providers should
pass ``provider``, request ``params``, and ``as_of`` for mutable data. Those
fields become part of the path and are recorded in a JSON manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from ffmodel.config import CACHE_DIR


@dataclass(frozen=True)
class DatasetKey:
    """Fields that uniquely identify a cached dataset artifact."""

    provider: str
    dataset: str
    season: int | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    as_of: str | date | datetime | None = None
    schema_version: int = 1


def _safe(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(value)).strip("-.")
    return text or "none"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _params_hash(params: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def cache_path(
    dataset: str | DatasetKey,
    season: int | None = None,
    cache_dir: Path | None = None,
    *,
    provider: str | None = None,
    params: Mapping[str, Any] | None = None,
    as_of: str | date | datetime | None = None,
    schema_version: int = 1,
) -> Path:
    root = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    if isinstance(dataset, DatasetKey):
        key = dataset
    elif provider is not None:
        key = DatasetKey(provider, dataset, season, params or {}, as_of, schema_version)
    else:
        # Backwards-compatible flat cache path.
        name = f"{dataset}_{season}.parquet" if season is not None else f"{dataset}.parquet"
        return root / name

    path = root / "raw" / _safe(key.provider) / _safe(key.dataset)
    path /= f"schema=v{key.schema_version}"
    if key.season is not None:
        path /= f"season={key.season}"
    if key.params:
        path /= f"params={_params_hash(key.params)}"
    if key.as_of is not None:
        path /= f"as_of={_safe(_jsonable(key.as_of))}"
    return path / "data.parquet"


def manifest_path(path: Path) -> Path:
    return path.with_name("manifest.json")


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        return version("ffmodel")
    except PackageNotFoundError:
        return "editable"


def _write_manifest(
    path: Path,
    df: pd.DataFrame,
    key: DatasetKey,
    *,
    source_url: str | None,
    license_name: str | None,
) -> None:
    schema = {column: str(dtype) for column, dtype in df.dtypes.items()}
    payload = {
        **asdict(key),
        "params": _jsonable(key.params),
        "as_of": _jsonable(key.as_of),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "license": license_name,
        "rows": len(df),
        "columns": list(df.columns),
        "schema_fingerprint": hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sha256": _file_checksum(path),
        "ffmodel_version": _package_version(),
    }
    manifest_path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """Read the manifest next to a structured cache artifact."""
    return json.loads(manifest_path(path).read_text(encoding="utf-8"))


def get_or_fetch(
    dataset: str | DatasetKey,
    fetch: Callable[[], pd.DataFrame],
    season: int | None = None,
    refresh: bool = False,
    cache_dir: Path | None = None,
    *,
    provider: str | None = None,
    params: Mapping[str, Any] | None = None,
    as_of: str | date | datetime | None = None,
    schema_version: int = 1,
    source_url: str | None = None,
    license_name: str | None = None,
) -> pd.DataFrame:
    """Return a cached frame, fetching and writing parquet on a cache miss."""
    path = cache_path(
        dataset,
        season,
        cache_dir,
        provider=provider,
        params=params,
        as_of=as_of,
        schema_version=schema_version,
    )
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"fetch for {dataset!r} returned {type(df).__name__}, not DataFrame")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    key = dataset if isinstance(dataset, DatasetKey) else None
    if key is None and provider is not None:
        key = DatasetKey(provider, dataset, season, params or {}, as_of, schema_version)
    if key is not None:
        _write_manifest(
            path, df, key, source_url=source_url, license_name=license_name
        )
    return df
