"""Reproducible checkpoints and acceptance gates for posterior comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GateSpec:
    """Metrics and fitted components required to promote one variant."""

    primary: tuple[str, ...]
    end_to_end: tuple[str, ...]
    protected: tuple[str, ...] = ()
    components: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def atomic_write_json(path: str | Path, payload: object) -> Path:
    """Write JSON through a sibling temporary file and atomically replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: str | Path) -> str:
    """Hash file names and contents under a fitted baseline directory."""
    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_fingerprint(item)))
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame, *, keys: Sequence[str] = ()) -> str:
    """Return an order-stable content fingerprint for a model input frame."""
    columns = sorted(frame.columns)
    ordered = frame.loc[:, columns].copy()
    sort_keys = [name for name in keys if name in ordered]
    if sort_keys:
        ordered = ordered.sort_values(sort_keys, kind="mergesort")
    ordered = ordered.reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
    digest.update(
        json.dumps([str(ordered[name].dtype) for name in columns]).encode("utf-8")
    )
    try:
        values = pd.util.hash_pandas_object(
            ordered, index=False, categorize=True
        ).to_numpy(dtype=np.uint64)
    except TypeError:
        values = pd.util.hash_pandas_object(
            ordered.astype(str), index=False, categorize=True
        ).to_numpy(dtype=np.uint64)
    digest.update(values.tobytes())
    return digest.hexdigest()


def combined_fingerprint(parts: Mapping[str, object]) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def posterior_sample_count(posterior) -> int:
    """Return the flattened chain-by-draw sample count for a posterior dataset."""
    missing = {"chain", "draw"} - set(posterior.sizes)
    if missing:
        raise ValueError(f"posterior is missing sampling dimensions: {sorted(missing)}")
    return int(posterior.sizes["chain"] * posterior.sizes["draw"])


def select_posterior_samples(posterior, sample_count: int):
    """Select evenly spaced posterior samples and return one aligned chain.

    Candidate components may be fitted with longer chains than an accepted
    baseline. Posterior predictions only require independent paired draws, so
    deterministic thinning provides a common draw axis while retaining the
    full fitted posterior for diagnostics and persistence.
    """
    available = posterior_sample_count(posterior)
    if sample_count <= 0 or sample_count > available:
        raise ValueError(
            f"sample_count must be between 1 and {available}, got {sample_count}"
        )
    if sample_count == available:
        return posterior
    indices = np.linspace(0, available - 1, sample_count, dtype=int)
    selected = posterior.stack(_posterior_sample=("chain", "draw")).isel(
        _posterior_sample=indices
    )
    selected = selected.reset_index("_posterior_sample", drop=True)
    return selected.rename({"_posterior_sample": "draw"}).expand_dims(chain=[0])


def ensure_manifest(
    path: str | Path,
    expected: Mapping[str, object],
    *,
    resume: bool,
) -> dict[str, object]:
    """Create or validate a holdout manifest before using cached artifacts."""
    destination = Path(path)
    expected_payload = dict(expected)
    expected_payload["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    if destination.exists():
        current = load_json(destination)
        if current != expected_payload:
            raise ValueError(
                f"checkpoint manifest does not match this run: {destination}"
            )
        if not resume:
            raise FileExistsError(
                f"checkpoint already exists; pass --resume to reuse it: {destination}"
            )
        return current
    if resume and any(destination.parent.iterdir()):
        raise ValueError(
            "cannot resume legacy or partial artifacts without a matching manifest: "
            f"{destination.parent}"
        )
    atomic_write_json(destination, expected_payload)
    return expected_payload


def _records_by_season(
    records: Iterable[Mapping[str, object]], model: str
) -> dict[int, Mapping[str, object]]:
    return {
        int(record["season"]): record
        for record in records
        if record.get("model") == model
    }


def _metric_weight(record: Mapping[str, object], metric: str) -> float:
    layer = metric.rsplit("_", 1)[0]
    return float(record.get(f"{layer}_n", 1.0))


def _pooled_metric(records: Sequence[Mapping[str, object]], metric: str) -> float:
    values = np.asarray([float(record[metric]) for record in records], dtype=float)
    weights = np.asarray(
        [_metric_weight(record, metric) for record in records], dtype=float
    )
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        return float("nan")
    return float(np.average(values, weights=np.clip(weights, 1.0, None)))


def _metric_check(
    baseline: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    metric: str,
    *,
    min_wins: int,
    min_relative_improvement: float,
) -> dict[str, object]:
    baseline_value = _pooled_metric(baseline, metric)
    candidate_value = _pooled_metric(candidate, metric)
    relative_improvement = (
        (baseline_value - candidate_value) / abs(baseline_value)
        if baseline_value != 0
        else float("nan")
    )
    wins = sum(
        float(candidate_record[metric]) < float(baseline_record[metric])
        for baseline_record, candidate_record in zip(baseline, candidate, strict=True)
    )
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "relative_improvement": relative_improvement,
        "wins": int(wins),
        "folds": len(candidate),
        "passed": bool(
            np.isfinite(relative_improvement)
            and relative_improvement > min_relative_improvement
            and wins >= min_wins
        ),
    }


def evaluate_gate(
    records: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
    candidate: str,
    spec: GateSpec,
    *,
    expected_holdouts: Sequence[int],
    baseline: str = "volume_v2",
    min_wins: int = 2,
    min_relative_improvement: float = 0.0,
    max_protected_regression: float = 0.005,
) -> dict[str, object]:
    """Apply strict predictive, cross-fold, and sampler-quality promotion gates."""
    baseline_by_season = _records_by_season(records, baseline)
    candidate_by_season = _records_by_season(records, candidate)
    expected = tuple(int(season) for season in expected_holdouts)
    scored = tuple(
        season
        for season in expected
        if season in baseline_by_season and season in candidate_by_season
    )
    baseline_rows = [baseline_by_season[season] for season in scored]
    candidate_rows = [candidate_by_season[season] for season in scored]
    effective_min_wins = min(min_wins, len(scored))

    required_checks = {
        metric: _metric_check(
            baseline_rows,
            candidate_rows,
            metric,
            min_wins=effective_min_wins,
            min_relative_improvement=min_relative_improvement,
        )
        for metric in dict.fromkeys((*spec.primary, *spec.end_to_end))
    }
    protected_checks = {}
    for metric in spec.protected:
        check = _metric_check(
            baseline_rows,
            candidate_rows,
            metric,
            min_wins=0,
            min_relative_improvement=-max_protected_regression,
        )
        check["passed"] = bool(
            np.isfinite(check["relative_improvement"])
            and check["relative_improvement"] >= -max_protected_regression
        )
        protected_checks[metric] = check

    diagnostic_index = {
        (int(item["season"]), str(item["component"])): bool(item["passed"])
        for item in diagnostics
    }
    diagnostic_checks = {
        f"{season}:{component}": diagnostic_index.get((season, component), False)
        for season in scored
        for component in spec.components
    }
    complete = scored == expected
    predictive_passed = bool(required_checks) and all(
        check["passed"] for check in required_checks.values()
    )
    protected_passed = all(check["passed"] for check in protected_checks.values())
    diagnostics_passed = bool(diagnostic_checks) and all(diagnostic_checks.values())
    passed = predictive_passed and protected_passed and diagnostics_passed
    status = "accepted" if complete and passed else "rejected" if complete else "pending"
    return {
        "candidate": candidate,
        "reference": baseline,
        "status": status,
        "holdouts_scored": list(scored),
        "holdouts_expected": list(expected),
        "required_metrics": required_checks,
        "protected_metrics": protected_checks,
        "diagnostics": diagnostic_checks,
        "checks": {
            "complete": complete,
            "predictive": predictive_passed,
            "protected_streams": protected_passed,
            "sampling_quality": diagnostics_passed,
        },
    }
