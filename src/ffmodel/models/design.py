"""Standardized design matrices shared by the season models.

Every layer that regresses on prior-season features builds its design the same
way: keep the candidates that actually vary, centre and scale each one on
constants recorded at fit time, and -- where the layer asks for it -- rotate
onto the SVD basis so collinear columns cannot hand the sampler an
unidentified direction. The scaling constants live on the calling model
(``feature_fill`` / ``feature_mean`` / ``feature_scale``) because they are part
of what a fitted model serializes; these helpers read and write those dicts
rather than owning them.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def varying_features(
    rows: pd.DataFrame, *candidates: Iterable[str]
) -> list[str]:
    """Names present in ``rows`` with at least one value and real spread.

    Candidate groups are concatenated in order and de-duplicated, so a base
    feature set and a model's opt-in extras can be passed separately. A column
    that is constant carries no information and would leave the sampler a flat
    direction, so it is dropped rather than scaled by its zero spread.
    """
    names: list[str] = []
    seen: set[str] = set()
    for group in candidates:
        for name in group:
            if name in seen or name not in rows:
                continue
            seen.add(name)
            values = pd.to_numeric(rows[name], errors="coerce")
            if values.notna().any() and values.fillna(0).std(ddof=0) > 1e-8:
                names.append(name)
    return names


def standardize(
    rows: pd.DataFrame,
    names: Iterable[str],
    fill: dict[str, float],
    mean: dict[str, float],
    scale: dict[str, float],
    *,
    fit: bool = False,
) -> np.ndarray:
    """Centre and scale ``names`` into a design matrix.

    When ``fit``, the median fill and the resulting mean and spread are
    recorded into the three dicts first; otherwise the recorded constants are
    reused, which is what keeps a holdout row on the training scale.
    """
    columns: list[np.ndarray] = []
    for name in names:
        values = pd.to_numeric(rows[name], errors="coerce")
        if fit:
            centre = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(centre)
            spread = float(filled.std(ddof=0))
            fill[name] = centre
            mean[name] = float(filled.mean())
            scale[name] = spread if spread > 1e-8 else 1.0
        standardized = values.fillna(fill[name]).to_numpy(dtype=float)
        columns.append((standardized - mean[name]) / scale[name])
    return np.column_stack(columns) if columns else np.zeros((len(rows), 0))


def collinearity_projection(matrix: np.ndarray) -> np.ndarray:
    """Basis spanning the design's numerical rank.

    Right singular vectors above the rank tolerance, so projecting through it
    drops the collinear directions a shared feature prior cannot identify.
    """
    if not matrix.shape[1]:
        return np.zeros((0, 0), dtype=float)
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = (
        max(matrix.shape) * np.finfo(float).eps * singular_values.max(initial=0.0)
    )
    rank = int((singular_values > tolerance).sum())
    return right[:rank].T


def project(matrix: np.ndarray, projection: np.ndarray | None) -> np.ndarray:
    """Rotate a design onto a fitted basis; ``None`` passes it through."""
    if projection is None:
        return matrix
    return matrix @ np.asarray(projection, dtype=float)
