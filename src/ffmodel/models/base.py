"""Shared PyMC plumbing: sampling defaults, diagnostics, and persistence.

Every model in this package fits through `sample_model` (uniform defaults and
seeding), checks convergence with `sampling_quality` (R-hat / ESS / divergences),
and serializes its posterior with `save_idata` / `load_idata`. PyMC and arviz
are imported lazily so the data/feature layers stay importable without them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def squeeze_unit(y: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] into the open interval (0, 1) for a Beta likelihood.

    Smithson & Verkuilen (2006): y' = (y*(n-1) + 0.5) / n. Removes exact 0/1,
    which a Beta cannot represent, without distorting interior values much.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    return (y * (n - 1) + 0.5) / n


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def simplex_shares(eta: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Softmax over axis 1, restricted to the entries ``live`` marks as supported.

    ``eta`` is (group, player, draw) and ``live`` is a boolean array of the same
    shape. Unsupported entries get exactly zero rather than a small positive
    share, and a group-draw with nobody supported gets zeros instead of a NaN.
    """
    high = np.max(np.where(live, eta, -np.inf), axis=1, keepdims=True)
    high = np.where(np.isfinite(high), high, 0.0)
    weight = np.where(live, np.exp(eta - high), 0.0)
    total = weight.sum(axis=1, keepdims=True)
    return np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)


def mean_preserving_shares(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    live: np.ndarray,
    *,
    iterations: int = 4,
) -> np.ndarray:
    """Perturbed roster shares whose draw-average matches the baseline allocation.

    The role models add Gaussian innovation to a log-odds vector and then take a
    softmax, which is how season-to-season role churn enters the projection. The
    softmax is not linear, so that noise does not merely spread the shares out —
    it moves their mean. Renormalization takes the surplus from whoever holds the
    most probability mass and hands it to everyone else, so the more concentrated
    the room, the more the leader loses. A quarterback room where the starter
    holds 0.90 gives up 0.0267 of share at the default innovation scale, about
    nine tenths of an attempt per game, and it does so with no model term
    claiming that is true.

    Fix it by solving for a per-player offset, constant across draws, that puts
    the draw-average back where the noiseless allocation had it. The update is
    the usual proportional-fitting step in log space: add the log ratio of the
    target share to the realized one, then recenter within the group, which the
    softmax is invariant to. Four passes bring the residual well inside Monte
    Carlo error for rooms of the size this pipeline allocates over.

    The baseline is the noiseless allocation *including* every other per-draw
    effect — availability, the carry hurdle, the quarterback gate. Those are
    estimated components and are supposed to move the mean; the innovation is a
    dispersion device and is not.
    """
    target = simplex_shares(baseline, live).mean(axis=2)
    offset = np.zeros(baseline.shape[:2], dtype=float)
    tiny = np.finfo(float).tiny
    for _ in range(iterations):
        realized = simplex_shares(perturbed + offset[..., None], live).mean(axis=2)
        step = np.log(np.maximum(target, tiny)) - np.log(np.maximum(realized, tiny))
        offset = offset + np.where((target > 0) & (realized > 0), step, 0.0)
        offset = np.clip(offset - offset.mean(axis=1, keepdims=True), -10.0, 10.0)
    return simplex_shares(perturbed + offset[..., None], live)


def default_sampling_cores(chains: int) -> int:
    """Worker processes to use for ``chains``, from the environment.

    ``FFMODEL_SAMPLING_CORES`` pins this explicitly; ``1`` restores fully serial
    sampling, which is what this package did unconditionally before and is worth
    keeping reachable for debugging and for machines where forking the sampler is
    unreliable. Otherwise chains run in parallel up to the number of CPUs.

    Chains are independent given their seeds, so this changes wall-clock only —
    ``pm.sample`` derives a per-chain seed from ``random_seed`` the same way at
    any core count.
    """
    requested = os.environ.get("FFMODEL_SAMPLING_CORES")
    if requested:
        try:
            pinned = int(requested)
        except ValueError as exc:
            raise ValueError(
                f"FFMODEL_SAMPLING_CORES must be an integer, got {requested!r}"
            ) from exc
        if pinned < 1:
            raise ValueError("FFMODEL_SAMPLING_CORES must be at least 1")
        return min(pinned, chains)
    return max(1, min(chains, os.cpu_count() or 1))


def sample_model(model, draws: int = 1000, tune: int = 1000, chains: int = 4,
                 seed: int = 42, cores: int | None = None, **kwargs):
    """Sample a PyMC model with project defaults; returns an InferenceData."""
    import pymc as pm

    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains,
            cores=default_sampling_cores(chains) if cores is None else cores,
            random_seed=seed, progressbar=False,
            target_accept=kwargs.pop("target_accept", 0.9), **kwargs,
        )


def sampling_quality(
    idata,
    var_names=None,
    *,
    rhat_threshold: float = 1.01,
    min_bulk_ess: float = 100.0,
) -> dict[str, object]:
    """Summarize convergence, effective samples, and NUTS divergences.

    The quality gate deliberately monitors global/variance terms supplied by
    the caller. Requiring every sparse individual-player effect to clear the
    same ESS threshold would make diagnostics noisy without improving a
    projection decision.
    """
    import arviz as az

    summary = az.summary(idata, var_names=var_names)

    # ``az.summary`` defaults to two decimal places, which is useful for a
    # human report but not a hard 1.01 quality threshold. Compute gates from
    # ArviZ's unrounded diagnostics and retain the summary only for display.
    rhat_values = np.asarray(az.rhat(idata, var_names=var_names).to_array()).ravel()
    ess_values = np.asarray(
        az.ess(idata, var_names=var_names, method="bulk").to_array()
    ).ravel()
    rhat_values = rhat_values[np.isfinite(rhat_values)]
    ess_values = ess_values[np.isfinite(ess_values)]
    max_rhat = float(rhat_values.max()) if len(rhat_values) else float("nan")
    min_ess = float(ess_values.min()) if len(ess_values) else float("nan")
    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().item())
    passed = bool(
        np.isfinite(max_rhat)
        and max_rhat < rhat_threshold
        and min_ess >= min_bulk_ess
        and divergences == 0
    )
    return {
        "summary": summary,
        "passed": passed,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_ess,
        "divergences": divergences,
        "rhat_threshold": rhat_threshold,
        "min_bulk_ess_threshold": min_bulk_ess,
    }


def save_idata(idata, path: str | Path) -> Path:
    """Write inference data after converting nested sampler metadata to JSON.

    Nutpie records its sampling configuration as nested dictionaries. NetCDF
    only permits scalar/string attributes, so serialize those values without
    discarding the provenance needed to reproduce a fit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = idata.copy()

    def safe_attributes(attributes):
        cleaned = {}
        for key, value in attributes.items():
            if isinstance(value, (str, bytes, int, float, bool, np.ndarray)):
                cleaned[key] = value
            else:
                cleaned[key] = json.dumps(value, default=str, sort_keys=True)
        return cleaned

    clean.attrs = safe_attributes(clean.attrs)
    for group in clean.groups():
        dataset = getattr(clean, group)
        dataset.attrs = safe_attributes(dataset.attrs)
        for variable in dataset.variables:
            dataset[variable].attrs = safe_attributes(dataset[variable].attrs)
    clean.to_netcdf(str(path))
    return path


def load_idata(path: str | Path):
    import arviz as az

    return az.from_netcdf(str(path))
