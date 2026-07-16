"""Shared PyMC plumbing: sampling defaults, diagnostics, and persistence.

Every model in this package fits through `sample_model` (uniform defaults and
seeding), checks convergence with `convergence_summary` (R-hat / ESS via arviz),
and serializes its posterior with `save_idata` / `load_idata`. PyMC and arviz
are imported lazily so the data/feature layers stay importable without them.
"""

from __future__ import annotations

import json
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


def sample_model(model, draws: int = 1000, tune: int = 1000, chains: int = 4,
                 seed: int = 42, **kwargs):
    """Sample a PyMC model with project defaults; returns an InferenceData."""
    import pymc as pm

    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, cores=1,
            random_seed=seed, progressbar=False,
            target_accept=kwargs.pop("target_accept", 0.9), **kwargs,
        )


def convergence_summary(idata, var_names=None):
    """Return an arviz summary and a boolean 'converged' (all R-hat < 1.01)."""
    import arviz as az

    summ = az.summary(idata, var_names=var_names)
    converged = bool((summ["r_hat"] < 1.01).all())
    return summ, converged


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
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
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
