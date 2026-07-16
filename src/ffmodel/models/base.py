"""Shared PyMC plumbing: sampling defaults, diagnostics, and persistence.

Every model in this package fits through `sample_model` (uniform defaults and
seeding), checks convergence with `convergence_summary` (R-hat / ESS via arviz),
and serializes its posterior with `save_idata` / `load_idata`. PyMC and arviz
are imported lazily so the data/feature layers stay importable without them.
"""

from __future__ import annotations

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


def save_idata(idata, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(path))
    return path


def load_idata(path: str | Path):
    import arviz as az

    return az.from_netcdf(str(path))
