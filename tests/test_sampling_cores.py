"""Chain parallelism is a wall-clock choice, not a modelling one.

``sample_model`` used to hardcode ``cores=1``, so every model in the pipeline ran
its chains one after another. The season-average pipeline fits eight components,
so a validation fold paid that serially eight times over. Chains are independent
given their seeds, so running them in parallel changes only how long it takes.
"""

import os

import pytest

from ffmodel.models.base import default_sampling_cores


def test_parallelism_is_capped_by_the_chain_count(monkeypatch):
    monkeypatch.delenv("FFMODEL_SAMPLING_CORES", raising=False)

    # Never more workers than there is work: two chains cannot use four cores.
    assert default_sampling_cores(2) <= 2
    assert default_sampling_cores(1) == 1


def test_parallelism_is_capped_by_the_machine(monkeypatch):
    monkeypatch.delenv("FFMODEL_SAMPLING_CORES", raising=False)

    assert default_sampling_cores(64) <= (os.cpu_count() or 1)


def test_the_environment_can_pin_it_back_to_serial(monkeypatch):
    # Serial sampling is what this package did unconditionally before, and it
    # stays reachable: forking the sampler is unreliable on some machines, and
    # a serial run is easier to attach a debugger to.
    monkeypatch.setenv("FFMODEL_SAMPLING_CORES", "1")

    assert default_sampling_cores(4) == 1


def test_a_pin_above_the_chain_count_is_clamped(monkeypatch):
    monkeypatch.setenv("FFMODEL_SAMPLING_CORES", "16")

    assert default_sampling_cores(3) == 3


@pytest.mark.parametrize("value", ["0", "-2", "many"])
def test_an_unusable_pin_is_rejected_rather_than_silently_ignored(monkeypatch, value):
    # Falling back to a default here would let a typo in a benchmark script
    # quietly change how long every subsequent comparison took.
    monkeypatch.setenv("FFMODEL_SAMPLING_CORES", value)

    with pytest.raises(ValueError, match="FFMODEL_SAMPLING_CORES"):
        default_sampling_cores(4)


@pytest.mark.slow
def test_core_count_does_not_move_the_posterior():
    """Same seed, same answer, whatever the core count.

    This is the property that makes the change safe to default on: if chains
    were seeded from the worker rather than from ``random_seed``, parallelism
    would silently change every fitted result.
    """
    import numpy as np

    pm = pytest.importorskip("pymc")
    from ffmodel.models.base import sample_model

    observed = np.random.default_rng(0).normal(2.0, 1.0, 200)

    def fit(cores: int):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0, 5)
            sigma = pm.HalfNormal("sigma", 5)
            pm.Normal("obs", mu, sigma, observed=observed)
        idata = sample_model(model, draws=250, tune=250, chains=2, seed=42, cores=cores)
        return float(idata.posterior["mu"].mean()), float(idata.posterior["sigma"].mean())

    assert fit(1) == pytest.approx(fit(2), abs=1e-12)
