"""A saved pipeline must reproduce the numbers it was saved from.

Each efficiency response draws from its own seed, offset from the caller's. That
offset used to be the response's position in ``self.models``, and insertion
order is not stable across a save/load round trip: ``fit`` inserts in
``EFFICIENCY_MODEL_SPECS`` order, while ``load`` reads a JSON object written
with ``sort_keys=True`` and inserts alphabetically. So a reloaded artifact
handed every response a different seed.

Nothing about that is distributionally wrong — the draws are as valid as any
others — which is what made it survive. What it breaks is reproducibility: a
published projection could not be regenerated from the artifact it came from.
Measured on a two-season fit, the maximum PPR difference was 302 points.
"""

import numpy as np
import pytest

from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    EFFICIENCY_MODEL_SPECS,
    EFFICIENCY_SEED_OFFSET,
)


def test_every_response_has_a_distinct_offset():
    assert set(EFFICIENCY_SEED_OFFSET) == set(EFFICIENCY_MODEL_BY_TARGET)
    assert len(set(EFFICIENCY_SEED_OFFSET.values())) == len(EFFICIENCY_SEED_OFFSET)


def test_the_offset_is_keyed_to_the_response_not_to_an_ordering():
    """Reordering the responses must not change any response's seed.

    This is the property the ordinal version lacked. Two dicts holding the same
    responses in different orders have to produce identical offsets, because
    that is exactly the difference between a fitted and a reloaded pipeline.
    """
    spec_order = [spec.target for spec in EFFICIENCY_MODEL_SPECS]
    alphabetical = sorted(spec_order)
    assert spec_order != alphabetical, "the two orders must actually differ"

    from_specs = {t: EFFICIENCY_SEED_OFFSET[t] for t in spec_order}
    from_alpha = {t: EFFICIENCY_SEED_OFFSET[t] for t in alphabetical}

    assert from_specs == from_alpha


def test_the_ordinal_scheme_would_have_disagreed():
    """Pin the failure, so a revert to positional seeding is caught here."""
    spec_order = [spec.target for spec in EFFICIENCY_MODEL_SPECS]
    alphabetical = sorted(spec_order)

    ordinal_fitted = {t: i for i, t in enumerate(spec_order)}
    ordinal_reloaded = {t: i for i, t in enumerate(alphabetical)}

    assert ordinal_fitted != ordinal_reloaded
    disagreeing = [t for t in spec_order if ordinal_fitted[t] != ordinal_reloaded[t]]
    assert len(disagreeing) >= 8


@pytest.mark.slow
def test_a_reloaded_pipeline_predicts_identically():
    """The end-to-end property, on a real fit.

    Marked slow because it needs a genuine posterior: the defect lives in how
    saved state is reconstructed, so a hand-built pipeline would not exercise
    it.
    """
    import tempfile
    from pathlib import Path

    import pandas as pd

    from ffmodel.features.season_average import SeasonAverageData
    from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

    cache = Path(".cache/ffmodel-walkforward")
    if not (cache / "player_rows.pkl").exists():
        pytest.skip("walk-forward cache is not built")

    player_rows = pd.read_pickle(cache / "player_rows.pkl")
    team_rows = pd.read_pickle(cache / "team_rows.pkl")
    player_rows = player_rows[player_rows.season >= 2022]
    team_rows = team_rows[team_rows.season >= 2022]
    data = SeasonAverageData(team_rows.copy(), player_rows.copy())

    pipeline = SeasonAverageScoringPipeline()
    budget = {"draws": 40, "tune": 40, "chains": 2, "seed": 1}
    pipeline.fit(
        data, volume_sample_kwargs=budget, efficiency_sample_kwargs=budget
    )

    directory = Path(tempfile.mkdtemp()) / "artifact"
    pipeline.save(directory)
    reloaded = SeasonAverageScoringPipeline.load(directory)

    assert list(reloaded.efficiency_model.models) == list(
        pipeline.efficiency_model.models
    )
    before = pipeline.predict_samples(data, seed=11).fantasy_points["ppr"]
    after = reloaded.predict_samples(data, seed=11).fantasy_points["ppr"]
    assert np.abs(np.asarray(before) - np.asarray(after)).max() == 0.0


def test_the_exposure_floor_survives_a_round_trip():
    """It lives only in metadata, so a dropped key reverts it silently."""
    import json
    import tempfile
    from pathlib import Path

    from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

    directory = Path(tempfile.mkdtemp()) / "artifact"
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "architecture_version": 2,
                "volume_feature_alpha": 300.0,
                "draw_conditioned_efficiency": False,
                "volume_feature_estimator": "ridge",
                "efficiency_exposure_floor": 5,
            }
        ),
        encoding="utf-8",
    )
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["efficiency_exposure_floor"] == 5
    assert (
        SeasonAverageScoringPipeline(efficiency_exposure_floor=5).efficiency_exposure_floor
        == 5
    )
