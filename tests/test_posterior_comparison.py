"""Contracts for resumable posterior comparisons and promotion gates."""

import numpy as np
import pytest

az = pytest.importorskip("arviz")

from ffmodel.evaluation.posterior_comparison import (
    GateSpec,
    atomic_write_json,
    ensure_manifest,
    evaluate_gate,
    frame_fingerprint,
    load_json,
    posterior_sample_count,
    select_posterior_samples,
)


def _records(candidate="candidate", seasons=(2022, 2023, 2024)):
    records = []
    for season in seasons:
        records.extend(
            [
                {
                    "season": season,
                    "model": "volume_v2",
                    "target_n": 100,
                    "target_mae": 1.0,
                    "target_crps": 0.8,
                    "carry_n": 100,
                    "carry_mae": 0.9,
                    "carry_crps": 0.7,
                },
                {
                    "season": season,
                    "model": candidate,
                    "target_n": 100,
                    "target_mae": 0.95,
                    "target_crps": 0.75,
                    "carry_n": 100,
                    "carry_mae": 0.902,
                    "carry_crps": 0.702,
                },
            ]
        )
    return records


def _diagnostics(candidate="candidate", seasons=(2022, 2023, 2024), passed=True):
    return [
        {"season": season, "component": candidate, "passed": passed}
        for season in seasons
    ]


def test_atomic_manifest_is_reused_only_with_exact_configuration(tmp_path):
    path = tmp_path / "run" / "manifest.json"
    payload = {"draws": 300, "features": ["history"]}

    ensure_manifest(path, payload, resume=False)

    assert load_json(path)["draws"] == 300
    assert ensure_manifest(path, payload, resume=True)["draws"] == 300
    with pytest.raises(FileExistsError):
        ensure_manifest(path, payload, resume=False)
    with pytest.raises(ValueError):
        ensure_manifest(path, {"draws": 301}, resume=True)


def test_atomic_json_replaces_complete_payload(tmp_path):
    path = tmp_path / "result.json"
    atomic_write_json(path, {"state": "first"})
    atomic_write_json(path, {"state": "complete", "folds": [2022]})

    assert load_json(path) == {"state": "complete", "folds": [2022]}


def test_frame_fingerprint_is_stable_under_key_ordering():
    import pandas as pd

    rows = pd.DataFrame(
        {"season": [2024, 2023], "player_key": ["b", "a"], "value": [2.0, 1.0]}
    )
    reordered = rows.iloc[::-1].reset_index(drop=True)

    assert frame_fingerprint(rows, keys=("season", "player_key")) == frame_fingerprint(
        reordered, keys=("season", "player_key")
    )


def test_posterior_samples_are_deterministically_aligned():
    idata = az.from_dict(
        posterior={"theta": np.arange(24, dtype=float).reshape(2, 6, 2)}
    )

    selected = select_posterior_samples(idata.posterior, 5)

    assert posterior_sample_count(idata.posterior) == 12
    assert posterior_sample_count(selected) == 5
    assert selected.sizes["chain"] == 1
    expected = idata.posterior["theta"].stack(sample=("chain", "draw")).to_numpy()[
        :, np.linspace(0, 11, 5, dtype=int)
    ]
    actual = selected["theta"].stack(sample=("chain", "draw")).to_numpy()
    assert np.array_equal(actual, expected)
    with pytest.raises(ValueError):
        select_posterior_samples(idata.posterior, 13)


def test_gate_accepts_complete_cross_fold_improvement_with_clean_diagnostics():
    spec = GateSpec(
        primary=("target_mae", "target_crps"),
        end_to_end=("target_mae", "target_crps"),
        protected=("carry_mae", "carry_crps"),
        components=("candidate",),
    )

    result = evaluate_gate(
        _records(),
        _diagnostics(),
        "candidate",
        spec,
        expected_holdouts=(2022, 2023, 2024),
    )

    assert result["status"] == "accepted"
    assert result["checks"] == {
        "complete": True,
        "predictive": True,
        "protected_streams": True,
        "sampling_quality": True,
    }


def test_gate_stays_pending_until_all_holdouts_are_scored():
    spec = GateSpec(
        primary=("target_mae",),
        end_to_end=("target_crps",),
        components=("candidate",),
    )

    result = evaluate_gate(
        _records(seasons=(2022,)),
        _diagnostics(seasons=(2022,)),
        "candidate",
        spec,
        expected_holdouts=(2022, 2023, 2024),
    )

    assert result["status"] == "pending"
    assert result["checks"]["complete"] is False


def test_gate_rejects_bad_diagnostics_or_material_stream_regression():
    spec = GateSpec(
        primary=("target_mae", "target_crps"),
        end_to_end=("target_mae", "target_crps"),
        protected=("carry_mae", "carry_crps"),
        components=("candidate",),
    )
    records = _records()
    for record in records:
        if record["model"] == "candidate":
            record["carry_mae"] = 0.92

    result = evaluate_gate(
        records,
        _diagnostics(passed=False),
        "candidate",
        spec,
        expected_holdouts=(2022, 2023, 2024),
    )

    assert result["status"] == "rejected"
    assert result["checks"]["protected_streams"] is False
    assert result["checks"]["sampling_quality"] is False
