"""Resumable posterior-controlled tests for stable volume pathways."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.evaluation.posterior_comparison import (
    GateSpec,
    atomic_write_json,
    combined_fingerprint,
    directory_fingerprint,
    ensure_manifest,
    evaluate_gate,
    file_fingerprint,
    frame_fingerprint,
    load_json,
    posterior_sample_count,
    select_posterior_samples,
)
from ffmodel.features.season_average import SeasonAverageData, build_season_average_data
from ffmodel.models.base import load_idata, sampling_quality, save_idata
from ffmodel.models.season_availability import SeasonAvailabilityModel
from ffmodel.models.season_opportunity import (
    QBPassPropensityModel,
    CARRY_ELIGIBILITY_EFFICIENCY_FEATURES,
    SNAP_HISTORY_FEATURES,
    SeasonCarryEligibilityModel,
    SeasonSnapShareModel,
    SeasonTargetRoleModel,
)
from ffmodel.models.volume_season_average import (
    SeasonAverageVolumePipeline,
    SeasonRosterShareModel,
)


SNAP_HISTORY = SNAP_HISTORY_FEATURES
SNAP_EFFICIENCY = (
    "prior_role_quality_signal",
    "prior_role_quality_signal_3yr",
    "prior_role_quality_signal_trend",
    "prior_role_room_quality_advantage",
)
QB_HISTORY = (
    "prior_qb_attempts_per_snap_3yr",
    "prior_qb_attempts_per_snap_trend",
    "prior_pass_role_3yr",
)
TARGET_HISTORY = (
    "prior_target_per_snap_3yr",
    "prior_target_per_snap_trend",
    "prior_target_role_3yr",
)
TARGET_EFFICIENCY = (
    "prior_rec_quality_signal",
    "prior_rec_quality_signal_3yr",
    "prior_rec_quality_signal_trend",
    "prior_rec_room_quality_advantage",
)
CARRY_HISTORY = (
    "prior_carry_per_snap_3yr",
    "prior_carry_per_snap_trend",
    "prior_carry_role_3yr",
)
ELIGIBILITY_EFFICIENCY = CARRY_ELIGIBILITY_EFFICIENCY_FEATURES
AVAILABILITY_HISTORY = (
    "prior_availability_3yr",
    "prior_availability_trend",
    "prior_snap_share_3yr",
    "prior_snap_share_trend",
)

CANDIDATE_ORDER = (
    "snap_history",
    "target_history",
    "carry_history",
    "qb_history",
    "carry_eligibility_efficiency",
    "snap_history_efficiency",
    "target_history_efficiency",
    "target_role_history",
    "availability_history",
    "availability_calibrated",
)

VARIANTS = OrderedDict(
    (
        ("volume_v2", {}),
        ("snap_history", {"snap": "snap_history"}),
        ("target_history", {"target": "target_history"}),
        ("carry_history", {"carry": "carry_history"}),
        ("qb_history", {"qb": "qb_history"}),
        (
            "all_history",
            {
                "snap": "snap_history",
                "target": "target_history",
                "carry": "carry_history",
                "qb": "qb_history",
            },
        ),
        (
            "carry_eligibility_efficiency",
            {"eligibility": "carry_eligibility_efficiency"},
        ),
        (
            "volume_v3",
            {
                "snap": "snap_history",
                "eligibility": "carry_eligibility_efficiency",
            },
        ),
        (
            "snap_history_efficiency",
            {"snap": "snap_history_efficiency"},
        ),
        (
            "target_history_efficiency",
            {"target": "target_history_efficiency"},
        ),
        (
            "target_role_history",
            {"target_role": "target_role_history"},
        ),
        (
            "target_history_role",
            {
                "target": "target_history",
                "target_role": "target_role_history",
            },
        ),
        (
            "combined",
            {
                "snap": "snap_history_efficiency",
                "target": "target_history_efficiency",
                "carry": "carry_history",
                "qb": "qb_history",
                "eligibility": "carry_eligibility_efficiency",
            },
        ),
        (
            "availability_history",
            {"availability": "availability_history"},
        ),
        (
            "availability_calibrated",
            {"availability": "availability_calibrated"},
        ),
    )
)

ALL_STREAM_METRICS = (
    "pass_mae",
    "pass_crps",
    "target_mae",
    "target_crps",
    "carry_mae",
    "carry_crps",
)

GATE_SPECS = {
    "snap_history": GateSpec(
        primary=("snap_mae", "snap_crps"),
        end_to_end=("target_mae", "target_crps", "carry_mae", "carry_crps"),
        protected=("pass_mae", "pass_crps"),
        components=("snap_history",),
    ),
    "target_history": GateSpec(
        primary=("target_mae", "target_crps"),
        end_to_end=("target_mae", "target_crps"),
        protected=("pass_mae", "pass_crps", "carry_mae", "carry_crps"),
        components=("target_history",),
    ),
    "carry_history": GateSpec(
        primary=("carry_mae", "carry_crps"),
        end_to_end=("carry_mae", "carry_crps"),
        protected=("pass_mae", "pass_crps", "target_mae", "target_crps"),
        components=("carry_history",),
    ),
    "qb_history": GateSpec(
        primary=("qb_propensity_mae", "qb_propensity_crps"),
        end_to_end=("pass_mae", "pass_crps"),
        protected=("target_mae", "target_crps", "carry_mae", "carry_crps"),
        components=("qb_history",),
    ),
    "all_history": GateSpec(
        primary=(
            "snap_mae",
            "snap_crps",
            "qb_propensity_mae",
            "qb_propensity_crps",
        ),
        end_to_end=ALL_STREAM_METRICS,
        components=("snap_history", "target_history", "carry_history", "qb_history"),
    ),
    "carry_eligibility_efficiency": GateSpec(
        primary=("carry_eligibility_brier",),
        end_to_end=("carry_mae", "carry_crps"),
        protected=("pass_mae", "pass_crps", "target_mae", "target_crps"),
        components=("carry_eligibility_efficiency",),
    ),
    "volume_v3": GateSpec(
        primary=("snap_mae", "snap_crps", "carry_eligibility_brier"),
        end_to_end=("target_mae", "target_crps", "carry_mae", "carry_crps"),
        protected=("pass_mae", "pass_crps"),
        components=("snap_history", "carry_eligibility_efficiency"),
    ),
    "snap_history_efficiency": GateSpec(
        primary=("snap_mae", "snap_crps"),
        end_to_end=("target_mae", "target_crps", "carry_mae", "carry_crps"),
        protected=("pass_mae", "pass_crps"),
        components=("snap_history_efficiency",),
    ),
    "target_history_efficiency": GateSpec(
        primary=("target_mae", "target_crps"),
        end_to_end=("target_mae", "target_crps"),
        protected=("pass_mae", "pass_crps", "carry_mae", "carry_crps"),
        components=("target_history_efficiency",),
    ),
    "target_role_history": GateSpec(
        primary=("target_role_brier",),
        end_to_end=("target_mae", "target_crps"),
        protected=("pass_mae", "pass_crps", "carry_mae", "carry_crps"),
        components=("target_role_history",),
    ),
    "target_history_role": GateSpec(
        primary=("target_role_brier",),
        end_to_end=("target_mae", "target_crps"),
        protected=("pass_mae", "pass_crps", "carry_mae", "carry_crps"),
        components=("target_history", "target_role_history"),
    ),
    "combined": GateSpec(
        primary=(
            "snap_mae",
            "snap_crps",
            "qb_propensity_mae",
            "qb_propensity_crps",
            "carry_eligibility_brier",
        ),
        end_to_end=ALL_STREAM_METRICS,
        components=(
            "snap_history_efficiency",
            "target_history_efficiency",
            "carry_history",
            "qb_history",
            "carry_eligibility_efficiency",
        ),
    ),
    "availability_history": GateSpec(
        primary=(
            "availability_mae",
            "availability_crps",
            "availability_coverage_error",
            "availability_any_brier",
        ),
        end_to_end=ALL_STREAM_METRICS,
        components=("availability_history",),
    ),
    "availability_calibrated": GateSpec(
        primary=(
            "availability_mae",
            "availability_crps",
            "availability_coverage_error",
            "availability_any_brier",
        ),
        end_to_end=ALL_STREAM_METRICS,
        components=("availability_calibrated",),
    ),
}

GATE_REFERENCES = {
    "snap_history_efficiency": "snap_history",
    "target_history_efficiency": "target_history",
    "combined": "all_history",
    "availability_calibrated": "availability_history",
    "target_history_role": "target_history",
}


def _candidate_configuration(label: str) -> dict[str, object]:
    configurations = {
        "snap_history": {"kind": "snap", "features": SNAP_HISTORY},
        "snap_history_efficiency": {
            "kind": "snap",
            "features": SNAP_HISTORY + SNAP_EFFICIENCY,
        },
        "qb_history": {"kind": "qb", "features": QB_HISTORY},
        "target_history": {"kind": "target", "features": TARGET_HISTORY},
        "target_history_efficiency": {
            "kind": "target",
            "features": TARGET_HISTORY + TARGET_EFFICIENCY,
        },
        "target_role_history": {
            "kind": "target_role",
            "features": TARGET_HISTORY,
        },
        "carry_history": {"kind": "carry", "features": CARRY_HISTORY},
        "carry_eligibility_efficiency": {
            "kind": "eligibility",
            "features": ELIGIBILITY_EFFICIENCY,
        },
        "availability_history": {
            "kind": "availability",
            "features": AVAILABILITY_HISTORY,
            "position_specific_concentration": False,
        },
        "availability_calibrated": {
            "kind": "availability",
            "features": AVAILABILITY_HISTORY,
            "position_specific_concentration": True,
        },
    }
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in configurations[label].items()
    }


def _new_candidate(label: str):
    configuration = _candidate_configuration(label)
    features = tuple(configuration["features"])
    kind = configuration["kind"]
    if kind == "snap":
        return SeasonSnapShareModel(extra_features=features)
    if kind == "qb":
        return QBPassPropensityModel(extra_features=features)
    if kind in {"target", "carry"}:
        return SeasonRosterShareModel(kind, extra_efficiency_features=features)
    if kind == "eligibility":
        return SeasonCarryEligibilityModel(extra_features=features)
    if kind == "target_role":
        return SeasonTargetRoleModel(extra_features=features)
    if kind == "availability":
        return SeasonAvailabilityModel(
            extra_features=features,
            position_specific_concentration=bool(
                configuration["position_specific_concentration"]
            ),
        )
    raise ValueError(f"unknown candidate kind: {kind}")


def _distribution(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    mean = samples.mean(axis=1)
    error = mean - observed
    return {
        "n": int(len(observed)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "crps": float(empirical_crps(observed, samples).mean()),
    }


def _metrics(prediction) -> dict[str, object]:
    rows = prediction.player_rows
    named = pd.to_numeric(
        rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
        errors="coerce",
    ).fillna(0).ne(1).to_numpy()
    team_games = pd.to_numeric(rows["team_games"], errors="coerce").to_numpy(float)
    result: dict[str, object] = {}
    for name, count, samples in (
        ("pass", "pass_att", prediction.pass_attempts_per_team_game),
        ("target", "targets", prediction.targets_per_team_game),
        ("carry", "rush_att", prediction.carries_per_team_game),
    ):
        observed = pd.to_numeric(rows[count], errors="coerce").to_numpy(float)
        observed = observed / team_games
        support = named.copy()
        if name == "pass":
            support &= rows["position"].eq("QB").to_numpy()
        result[name] = _distribution(observed[support], samples[support])

    snap_observed = pd.to_numeric(
        rows.get("snap_counts_observed", pd.Series(0, index=rows.index)),
        errors="coerce",
    ).fillna(0).gt(0).to_numpy()
    support = named & snap_observed
    result["snap"] = _distribution(
        pd.to_numeric(rows.loc[support, "snap_share"], errors="coerce").to_numpy(
            float
        ),
        prediction.snap_share[support],
    )
    quarterback = rows["position"].eq("QB").to_numpy()
    snaps = pd.to_numeric(rows["offense_snaps"], errors="coerce").fillna(0).to_numpy(
        float
    )
    qb_support = named & snap_observed & quarterback & (snaps > 0)
    qb_rate = pd.to_numeric(rows["pass_att"], errors="coerce").fillna(0).to_numpy(
        float
    )
    qb_rate = qb_rate[qb_support] / snaps[qb_support]
    result["qb_propensity"] = _distribution(
        qb_rate, prediction.qb_pass_propensity[qb_support]
    )
    eligible = pd.to_numeric(rows.loc[named, "rush_att"], errors="coerce")
    eligible = eligible.fillna(0).gt(0)
    eligibility_probability = prediction.carry_eligibility_probability[named].mean(
        axis=1
    )
    result["carry_eligibility"] = {
        "n": int(named.sum()),
        "brier": float(
            np.square(eligibility_probability - eligible.to_numpy(float)).mean()
        ),
    }
    target_support = named & rows["position"].isin(("RB", "WR", "TE")).to_numpy()
    observed_target_role = (
        pd.to_numeric(rows.loc[target_support, "targets"], errors="coerce")
        .fillna(0)
        .to_numpy(float)
        / team_games[target_support]
        >= 1.0
    )
    target_role_probability = prediction.target_role_probability[target_support].mean(
        axis=1
    )
    result["target_role"] = {
        "n": int(target_support.sum()),
        "brier": float(
            np.square(
                target_role_probability - observed_target_role.astype(float)
            ).mean()
        ),
    }

    observed_availability = pd.to_numeric(
        rows.loc[named, "observed_availability"], errors="coerce"
    ).to_numpy(float)
    availability_samples = prediction.availability[named]
    availability = _distribution(observed_availability, availability_samples)
    coverage = float(
        interval_coverage(
            observed_availability, availability_samples, level=0.8
        )["coverage"]
    )
    any_probability = (prediction.games_active[named] > 0).mean(axis=1)
    any_observed = observed_availability > 0
    availability.update(
        {
            "coverage_80": coverage,
            "coverage_error": abs(coverage - 0.8),
            "any_brier": float(
                np.square(any_probability - any_observed.astype(float)).mean()
            ),
        }
    )
    result["availability"] = availability
    return result


def _flatten(season: int, model: str, metrics: dict[str, object]) -> dict[str, object]:
    record: dict[str, object] = {"season": season, "model": model}
    for layer, values in metrics.items():
        for metric, value in values.items():
            record[f"{layer}_{metric}"] = value
    return record


def _pooled(records: list[dict[str, object]]) -> list[dict[str, object]]:
    frames = pd.DataFrame(records)
    if frames.empty:
        return []
    output = []
    distribution_layers = (
        "pass",
        "target",
        "carry",
        "snap",
        "qb_propensity",
        "availability",
    )
    for model, group in frames.groupby("model"):
        record: dict[str, object] = {"model": model}
        for layer in distribution_layers:
            n = int(group[f"{layer}_n"].sum())
            record[f"{layer}_n"] = n
            for metric in ("mae", "crps"):
                record[f"{layer}_{metric}"] = float(
                    np.average(
                        group[f"{layer}_{metric}"], weights=group[f"{layer}_n"]
                    )
                )
            record[f"{layer}_rmse"] = float(
                np.sqrt(
                    np.average(
                        np.square(group[f"{layer}_rmse"]),
                        weights=group[f"{layer}_n"],
                    )
                )
            )
        for metric in ("coverage_80", "coverage_error", "any_brier"):
            record[f"availability_{metric}"] = float(
                np.average(
                    group[f"availability_{metric}"],
                    weights=group["availability_n"],
                )
            )
        n = int(group["carry_eligibility_n"].sum())
        record["carry_eligibility_n"] = n
        record["carry_eligibility_brier"] = float(
            np.average(
                group["carry_eligibility_brier"],
                weights=group["carry_eligibility_n"],
            )
        )
        n = int(group["target_role_n"].sum())
        record["target_role_n"] = n
        record["target_role_brier"] = float(
            np.average(
                group["target_role_brier"], weights=group["target_role_n"]
            )
        )
        output.append(record)
    return output


def _jsonable_mapping(values) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def _model_state(model) -> dict[str, object]:
    state: dict[str, object] = {}
    for attribute in ("feature_names", "positions", "players"):
        if hasattr(model, attribute):
            state[attribute] = list(getattr(model, attribute))
    for attribute in (
        "feature_fill",
        "feature_mean",
        "feature_scale",
        "cold_role_prior",
        "availability_prior",
    ):
        if hasattr(model, attribute):
            state[attribute] = _jsonable_mapping(getattr(model, attribute))
    for attribute in ("extra_features", "extra_efficiency_features"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            state[attribute] = None if value is None else list(value)
    for attribute in (
        "role_innovation_scale",
        "per_snap_weight",
        "innovation_cap",
    ):
        if hasattr(model, attribute):
            state[attribute] = float(getattr(model, attribute))
    if hasattr(model, "position_specific_concentration"):
        state["position_specific_concentration"] = bool(
            model.position_specific_concentration
        )
    if hasattr(model, "feature_projection"):
        projection = model.feature_projection
        state["feature_projection"] = (
            None if projection is None else np.asarray(projection).tolist()
        )
    return state


def _restore_model_state(model, state: dict[str, object]) -> None:
    for attribute in ("feature_names", "positions", "players"):
        if attribute in state:
            setattr(model, attribute, list(state[attribute]))
    for attribute in (
        "feature_fill",
        "feature_mean",
        "feature_scale",
        "cold_role_prior",
        "availability_prior",
    ):
        if attribute in state:
            setattr(model, attribute, _jsonable_mapping(state[attribute]))
    for attribute in ("extra_features", "extra_efficiency_features"):
        if attribute in state:
            value = state[attribute]
            setattr(model, attribute, None if value is None else tuple(value))
    for attribute in (
        "role_innovation_scale",
        "per_snap_weight",
        "innovation_cap",
    ):
        if attribute in state:
            setattr(model, attribute, float(state[attribute]))
    if "position_specific_concentration" in state:
        model.position_specific_concentration = bool(
            state["position_specific_concentration"]
        )
    if "feature_projection" in state:
        projection = state["feature_projection"]
        model.feature_projection = (
            None if projection is None else np.asarray(projection, dtype=float)
        )


def _diagnostics(model) -> dict[str, object]:
    if isinstance(model, SeasonSnapShareModel):
        variables = ["intercept", "position_effect", "beta", "concentration"]
    elif isinstance(model, QBPassPropensityModel):
        variables = ["intercept", "beta", "concentration"]
    elif isinstance(model, SeasonCarryEligibilityModel):
        variables = ["intercept", "position_effect", "beta"]
    elif isinstance(model, SeasonTargetRoleModel):
        variables = ["intercept", "position_effect", "beta"]
    elif isinstance(model, SeasonAvailabilityModel):
        variables = [
            "any_intercept",
            "any_position_effect",
            "any_beta",
            "rate_intercept",
            "rate_position_effect",
            "rate_beta",
            "rate_concentration",
        ]
    elif isinstance(model, SeasonRosterShareModel):
        variables = ["beta"]
    else:
        raise TypeError(f"unsupported candidate model: {type(model).__name__}")
    result = sampling_quality(model.idata, variables)
    return {
        key: value
        for key, value in result.items()
        if key != "summary"
    }


def _atomic_save_idata(idata, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".nc", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_idata(idata, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _load_or_fit_candidate(
    label: str,
    train_rows: pd.DataFrame,
    fit_kwargs: dict[str, object],
    holdout_dir: Path,
    run_fingerprint: str,
    *,
    resume: bool,
    score_only: bool,
):
    posterior_path = holdout_dir / f"{label}.nc"
    metadata_path = holdout_dir / f"{label}.metadata.json"
    if resume or score_only:
        if posterior_path.exists() and metadata_path.exists():
            metadata = load_json(metadata_path)
            if metadata.get("run_fingerprint") != run_fingerprint:
                raise ValueError(f"candidate checkpoint does not match run: {label}")
            if metadata.get("posterior_sha256") != file_fingerprint(posterior_path):
                raise ValueError(f"candidate posterior checksum failed: {posterior_path}")
            model = _new_candidate(label)
            _restore_model_state(model, metadata["model_state"])
            model.idata = load_idata(posterior_path)
            print(f"resumed {label} from {posterior_path}", flush=True)
            return model, metadata["diagnostics"]
        if score_only:
            raise FileNotFoundError(f"score-only candidate is incomplete: {label}")

    model = _new_candidate(label)
    print(f"fitting {label}", flush=True)
    model.fit(train_rows, **fit_kwargs)
    diagnostics = _diagnostics(model)
    _atomic_save_idata(model.idata, posterior_path)
    metadata = {
        "candidate": label,
        "configuration": _candidate_configuration(label),
        "run_fingerprint": run_fingerprint,
        "posterior_sha256": file_fingerprint(posterior_path),
        "model_state": _model_state(model),
        "diagnostics": diagnostics,
    }
    atomic_write_json(metadata_path, metadata)
    print(f"checkpointed {label} to {posterior_path}", flush=True)
    return model, diagnostics


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    files = (
        Path(__file__),
        root / "src/ffmodel/features/season_average.py",
        root / "src/ffmodel/features/season_efficiency.py",
        root / "src/ffmodel/features/season_pathways.py",
        root / "src/ffmodel/evaluation/metrics.py",
        root / "src/ffmodel/evaluation/posterior_comparison.py",
        root / "src/ffmodel/evaluation/efficiency_volume_pathways.py",
        root / "src/ffmodel/models/base.py",
        root / "src/ffmodel/models/season_availability.py",
        root / "src/ffmodel/models/season_opportunity.py",
        root / "src/ffmodel/models/volume_season_average.py",
    )
    return combined_fingerprint(
        {path.relative_to(root).as_posix(): file_fingerprint(path) for path in files}
    )


def _collect_records(output_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(output_dir.glob("holdout-*/metrics/*.json")):
        payload = load_json(path)
        records.append(payload["record"])
    return records


def _collect_diagnostics(output_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(output_dir.glob("holdout-*/*.metadata.json")):
        payload = load_json(path)
        season = int(path.parent.name.removeprefix("holdout-"))
        records.append(
            {
                "season": season,
                "component": payload["candidate"],
                "passed": payload["diagnostics"]["passed"],
            }
        )
    return records


def _write_report(args) -> dict[str, object]:
    records = _collect_records(args.output_dir)
    diagnostics = _collect_diagnostics(args.output_dir)
    available_models = {record["model"] for record in records}
    gates = {}
    for candidate, spec in GATE_SPECS.items():
        reference = GATE_REFERENCES.get(candidate, "volume_v2")
        if candidate not in available_models or reference not in available_models:
            continue
        gates[candidate] = evaluate_gate(
            records,
            diagnostics,
            candidate,
            spec,
            expected_holdouts=args.gate_holdouts,
            baseline=reference,
            min_wins=args.min_holdout_wins,
            min_relative_improvement=args.min_relative_improvement,
            max_protected_regression=args.max_protected_regression,
        )
    report = {
        "configuration": {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "nuts_sampler": args.nuts_sampler,
            "gate_holdouts": args.gate_holdouts,
            "min_holdout_wins": args.min_holdout_wins,
            "min_relative_improvement": args.min_relative_improvement,
            "max_protected_regression": args.max_protected_regression,
        },
        "folds": records,
        "pooled": _pooled(records),
        "gates": gates,
    }
    atomic_write_json(args.output_dir / "report.json", report)
    return report


def _score_variant(
    label: str,
    pipeline: SeasonAverageVolumePipeline,
    test: SeasonAverageData,
    metrics_dir: Path,
    run_fingerprint: str,
    *,
    resume: bool,
    seed: int,
) -> dict[str, object]:
    path = metrics_dir / f"{label}.json"
    if resume and path.exists():
        payload = load_json(path)
        if payload.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"metric checkpoint does not match run: {path}")
        print(f"resumed score for {label}", flush=True)
        return payload["record"]
    with _aligned_prediction_posteriors(pipeline):
        prediction = pipeline.predict_samples(test, seed=seed)
    record = _flatten(
        int(test.player_rows["season"].iloc[0]), label, _metrics(prediction)
    )
    atomic_write_json(
        path,
        {"run_fingerprint": run_fingerprint, "record": record},
    )
    print(record, flush=True)
    return record


@contextmanager
def _aligned_prediction_posteriors(pipeline: SeasonAverageVolumePipeline):
    """Temporarily align independently fitted components for prediction."""
    attributes = (
        "team_model",
        "availability_model",
        "workload_model",
        "snap_model",
        "qb_propensity_model",
        "target_role_model",
        "carry_eligibility_model",
        "target_model",
        "carry_model",
    )
    fitted = []
    for attribute in attributes:
        model = getattr(pipeline, attribute)
        idata = getattr(model, "idata", None)
        if idata is not None and hasattr(idata, "posterior"):
            fitted.append(model)
    if not fitted:
        yield
        return
    target = min(posterior_sample_count(model.idata.posterior) for model in fitted)
    originals = {}
    try:
        for model in fitted:
            posterior = model.idata.posterior
            if posterior_sample_count(posterior) != target:
                originals[id(model)] = (model, model.idata)
                model.idata = SimpleNamespace(
                    posterior=select_posterior_samples(posterior, target)
                )
        yield
    finally:
        for model, idata in originals.values():
            model.idata = idata


def _manifest_payload(
    args,
    holdout: int,
    train: SeasonAverageData,
    test: SeasonAverageData,
    baseline_dir: Path,
) -> dict[str, object]:
    data_fingerprints = {
        "train_team": frame_fingerprint(
            train.team_rows, keys=("season", "team")
        ),
        "train_player": frame_fingerprint(
            train.player_rows, keys=("season", "team", "player_key")
        ),
        "test_team": frame_fingerprint(test.team_rows, keys=("season", "team")),
        "test_player": frame_fingerprint(
            test.player_rows, keys=("season", "team", "player_key")
        ),
    }
    return {
        "holdout": int(holdout),
        "seasons": list(args.seasons),
        "fit": {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "nuts_sampler": args.nuts_sampler,
            "seed": args.seed,
        },
        "candidate_configurations": {
            label: _candidate_configuration(label) for label in CANDIDATE_ORDER
        },
        "data_fingerprints": data_fingerprints,
        "baseline_fingerprint": directory_fingerprint(baseline_dir),
        "source_fingerprint": _source_fingerprint(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2014, 2025)))
    parser.add_argument("--holdouts", nargs="+", type=int, default=[2022])
    parser.add_argument(
        "--gate-holdouts", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nuts-sampler", choices=("pymc", "nutpie"), default="nutpie")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--candidates", nargs="+", choices=CANDIDATE_ORDER, default=list(CANDIDATE_ORDER)
    )
    parser.add_argument("--min-holdout-wins", type=int, default=2)
    parser.add_argument("--min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--max-protected-regression", type=float, default=0.005)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(".cache/season-average-validation/volume-v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            ".cache/season-average-validation/efficiency-pathway-posteriors-v2"
        ),
    )
    args = parser.parse_args(argv)
    if args.score_only:
        args.resume = True
    if args.nuts_sampler == "nutpie" and "NUMBA_CACHE_DIR" not in os.environ:
        os.environ["NUMBA_CACHE_DIR"] = str(Path(tempfile.gettempdir()) / "ffmodel-numba")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_manifest = {
        "seasons": list(args.seasons),
        "fit": {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "nuts_sampler": args.nuts_sampler,
            "seed": args.seed,
        },
        "baseline_dir": str(args.baseline_dir.resolve()),
        "candidate_configurations": {
            label: _candidate_configuration(label) for label in CANDIDATE_ORDER
        },
        "source_fingerprint": _source_fingerprint(),
    }
    ensure_manifest(
        args.output_dir / "experiment.json",
        experiment_manifest,
        resume=args.resume,
    )

    data = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    )
    fit_kwargs = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "nuts_sampler": args.nuts_sampler,
        "seed": args.seed,
    }
    for holdout in args.holdouts:
        train = SeasonAverageData(
            data.team_rows[data.team_rows["season"] < holdout].copy(),
            data.player_rows[data.player_rows["season"] < holdout].copy(),
        )
        test = SeasonAverageData(
            data.team_rows[data.team_rows["season"] == holdout].copy(),
            data.player_rows[data.player_rows["season"] == holdout].copy(),
        )
        if test.player_rows.empty:
            raise ValueError(f"holdout has no player rows: {holdout}")
        baseline_dir = args.baseline_dir / f"holdout-{holdout}-final"
        holdout_dir = args.output_dir / f"holdout-{holdout}"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        manifest = _manifest_payload(args, holdout, train, test, baseline_dir)
        ensure_manifest(
            holdout_dir / "manifest.json", manifest, resume=args.resume
        )
        run_fingerprint = combined_fingerprint(manifest)

        pipeline = SeasonAverageVolumePipeline.load(baseline_dir)
        original = {
            "availability": pipeline.availability_model,
            "snap": pipeline.snap_model,
            "qb": pipeline.qb_propensity_model,
            "target": pipeline.target_model,
            "target_role": pipeline.target_role_model,
            "carry": pipeline.carry_model,
            "eligibility": pipeline.carry_eligibility_model,
        }
        candidates = {}
        diagnostics = {}
        for label in CANDIDATE_ORDER:
            posterior_path = holdout_dir / f"{label}.nc"
            metadata_path = holdout_dir / f"{label}.metadata.json"
            should_load = args.resume and posterior_path.exists() and metadata_path.exists()
            if label not in args.candidates and not should_load:
                continue
            candidates[label], diagnostics[label] = _load_or_fit_candidate(
                label,
                train.player_rows,
                fit_kwargs,
                holdout_dir,
                run_fingerprint,
                resume=args.resume,
                score_only=args.score_only,
            )
            _write_report(args)

        metrics_dir = holdout_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for label, replacements in VARIANTS.items():
            dependencies = set(replacements.values())
            if not dependencies <= set(candidates):
                continue
            for component, model in original.items():
                if component == "availability":
                    pipeline.availability_model = model
                elif component == "snap":
                    pipeline.snap_model = model
                elif component == "qb":
                    pipeline.qb_propensity_model = model
                elif component == "target":
                    pipeline.target_model = model
                elif component == "target_role":
                    pipeline.target_role_model = model
                elif component == "carry":
                    pipeline.carry_model = model
                elif component == "eligibility":
                    pipeline.carry_eligibility_model = model
            for component, candidate_label in replacements.items():
                model = candidates[candidate_label]
                if component == "availability":
                    pipeline.availability_model = model
                elif component == "snap":
                    pipeline.snap_model = model
                elif component == "qb":
                    pipeline.qb_propensity_model = model
                elif component == "target":
                    pipeline.target_model = model
                elif component == "target_role":
                    pipeline.target_role_model = model
                elif component == "carry":
                    pipeline.carry_model = model
                elif component == "eligibility":
                    pipeline.carry_eligibility_model = model
            _score_variant(
                label,
                pipeline,
                test,
                metrics_dir,
                run_fingerprint,
                resume=args.resume,
                seed=args.seed,
            )
            _write_report(args)

    report = _write_report(args)
    print(report["gates"], flush=True)


if __name__ == "__main__":
    main()
