"""Walk-forward calibration screen for player-season role regimes.

This validates only the shared-state *predictor*.  It does not alter volume
or scoring, so a failure here cannot affect an accepted production model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ffmodel.features.season_average import build_season_average_data
from ffmodel.models.season_regime import (
    REGIME_NAMES,
    SeasonRegimeModel,
    realized_regimes,
)


def _metrics(probability: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    target = np.eye(len(REGIME_NAMES))[observed]
    return {
        "accuracy": float((probability.argmax(axis=1) == observed).mean()),
        "log_loss": float(
            -np.log(np.clip(probability[np.arange(len(observed)), observed], 1e-12, 1.0)).mean()
        ),
        "brier": float(((probability - target) ** 2).sum(axis=1).mean()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2014, 2025)))
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)

    rows = build_season_average_data(
        args.seasons, source="nflverse", roster_mode="point_in_time"
    ).player_rows
    report: dict[str, object] = {"seasons": args.seasons, "holdouts": args.holdouts, "folds": {}}
    for holdout in args.holdouts:
        train = rows.loc[rows["season"].lt(holdout)].reset_index(drop=True)
        test = rows.loc[rows["season"].eq(holdout)].reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(f"holdout {holdout} needs both training and test rows")
        model = SeasonRegimeModel().fit(train)
        probability = model.predict_proba(test)
        labels = realized_regimes(test, model.thresholds)
        observed = np.array([REGIME_NAMES.index(label) for label in labels])
        baseline_probability = np.zeros_like(probability)
        train_labels = realized_regimes(train, model.thresholds)
        train_frequency = np.array([(train_labels == name).mean() for name in REGIME_NAMES])
        baseline_probability[:] = train_frequency
        fold = {
            "n_players": int(len(test)),
            "thresholds": model.thresholds.lead_role_threshold,
            "observed_frequency": {name: float((labels == name).mean()) for name in REGIME_NAMES},
            "mean_predicted_probability": {
                name: float(probability[:, index].mean()) for index, name in enumerate(REGIME_NAMES)
            },
            "baseline": _metrics(baseline_probability, observed),
            "regime_model": _metrics(probability, observed),
        }
        report["folds"][str(holdout)] = fold
        print(
            f"{holdout}: log loss baseline={fold['baseline']['log_loss']:.3f} "
            f"regime={fold['regime_model']['log_loss']:.3f}; "
            f"Brier baseline={fold['baseline']['brier']:.3f} "
            f"regime={fold['regime_model']['brier']:.3f}; "
            f"accuracy={fold['regime_model']['accuracy']:.3f}"
        )
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
