"""Choose ``cold_role_multiplier_cap`` on an inner fold.

The cold-role widening was promoted on strong evidence, but the number that
decides how wide it goes was not part of it. The cap binds in both modes on real
data -- measured mode asks for a ratio over ten and the cap is six -- so the cap
rather than the measurement sets where cold rows land, and it was chosen before
any result and never selected against folds.

Two things make that worth fixing rather than leaving. Post-promotion coverage
sits slightly conservative at both levels, z between -0.6 and -1.1 across scoring
systems, which is the direction a slightly-too-large cap would push it. And the
last cap this package selected -- ``innovation_cap`` -- was chosen on a criterion
that turned out to be uninterpretable for the streams it scored, so "a cap was
selected" is not on its own reassuring.

## The criterion, fixed before the numbers

Mean CRPS over the three scoring formats on **total fantasy points**, which is
what the package publishes. CRPS is a proper scoring rule: it is minimised by
the true predictive distribution and penalises over- and under-dispersion alike,
so it can select a width without needing a coverage statistic to behave. That
matters here because the criterion that picked ``innovation_cap`` at 0.25 was
mean distance from nominal coverage on carry and target counts, and half of
carry rows are zero, so every interval containing zero covered them and the
population rate could not reach nominal however good the model was. A criterion
that reads guaranteed coverage as over-wide intervals rewards narrowing.

Coverage is reported alongside as a diagnostic, and does not select.

## Why it is cheap

``cold_role_multiplier`` is consumed at prediction time and enters no
likelihood, and within a fold the only thing a candidate changes is where the
ratio is clipped. So the uncapped ratio is computed once per fold and each
candidate is a clip of it -- one pipeline fit per fold rather than one per
candidate.

## Nesting

Searching candidates on the seasons they are scored against is selecting on the
test set:

    for each outer holdout H:
        fit on seasons < H-1, sweep caps against H-1, pick
        fit on seasons < H,   score H with the winner and with the incumbent

2025 is not among the holdouts and must not be: it is the one season no choice
in this package has seen.

    python scripts/select_cold_role_cap.py
    python scripts/select_cold_role_cap.py --drop-seasons 2016
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _walkforward_data import DEFAULT_CACHE, HOLDOUTS, frames_fingerprint, load_frames

from ffmodel.evaluation.efficiency_posterior import score_fantasy_points_posterior
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.base import calibrate_innovation_scale
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline

SCORING_FORMATS = ("standard", "half_ppr", "ppr")
INCUMBENT = 6.0
ALLOCATORS = ("target", "carry")


def uncapped_ratios(volume_model, train_rows: pd.DataFrame) -> dict[str, float]:
    """The cold-to-base scale ratio each allocator would use with no cap.

    Computed once per fold. Every candidate is then a clip of this, which is
    exact rather than an approximation: the cap enters ``_fit_cold_role_multiplier``
    only as the upper bound of a clip.
    """
    ratios: dict[str, float] = {}
    for name in ALLOCATORS:
        model = getattr(volume_model, f"{name}_model")
        prepared = model._prepare(train_rows)
        cold_rms, _ = model._cold_and_warm_dispersion(prepared)
        if not np.isfinite(cold_rms) or model.role_innovation_scale <= 1e-8:
            ratios[name] = 1.0
            continue
        if model.calibrated_innovation:
            allocation, mask = model._innovation_rooms(prepared)
            cold_scale = calibrate_innovation_scale(
                allocation, mask, cold_rms, seed=model.innovation_calibration_seed
            )
        else:
            cold_scale = cold_rms
        ratios[name] = float(cold_scale / model.role_innovation_scale)
    return ratios


def apply_cap(volume_model, ratios: dict[str, float], cap: float | None) -> dict[str, float]:
    """Set each allocator's multiplier as though it had been fitted under ``cap``."""
    applied: dict[str, float] = {}
    for name in ALLOCATORS:
        model = getattr(volume_model, f"{name}_model")
        bound = float("inf") if cap is None else float(cap)
        model.cold_role_multiplier_cap = bound
        model.cold_role_multiplier = float(np.clip(ratios[name], 1.0, bound))
        applied[name] = model.cold_role_multiplier
    return applied


def score(pipeline, test: SeasonAverageData) -> dict[str, dict[str, float]]:
    prediction = pipeline.predict_samples(test, seed=42)
    return {
        scoring: score_fantasy_points_posterior(prediction, scoring=scoring)
        for scoring in SCORING_FORMATS
    }


def criterion(scores: dict[str, dict[str, float]]) -> float:
    """Mean CRPS across scoring formats. Lower is better. Fixed in advance."""
    return float(np.mean([scores[s]["crps"] for s in SCORING_FORMATS]))


def coverage_gap(scores: dict[str, dict[str, float]]) -> float:
    """Reported, never selected on. See the module docstring."""
    gaps = []
    for scoring in SCORING_FORMATS:
        gaps.append(abs(scores[scoring]["coverage_80"] - 0.80))
        gaps.append(abs(scores[scoring]["coverage_95"] - 0.95))
    return float(np.mean(gaps))


def fit_pipeline(player_rows, team_rows, seasons, sample_kwargs):
    data = SeasonAverageData(
        team_rows[team_rows.season.isin(seasons)].copy(),
        player_rows[player_rows.season.isin(seasons)].copy(),
    )
    pipeline = SeasonAverageScoringPipeline()
    pipeline.fit(
        data,
        volume_sample_kwargs=sample_kwargs,
        efficiency_sample_kwargs=sample_kwargs,
    )
    return pipeline, data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", default="cold_role_cap")
    parser.add_argument(
        "--caps",
        nargs="+",
        default=["1.0", "2.0", "3.0", "4.0", "6.0", "8.0", "None"],
        help="candidate caps; 1.0 is the widening off and None is uncapped, "
             "both kept so the sweep can report that the bound does nothing "
             "useful in either direction",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--holdouts", nargs="+", type=int, default=list(HOLDOUTS))
    parser.add_argument(
        "--drop-seasons",
        nargs="*",
        type=int,
        default=(),
        help="exclude these seasons from training. 2016 carries about 280 rows "
             "no other season has, almost all of them cold -- see "
             "docs/data-quality-2026-08.md. Excluding it is a deliberate "
             "choice to be measured, not a default",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("scripts/validation_runs")
    )
    args = parser.parse_args(argv)

    player_rows, team_rows = load_frames(args.cache_dir)
    if args.drop_seasons:
        dropped = set(args.drop_seasons)
        player_rows = player_rows[~player_rows.season.isin(dropped)]
        team_rows = team_rows[~team_rows.season.isin(dropped)]
        print(f"dropped seasons {sorted(dropped)} from training and scoring")
    sample_kwargs = {"draws": args.draws, "tune": args.draws, "chains": args.chains}
    candidates: list[float | None] = [
        None if c.lower() == "none" else float(c) for c in args.caps
    ]

    report: dict[str, object] = {
        "_frames": frames_fingerprint(player_rows, team_rows, args.cache_dir),
        "criterion": "mean CRPS over standard/half_ppr/ppr on total fantasy points",
        "candidates": [str(c) for c in candidates],
        "incumbent": INCUMBENT,
        "dropped_seasons": sorted(args.drop_seasons),
    }

    for holdout in args.holdouts:
        started = time.perf_counter()
        inner_season = holdout - 1
        seasons = sorted(player_rows.season.unique())

        inner_pipeline, inner_train = fit_pipeline(
            player_rows, team_rows, [s for s in seasons if s < inner_season], sample_kwargs
        )
        inner_test = SeasonAverageData(
            team_rows[team_rows.season == inner_season].copy(),
            player_rows[player_rows.season == inner_season].copy(),
        )
        inner_ratios = uncapped_ratios(inner_pipeline.volume_model, inner_train.player_rows)
        print(
            f"[{holdout}] inner uncapped ratios: "
            + ", ".join(f"{k}={v:.3f}" for k, v in inner_ratios.items()),
            flush=True,
        )

        inner: dict[str, dict] = {}
        for cap in candidates:
            applied = apply_cap(inner_pipeline.volume_model, inner_ratios, cap)
            scores = score(inner_pipeline, inner_test)
            inner[str(cap)] = {
                "multipliers": applied,
                "scores": scores,
                "crps": criterion(scores),
                "coverage_gap": coverage_gap(scores),
            }
            print(
                f"[{holdout}] inner cap={cap}: crps={inner[str(cap)]['crps']:.4f} "
                f"cov_gap={inner[str(cap)]['coverage_gap']:.4f} "
                f"ppr_cov95={scores['ppr']['coverage_95']:.3f}",
                flush=True,
            )

        chosen_name = min(inner, key=lambda k: inner[k]["crps"])
        chosen = None if chosen_name == "None" else float(chosen_name)
        print(f"[{holdout}] inner fold picks cap={chosen_name}", flush=True)

        outer_pipeline, outer_train = fit_pipeline(
            player_rows, team_rows, [s for s in seasons if s < holdout], sample_kwargs
        )
        outer_test = SeasonAverageData(
            team_rows[team_rows.season == holdout].copy(),
            player_rows[player_rows.season == holdout].copy(),
        )
        outer_ratios = uncapped_ratios(outer_pipeline.volume_model, outer_train.player_rows)

        incumbent_multipliers = apply_cap(
            outer_pipeline.volume_model, outer_ratios, INCUMBENT
        )
        incumbent_scores = score(outer_pipeline, outer_test)
        selected_multipliers = apply_cap(
            outer_pipeline.volume_model, outer_ratios, chosen
        )
        selected_scores = score(outer_pipeline, outer_test)

        report[str(holdout)] = {
            "inner_season": inner_season,
            "inner": inner,
            "inner_ratios": inner_ratios,
            "chosen_cap": chosen_name,
            "outer_ratios": outer_ratios,
            "outer_incumbent": incumbent_scores,
            "outer_incumbent_multipliers": incumbent_multipliers,
            "outer_selected": selected_scores,
            "outer_selected_multipliers": selected_multipliers,
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(
            f"[{holdout}] incumbent crps={criterion(incumbent_scores):.4f} "
            f"selected crps={criterion(selected_scores):.4f} "
            f"({report[str(holdout)]['seconds']:.0f}s)",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
