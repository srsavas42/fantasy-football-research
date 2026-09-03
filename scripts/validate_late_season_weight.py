"""Walk-forward the late-season role weight, optionally varying it by experience.

``prior_target_role`` and its siblings ship as

    0.65 * prior_full_season_share + 0.35 * prior_late_season_share

with "late" meaning weeks >= 10. The 0.35 is a single constant applied to every
player and nothing in this repo records it being swept.

scripts/screen_late_season_weight.py solves for the weight minimising squared
error against next season's share, and gets 0.035 to 0.095 pooled depending on
specification -- far below 0.35 in every arm. Broken out by experience the
optimum falls monotonically with age (roughly 0.14-0.18 entering year two,
0.12-0.18 in years three to four, and about zero from year five on), which is
the "young players break out late" hypothesis, though every bucket interval is
wide enough to overlap its neighbours.

Neither result transfers automatically. That screen is a linear fit predicting
raw share from two predictors; the pipeline feeds ``prior_target_role`` into
``_role_prior`` at a weight of 0.25 for targets -- the geometric blend against
``prior_target_per_snap`` carries the other 0.75 -- and then through a softmax
against projected exposure. This runs the actual model.

Both role columns the blend needs are already on the cached frames and the
identity ``role == 0.65*full + 0.35*late`` holds there exactly, so an arm is a
recomputation rather than a rebuild: every arm reads byte-identical frames apart
from the three columns under test.

    python scripts/validate_late_season_weight.py late10 --late-weight 0.10
    python scripts/validate_late_season_weight.py lateage \\
        --late-weight-young 0.20 --late-weight-old 0.02

The shipped 0.35 baseline is scripts/validation_runs/wf_roombase.json, which is
the default configuration on these same frames.
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
from _walkforward_data import frames_fingerprint, load_frames

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

ROLE_BLENDS = (
    ("prior_target_role", "prior_target_share", "prior_late_target_share"),
    ("prior_carry_role", "prior_carry_share", "prior_late_carry_share"),
    ("prior_pass_role", "prior_pass_attempt_share", "prior_late_pass_attempt_share"),
)
SHIPPED_LATE_WEIGHT = 0.35


def reweight(rows: pd.DataFrame, weight: np.ndarray) -> pd.DataFrame:
    """Rebuild the three role columns at a per-row late weight."""
    out = rows.copy()
    for role, full_col, late_col in ROLE_BLENDS:
        full = pd.to_numeric(out[full_col], errors="coerce")
        late = pd.to_numeric(out[late_col], errors="coerce")
        # Missing stays missing: ``_role_prior`` falls through to the draft
        # prior and then the position fallback for those rows, and filling
        # here would silently take that path away.
        out[role] = (1.0 - weight) * full + weight * late
    return out


def row_weight(rows: pd.DataFrame, args) -> np.ndarray:
    if args.late_weight is not None:
        return np.full(len(rows), float(args.late_weight))
    experience = pd.to_numeric(
        rows.get("experience", pd.Series(np.nan, index=rows.index)), errors="coerce"
    ).to_numpy(float)
    # Unknown experience takes the veteran weight rather than the young one:
    # the young arm is the deviation from shipped behaviour and should apply
    # only where the frame actually says the player is young.
    young = np.isfinite(experience) & (experience <= args.young_max_experience)
    return np.where(young, float(args.late_weight_young), float(args.late_weight_old))


def distribution(observed, samples) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    mean = samples.mean(axis=1)
    return {
        "n": int(len(observed)),
        "mae": float(np.abs(observed - mean).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "cov80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "cov95": float(interval_coverage(observed, samples, level=0.95)["coverage"]),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--late-weight", type=float, default=None)
    parser.add_argument("--late-weight-young", type=float, default=None)
    parser.add_argument("--late-weight-old", type=float, default=None)
    parser.add_argument("--young-max-experience", type=float, default=3.0)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-walkforward"))
    parser.add_argument("--holdouts", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/validation_runs"))
    args = parser.parse_args(argv)

    constant = args.late_weight is not None
    varying = args.late_weight_young is not None and args.late_weight_old is not None
    if constant == varying:
        raise SystemExit(
            "pass exactly one of --late-weight or "
            "(--late-weight-young and --late-weight-old)"
        )

    player_rows, team_rows = load_frames(args.cache_dir)
    weight = row_weight(player_rows, args)
    player_rows = reweight(player_rows, weight)
    if constant:
        described = f"constant {args.late_weight}"
    else:
        described = (
            f"{args.late_weight_young} for experience <= "
            f"{args.young_max_experience}, {args.late_weight_old} otherwise"
        )
    print(f"late weight: {described}  (shipped {SHIPPED_LATE_WEIGHT})")

    report: dict[str, object] = {
        "_frames": frames_fingerprint(player_rows, team_rows, args.cache_dir),
        "_late_weight": {
            "constant": args.late_weight,
            "young": args.late_weight_young,
            "old": args.late_weight_old,
            "young_max_experience": args.young_max_experience,
            "rows_at_young_weight": int((weight != weight.max()).sum())
            if not constant else 0,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"wf_{args.label}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    sample_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    for holdout in args.holdouts:
        started = time.perf_counter()
        train = SeasonAverageData(
            team_rows[team_rows.season < holdout].copy(),
            player_rows[player_rows.season < holdout].copy(),
        )
        test = SeasonAverageData(
            team_rows[team_rows.season == holdout].copy(),
            player_rows[player_rows.season == holdout].copy(),
        )
        pipeline = SeasonAverageVolumePipeline()
        pipeline.fit(train, **sample_kwargs)
        prediction = pipeline.predict_samples(test, seed=42)

        rows = prediction.player_rows
        named = (
            pd.to_numeric(
                rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
                errors="coerce",
            ).fillna(0).ne(1).to_numpy()
        )
        quarterback = rows["position"].eq("QB").to_numpy()
        games = pd.to_numeric(rows["team_games"], errors="coerce").to_numpy(float)
        experience = pd.to_numeric(
            rows.get("experience", pd.Series(np.nan, index=rows.index)), errors="coerce"
        ).to_numpy(float)
        young = np.isfinite(experience) & (experience <= args.young_max_experience)

        def per_game(column: str, mask: np.ndarray) -> np.ndarray:
            values = pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
            return values[mask] / games[mask]

        fold: dict[str, object] = {}
        for stream, column, samples in (
            ("target", "targets", prediction.targets_per_team_game),
            ("carry", "rush_att", prediction.carries_per_team_game),
        ):
            fold[stream] = distribution(per_game(column, named), samples[named])
            # The whole point of the varying arm is that it should move young
            # players and leave everyone else alone, so score them separately.
            for label, at in (("young", named & young), ("old", named & ~young)):
                if at.sum() >= 20:
                    fold[f"{stream}_{label}"] = distribution(
                        per_game(column, at), samples[at]
                    )
        fold["pass_qb"] = distribution(
            per_game("pass_att", quarterback & named),
            prediction.pass_attempts_per_team_game[quarterback & named],
        )
        snaps_seen = (
            pd.to_numeric(rows["snap_counts_observed"], errors="coerce")
            .fillna(0).gt(0).to_numpy()
        )
        fold["snap"] = distribution(
            pd.to_numeric(
                rows.loc[snaps_seen & named, "snap_share"], errors="coerce"
            ).to_numpy(float),
            prediction.snap_share[snaps_seen & named],
        )
        fold["fit_seconds"] = round(time.perf_counter() - started, 1)
        report[str(holdout)] = fold
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"holdout {holdout} done in {fold['fit_seconds']}s -> {path}")


if __name__ == "__main__":
    main()
