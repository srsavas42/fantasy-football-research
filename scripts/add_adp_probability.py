"""Add a probability-of-beating-ADP column to an existing projection CSV.

Deliberately does not re-fit the season-average pipeline. Everything this needs
is already on disk: ``model_only`` in the projection CSV is the pipeline's own
unblended mean, exactly the quantity ``AdpEdgeModel`` was validated against, and
the ADP curve is cheap to refit (a per-position log-rank regression with a
residual pool, no MCMC) on the same history ``project_season.py`` used.

The probability model itself is fit on every scored holdout pooled -- 2023,
2024 and 2025 together -- rather than leave-one-season-out. Leave-one-out was
the validation question (does this generalize to a season it never saw); this
is the deployment question (best use of all history for a season, 2026, with no
outcome yet to hold out). The same distinction ``project_season.py`` makes for
the blend weight, which uses the most recent fold's weight rather than
refitting on 2026.

Left blank rather than filled with the base rate for any player whose
``adp_drafted`` is 0. Those rows carry a fallback ``adp_rank`` of 301 so the
availability regression has a number to read, not a real board position -- there
is no board expectation for them to beat, so a stated probability would be an
opinion about a comparison that does not exist.

    python scripts/add_adp_probability.py --projection projections/2026_ppr.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from project_season import observed_points  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffmodel.models.adp_edge import AdpEdgeModel  # noqa: E402
from ffmodel.models.market_blend import RankCurve  # noqa: E402


def fit_edge_model(report_paths: list[Path]) -> AdpEdgeModel:
    """Pooled fit across every scored holdout -- the deployment version."""
    gaps, beats = [], []
    for path in report_paths:
        rows = json.loads(path.read_text("utf-8"))["folds"][path.stem.split("_")[-1]]["rows"]
        model = np.array(rows["model_mean"], dtype=float)
        adp = np.array(rows["adp_mean"], dtype=float)
        observed = np.array(rows["observed"], dtype=float)
        gaps.append(model - adp)
        beats.append((observed > adp).astype(float))
    return AdpEdgeModel().fit(np.concatenate(gaps), np.concatenate(beats))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, default=Path("projections/2026_ppr.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument(
        "--reports", type=Path, nargs="+",
        default=[Path("reports/mva_2023.json"), Path("reports/mva_2024.json"),
                 Path("reports/mva_2025.json")],
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    out = pd.read_csv(args.projection)
    for column in ("model_only", "adp_rank", "adp_drafted", "position"):
        if column not in out.columns:
            raise SystemExit(f"{args.projection} is missing column {column!r}")

    edge_model = fit_edge_model(args.reports)
    print(f"AdpEdgeModel: slope={edge_model.slope:+.3f} "
          f"base_rate={edge_model.base_rate:.1%} "
          f"(fit on {sum(1 for _ in args.reports)} pooled seasons)")

    # The ADP curve, refit on exactly the history project_season.py used --
    # every observed season before the projection season, drafted players only.
    player_rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    history = player_rows[player_rows.season < args.season].reset_index(drop=True)
    history = history[
        pd.to_numeric(history.get("is_replacement_player"), errors="coerce")
        .fillna(0).ne(1)
    ].reset_index(drop=True)
    points = np.full(len(history), np.nan)
    for _, block in history.groupby("season"):
        points[block.index] = observed_points(block.reset_index(drop=True), args.scoring)
    usable = np.isfinite(points)
    curve = RankCurve().fit(history[usable].reset_index(drop=True), points[usable])

    # RankCurve reads adp_rank / adp_drafted / position off the frame it is
    # handed; the projection CSV already carries exactly those three.
    adp_samples = curve.predict_samples(out, draws=args.draws, seed=args.seed)
    adp_mean = np.where(
        np.isfinite(adp_samples).all(axis=1), adp_samples.mean(axis=1), np.nan
    )
    drafted = pd.to_numeric(out["adp_drafted"], errors="coerce").fillna(0).eq(1).to_numpy()
    adp_mean = np.where(drafted, adp_mean, np.nan)

    gap = out["model_only"].to_numpy(dtype=float) - adp_mean
    probability = edge_model.predict(gap)
    probability = np.where(drafted & np.isfinite(gap), probability, np.nan)

    out["adp_projection"] = adp_mean
    out["prob_beat_adp"] = probability
    covered = int(np.isfinite(probability).sum())
    print(f"probability computed for {covered} of {len(out)} rows "
          f"({int(drafted.sum())} carry a real adp_drafted flag)")

    out.to_csv(args.projection, index=False)
    print(f"rewrote {args.projection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
