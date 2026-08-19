"""How does the model do on the players people actually draft?

Pooled metrics run over every rostered player, which is the right population for
asking whether the model is calibrated but the wrong one for asking whether it is
useful. Most of those rows are fringe players whose season is a handful of
points; a projection is bought and sold on the couple of hundred players who go
in a draft.

This restricts the 2025 out-of-sample score to the FantasyPros preseason ADP
top-N and reports by draft tier, because a miss on a second-round pick costs
more than a miss on the last pick of the draft.

Two things to keep in mind reading the output.

**Absolute error rises with ADP rank.** Early picks score more points, so they
have more points to be wrong about. Relative error is the comparable number
across tiers, and it is reported alongside.

**ADP is itself a forecast**, made by the market before the season. Comparing
the model against realized points on the ADP pool is a fair test of the model;
comparing the model *to* ADP as a competing projection is a different exercise
and this does not do it.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points

MODEL_POSITIONS = ("QB", "RB", "WR", "TE")
TIERS = ((1, 50), (51, 100), (101, 200), (201, 300))


def normalise(name: str) -> str:
    """Match key: lowercase letters only, generational suffixes dropped.

    Deliberately not fuzzy. A fuzzy join here would silently pair the wrong
    player and the error would land in a metric rather than in an exception.
    """
    text = str(name).lower()
    text = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", text)
    return re.sub(r"[^a-z]", "", text)


def load_adp(season: int, directory: Path) -> pd.DataFrame:
    path = directory / f"FantasyPros_{season}_Overall_ADP_Rankings.csv"
    adp = pd.read_csv(path)
    parsed = adp["Player (Bye)"].astype(str).str.strip().str.extract(
        r"^(?P<name>.*?)\s+(?P<team>[A-Z]{2,3})\s*\(\w+\)$"
    )
    adp["adp_name"] = parsed["name"].fillna(adp["Player (Bye)"].astype(str).str.strip())
    adp["adp_team"] = parsed["team"]
    adp["adp_position"] = adp["POS"].astype(str).str.extract(r"^([A-Z]+)")[0]
    adp["adp_rank"] = pd.to_numeric(adp["Rank"], errors="coerce")
    adp["key"] = adp["adp_name"].map(normalise)
    return adp


def summarise(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "n": int(len(observed)),
        "observed_mean": float(observed.mean()),
        "mae": float(np.abs(observed - mean).mean()),
        "mae_pct": float(np.abs(observed - mean).mean() / max(observed.mean(), 1e-9)),
        "bias": float((mean - observed).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "cov80": float(interval_coverage(observed, samples, 0.80)["coverage"]),
        "cov95": float(interval_coverage(observed, samples, 0.95)["coverage"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--adp-dir", type=Path, default=Path("ADP"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025"))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path(
        f"scripts/validation_runs/adp_top{args.top}_{args.season}.json"
    )

    adp = load_adp(args.season, args.adp_dir)
    drafted = adp[
        adp.adp_rank.le(args.top) & adp.adp_position.isin(MODEL_POSITIONS)
    ].copy()

    pr = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    tr = pd.read_pickle(args.cache_dir / "team_rows.pkl")
    train = SeasonAverageData(
        tr[tr.season < args.season].copy(), pr[pr.season < args.season].copy()
    )
    test = SeasonAverageData(
        tr[tr.season == args.season].copy(), pr[pr.season == args.season].copy()
    )

    pipeline = SeasonAverageScoringPipeline()
    sample_kwargs = {"draws": args.draws, "tune": args.draws, "chains": 4}
    pipeline.fit(
        train,
        volume_sample_kwargs=sample_kwargs,
        efficiency_sample_kwargs=sample_kwargs,
    )
    print("fitted", flush=True)
    prediction = pipeline.predict_samples(test, seed=42)

    rows = prediction.player_rows.reset_index(drop=True)
    observed = fantasy_points(observed_scoring_rows(rows), args.scoring).to_numpy(float)
    samples = np.asarray(prediction.fantasy_points[args.scoring], dtype=float)
    named = (
        pd.to_numeric(
            rows.get("is_replacement_player", pd.Series(0, index=rows.index)),
            errors="coerce",
        )
        .fillna(0)
        .ne(1)
        .to_numpy()
    )
    finite = np.isfinite(observed) & np.isfinite(samples).all(axis=1)
    rows["key"] = rows["player_name"].map(normalise)

    # One model row per key. Duplicates would otherwise let a single ADP entry
    # pull in two rows and double-count the player.
    ranked = dict(zip(drafted["key"], drafted["adp_rank"]))
    rank_of = rows["key"].map(ranked)
    matched = rank_of.notna().to_numpy() & named & finite

    duplicated = rows.loc[matched, "key"].duplicated().sum()
    unmatched = sorted(set(drafted["key"]) - set(rows.loc[named & finite, "key"]))
    report: dict[str, object] = {
        "season": args.season,
        "scoring": args.scoring,
        "top": args.top,
        "drafted_at_model_positions": int(len(drafted)),
        "matched": int(matched.sum()),
        "duplicate_model_rows": int(duplicated),
        "unmatched_names": [
            drafted.loc[drafted.key.eq(k), "adp_name"].iloc[0] for k in unmatched
        ],
    }
    print(
        f"\nADP top {args.top}: {len(drafted)} at model positions, "
        f"{int(matched.sum())} matched, {len(unmatched)} unmatched"
    )
    if unmatched:
        print(f"  unmatched: {', '.join(report['unmatched_names'])}")

    everyone = named & finite
    groups: dict[str, dict[str, float]] = {
        f"ADP top {args.top}": summarise(observed[matched], samples[matched]),
        "all rostered": summarise(observed[everyone], samples[everyone]),
    }
    for low, high in TIERS:
        band = matched & rank_of.between(low, high).to_numpy()
        if band.sum() >= 10:
            groups[f"ADP {low}-{high}"] = summarise(observed[band], samples[band])
    for position in MODEL_POSITIONS:
        band = matched & rows["position"].eq(position).to_numpy()
        if band.sum() >= 10:
            groups[f"top {args.top}, {position}"] = summarise(
                observed[band], samples[band]
            )
    report["groups"] = groups

    print(f"\n{args.season} {args.scoring.upper()} SEASON TOTALS\n")
    header = (
        f"  {'group':22s} {'n':>4s} {'obs mean':>9s} {'MAE':>8s} {'MAE %':>7s} "
        f"{'bias':>8s} {'CRPS':>8s} {'cov80':>7s} {'cov95':>7s}"
    )
    print(header)
    for label, values in groups.items():
        print(
            f"  {label:22s} {values['n']:>4d} {values['observed_mean']:>9.1f} "
            f"{values['mae']:>8.2f} {values['mae_pct']:>6.1%} {values['bias']:>+8.2f} "
            f"{values['crps']:>8.2f} {values['cov80']:>7.3f} {values['cov95']:>7.3f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")

    # Per-row predictions, so a baseline scored on a subset of these rows can be
    # compared against the model on that same subset rather than against the
    # pooled number. Without this the only honest comparison is the one that
    # happens to share a population.
    per_row = output.with_suffix(".rows.csv")
    pd.DataFrame(
        {
            "key": rows.loc[matched, "key"].to_numpy(),
            "player_name": rows.loc[matched, "player_name"].to_numpy(),
            "position": rows.loc[matched, "position"].to_numpy(),
            "adp_rank": rank_of[matched].to_numpy(),
            "observed": observed[matched],
            "predicted": samples[matched].mean(axis=1),
        }
    ).to_csv(per_row, index=False)
    print(f"wrote {per_row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
