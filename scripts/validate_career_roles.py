"""Multi-year role history where the volume layer has none, and a better
construction where it has the weaker one.

Two different questions, run together because they share a harness.

The **role share** models -- who gets the targets and who gets the carries --
read no multi-year history at all. ``prior_target_role_3yr`` and
``prior_carry_role_3yr`` are built by ``season_pathways`` and reach nothing. An
observed season share is a noisy measurement of the role a player holds, so this
is the same problem the efficiency layer had, on the layer that dominates
season-total error: volume MAE runs 11.9 to 12.7 against efficiency responses
measured in single yards.

The **snap** model already reads ``prior_snap_share_3yr``, so there the question
is not whether history helps but whether it is being built the right way. The
``_3yr`` columns are an EWMA of each season's *share*, in which a three-game
season and a seventeen-game season weigh the same. ``prior_snap_share_career``
accumulates the player's snaps and his team's snaps separately with 0.7 decay,
so a season counts for what it holds. That construction beat the rate-EWMA on
every efficiency response.

Arms per stream:

    baseline    the shipping model
    career      plus that stream's own decayed exposure-weighted share

Snaps observed in the holdout are handed to the share models in both arms, so
the arms differ only in what they know about the player's history and not in how
much football he played.

    python scripts/validate_career_roles.py --holdouts 2023
    python scripts/validate_career_roles.py --merge a.json b.json c.json
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.models.season_opportunity import SeasonSnapShareModel
from ffmodel.models.volume_season_average import SeasonRosterShareModel

MATERIAL = 0.0025

# stream -> (observed column, the career feature it gains)
SHARE_STREAMS = {
    "target": ("target_share", "prior_target_share_career"),
    "carry": ("carry_share", "prior_carry_share_career"),
}
SNAP_FEATURE = "prior_snap_share_career"
ARMS = ("baseline", "career")


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    mean = samples.mean(axis=1)
    return {
        "mae": float(np.abs(observed - mean).mean()),
        "rmse": float(np.sqrt(np.mean((observed - mean) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "n": int(len(observed)),
    }


def _observed_snaps(prepared: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(
            prepared.get("snap_share", pd.Series(np.nan, index=prepared.index)),
            errors="coerce",
        )
        .fillna(pd.to_numeric(prepared.get("observed_availability"), errors="coerce"))
        .fillna(0.5)
        .clip(1e-5, 1.0)
        .to_numpy(dtype=float)
    )


def _evaluate_share(train, test, stream, arm, *, fit_kwargs, seed):
    observed_column, feature = SHARE_STREAMS[stream]
    extra = (feature,) if arm == "career" else ()
    model = SeasonRosterShareModel(stream=stream, extra_features=extra)
    model.fit(train, **fit_kwargs)
    # One exposure column per posterior draw -- chains times draws, not the
    # --draws argument.
    sizes = model.idata.posterior.sizes
    draws = int(sizes["chain"]) * int(sizes["draw"])
    prepared = model._design(test)["rows"]
    snaps = _observed_snaps(prepared)
    prediction = model.predict_share_samples(
        test, snap_samples=np.repeat(snaps[:, None], draws, axis=1), seed=seed
    )
    rows = prediction.rows
    observed = pd.to_numeric(
        rows[observed_column], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    samples = prediction.shares
    out = {"overall": _metrics(observed, samples), "features": len(model.feature_names)}
    # A pooled average over the roster is dominated by zero-share filler that
    # moves for other reasons; the population the feature is about is players
    # who actually hold a role.
    holds_role = observed > 0.02
    if holds_role.sum() >= 20:
        out["holds_role"] = _metrics(observed[holds_role], samples[holds_role])
    history = pd.to_numeric(rows.get(feature), errors="coerce").to_numpy()
    veteran = np.isfinite(history) & holds_role
    if veteran.sum() >= 20:
        out["has_history"] = _metrics(observed[veteran], samples[veteran])
    del model, prediction, samples
    gc.collect()
    return out


def _evaluate_snap(train, test, arm, *, fit_kwargs, seed):
    model = SeasonSnapShareModel()
    if arm == "career":
        model.extra_features = tuple(
            dict.fromkeys((*model.extra_features, SNAP_FEATURE))
        )
    model.fit(train, **fit_kwargs)
    prediction = model.predict_samples(test, seed=seed)
    rows = prediction.rows
    observed = pd.to_numeric(rows["snap_share"], errors="coerce").to_numpy(dtype=float)
    samples = np.asarray(prediction.shares, dtype=float)
    keep = np.isfinite(observed) & np.isfinite(samples).all(axis=1)
    observed, samples = observed[keep], samples[keep]
    out = {"overall": _metrics(observed, samples), "features": len(model.feature_names)}
    playing = observed > 0.10
    if playing.sum() >= 20:
        out["plays"] = _metrics(observed[playing], samples[playing])
    del model, prediction, samples
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = sorted(int(k) for k in folds)
    for stream in report["streams"]:
        print(f"\n{'=' * 88}\n{stream}\n{'=' * 88}")
        populations = sorted(
            {p for h in holdouts for p in folds[str(h)][stream]["baseline"]
             if p != "features"}
        )
        for population in populations:
            rows = []
            for arm in ARMS:
                values = [
                    folds[str(h)][stream][arm][population]
                    for h in holdouts
                    if population in folds[str(h)][stream][arm]
                ]
                if not values:
                    continue
                rows.append({
                    "arm": arm,
                    "n": int(np.mean([v["n"] for v in values])),
                    **{m: float(np.mean([v[m] for v in values]))
                       for m in ("mae", "crps", "coverage_80")},
                })
            if len(rows) < 2:
                continue
            table = pd.DataFrame(rows).set_index("arm")
            base = table.loc["baseline"]
            for metric in ("mae", "crps"):
                table[f"{metric}_delta"] = (table[metric] - base[metric]) / base[metric]
            for arm in ARMS:
                scored = [
                    h for h in holdouts if population in folds[str(h)][stream][arm]
                ]
                wins = sum(
                    folds[str(h)][stream][arm][population]["crps"]
                    < folds[str(h)][stream]["baseline"][population]["crps"]
                    for h in scored
                )
                table.loc[arm, "crps_folds_won"] = f"{wins}/{len(scored)}"
            print(f"\n-- {population} (n~{int(table['n'].iloc[0])}) --")
            print(table[
                ["mae", "crps", "coverage_80", "mae_delta", "crps_delta", "crps_folds_won"]
            ].to_string(float_format=lambda v: f"{v:.6f}" if abs(v) > 1e-4 else f"{v:+.2%}"))
    print(f"\nmateriality floor {MATERIAL:.2%}; a smaller move is not a result")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    print(f"wrote {args.report_json}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--streams", nargs="+", default=["target", "carry", "snap"])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--tune", type=int, default=400)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/career_roles.json")
    )
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        streams: list[str] = []
        for path in args.merge:
            blob = json.loads(path.read_text("utf-8"))
            folds.update(blob["folds"])
            streams = blob.get("streams", streams)
        return _report({"streams": streams, "folds": folds}, args)

    player_rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    player_rows = player_rows[player_rows.season.lt(2026)]
    for column in (*(f for _, f in SHARE_STREAMS.values()), SNAP_FEATURE):
        if column not in player_rows:
            raise SystemExit(f"{column} is not in the cache; rebuild it first")
        print(f"{column}: {int(player_rows[column].notna().sum())} rows", flush=True)

    fit_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    report: dict[str, object] = {"streams": args.streams, "folds": {}}
    for holdout in args.holdouts:
        train = player_rows[player_rows.season.lt(holdout)].copy()
        test = player_rows[player_rows.season.eq(holdout)].copy()
        fold: dict = {}
        for stream in args.streams:
            fold[stream] = {}
            for arm in ARMS:
                fold[stream][arm] = (
                    _evaluate_snap(train, test, arm, fit_kwargs=fit_kwargs, seed=args.seed)
                    if stream == "snap"
                    else _evaluate_share(
                        train, test, stream, arm, fit_kwargs=fit_kwargs, seed=args.seed
                    )
                )
                block = fold[stream][arm]["overall"]
                print(
                    f"{holdout} {stream:7s} {arm:9s} CRPS {block['crps']:.6f}  "
                    f"MAE {block['mae']:.6f}  feat {fold[stream][arm]['features']}",
                    flush=True,
                )
        report["folds"][str(holdout)] = fold
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
