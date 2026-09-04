"""Does the pipeline use ADP as hard as the data says it should?

The suspicion behind task 42 is attenuation: the ADP columns are in the design
matrix, and the model still loses to a regression on the same columns, so
something is shrinking their influence between the frame and the prediction.
Three mechanisms are available and they are not the same thing.

* **The prior.** Every submodel puts one ``Normal(0, feature_prior_scale)`` on
  the whole coefficient vector. A feature whose honest effect is larger than
  that prior allows gets pulled in, and adding more columns spreads the same
  prior mass thinner.
* **The rotation.** ``_matrix`` projects the standardized features onto their
  principal directions before the likelihood sees them, so no coefficient
  belongs to a feature. A direction dominated by prior-usage columns carries
  whatever ADP loading happens to align with it, and the ADP-specific part can
  be shrunk as a side effect of shrinking something else.
* **The target.** The probe that motivated the interaction terms regressed
  *fantasy points* on rank. The submodels regress *snap share* and *role
  shares* on rank. A term that helps for points need not help for exposure.

This measures the first two directly. It fits the snap model as shipped, reads
the implied per-feature coefficients back through the projection the way
docs/snap-prior-2026-08.md does, and compares them against an unregularized
least-squares fit of the same target on the same standardized columns. The
ratio is how much of the data's own signal survives the prior and the rotation.

A ratio near one means the prior is not binding and attenuation is not the
story. A ratio well under one, concentrated on the ADP columns, means it is.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.features.market import ADP_FEATURES, ADP_INTERACTION_FEATURES
from ffmodel.models.season_opportunity import SeasonSnapShareModel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=2022)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--interactions", action="store_true")
    parser.add_argument("--prior-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path(
        f"scripts/validation_runs/adp_attenuation_{args.holdout}.json"
    )

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    train = rows[rows.season < args.holdout].copy()

    extra = ADP_FEATURES + (ADP_INTERACTION_FEATURES if args.interactions else ())
    model = SeasonSnapShareModel()
    model.extra_features = tuple(dict.fromkeys((*model.extra_features, *extra)))
    if args.prior_scale is not None:
        model.feature_prior_scale = args.prior_scale
    model.fit(train, draws=args.draws, tune=args.draws, chains=args.chains)

    # Reproduce the model's own filter and response rather than approximating
    # them. The snap model fits ``snap_share / observed_availability`` on rows
    # with an observed positive snap count, and builds its design on exactly
    # those rows; comparing against a regression on any other population would
    # be measuring the population, not the prior.
    prepared = model._prepare(train)
    snap_observed = (
        pd.to_numeric(prepared.get("snap_counts_observed"), errors="coerce")
        .fillna(0)
        .gt(0)
    )
    snap_share = pd.to_numeric(prepared.get("snap_share"), errors="coerce")
    availability = pd.to_numeric(
        prepared.get("observed_availability"), errors="coerce"
    )
    valid = snap_observed & snap_share.gt(0) & availability.gt(0)
    prepared = prepared[valid].reset_index(drop=True)
    response = np.clip(
        snap_share[valid].to_numpy(float) / availability[valid].to_numpy(float),
        1e-4,
        1.0 - 1e-4,
    )
    projection = np.asarray(model.feature_projection, dtype=float)
    standardized = np.column_stack(
        [
            (
                pd.to_numeric(prepared[name], errors="coerce")
                .fillna(model.feature_fill[name])
                .to_numpy(float)
                - model.feature_mean[name]
            )
            / model.feature_scale[name]
            for name in model.feature_names
        ]
    )

    beta = (
        model.idata.posterior["beta"]
        .stack(sample=("chain", "draw"))
        .to_numpy()
        .mean(axis=1)
    )
    # Back through the rotation: what the model implies per original feature.
    implied = projection @ beta

    # The same target on the same rows, with no prior and no rotation. Position
    # dummies and an intercept are carried so the free fit has the same nuisance
    # structure the model gets from its intercept and position effect -- without
    # them the ADP columns would absorb level differences the model never asks
    # them to carry, and the comparison would overstate what was shrunk.
    logit_response = np.log(response) - np.log(1.0 - response)
    dummies = [np.ones(len(prepared))]
    for position in sorted(prepared["position"].unique())[:-1]:
        dummies.append(prepared["position"].eq(position).to_numpy(float))
    free_design = np.column_stack([standardized, *dummies])
    solution, *_ = np.linalg.lstsq(free_design, logit_response, rcond=None)
    ols = solution[: standardized.shape[1]]

    adp_names = set(extra)
    report = []
    for name, model_beta, free_beta in zip(model.feature_names, implied, ols):
        report.append(
            {
                "feature": name,
                "is_adp": name in adp_names,
                "model": float(model_beta),
                "unregularized": float(free_beta),
                "ratio": float(model_beta / free_beta) if abs(free_beta) > 1e-9 else None,
            }
        )

    print(
        f"\nSNAP MODEL, TRAINED BEFORE {args.holdout}, PRIOR "
        f"{model.feature_prior_scale}"
        + (" (with interactions)" if args.interactions else "")
        + "\n"
    )
    print(f"  {'feature':28s} {'model':>9s} {'free':>9s} {'kept':>7s}")
    for row in sorted(report, key=lambda r: (not r["is_adp"], r["feature"])):
        kept = f"{row['ratio']:>6.2f}x" if row["ratio"] is not None else "     -"
        mark = "*" if row["is_adp"] else " "
        print(
            f" {mark}{row['feature']:28s} {row['model']:>9.3f} "
            f"{row['unregularized']:>9.3f} {kept:>7s}"
        )

    # Median of the per-feature ratios, not a ratio of magnitudes. Two features
    # here are near-collinear -- depth_rank and is_replacement_player both
    # separate backups from starters -- so the unregularized fit gives them huge
    # cancelling coefficients whose difference is identified and whose levels
    # are not. Their magnitude swamps a root-mean-square and would report the
    # whole feature set as 93% shrunk when almost none of it is.
    def kept(selector) -> tuple[float, int]:
        ratios = [
            r["ratio"] for r in report if selector(r) and r["ratio"] is not None
        ]
        return (float(np.median(ratios)) if ratios else float("nan"), len(ratios))

    adp_ratio, adp_n = kept(lambda r: r["is_adp"])
    other_ratio, other_n = kept(lambda r: not r["is_adp"])
    collinear = sorted(
        (r for r in report if r["ratio"] is not None),
        key=lambda r: abs(r["unregularized"]),
        reverse=True,
    )[:2]
    print(
        f"\n  median coefficient kept, ADP columns ({adp_n}):   {adp_ratio:.2f}x"
    )
    print(
        f"  median coefficient kept, every other ({other_n}):  {other_ratio:.2f}x"
    )
    print(
        "  largest unregularized coefficients (collinear, not shrunk): "
        + ", ".join(f"{r['feature']} {r['unregularized']:+.1f}" for r in collinear)
    )
    print(
        "\n  A ratio near one on the ADP columns means the prior is not binding "
        "on\n  them and attenuation is not the mechanism -- which would leave "
        "the\n  target: these columns predict snap share, not points."
    )

    payload = {
        "holdout": args.holdout,
        "prior_scale": model.feature_prior_scale,
        "interactions": args.interactions,
        "features": report,
        "median_kept": {"adp": adp_ratio, "other": other_ratio},
        "collinear": [r["feature"] for r in collinear],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
