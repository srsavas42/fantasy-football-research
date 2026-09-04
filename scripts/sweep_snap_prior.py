"""Does a wider feature prior fix the backup-quarterback exposure error?

The carry allocator over-projects quarterbacks by a third, and substituting
observed snap share removes two thirds of that, so the error is in the exposure
the allocator reads. Measured on the snap model's own response, the mean is
right for starters and wrong for backups: predicted 0.253 against 0.209
observed, a 21% over-projection, while starters land at 0.856 against 0.867.

Under the historical prior the implied effects on ``depth_rank`` and
``is_replacement_player`` sit 4.4 and 5.5 prior standard deviations from zero
with posterior widths below the prior's own. Both separate backups. That is what
a prior still pulling against a confident likelihood looks like.

The prior scale is the only lever available. ``_matrix`` rotates the
standardized features onto their principal directions before the likelihood
sees them, so a coefficient belongs to a combination of features and there is
nothing feature-specific to widen.

Scored on a held-out season, because a wider prior fits the training data better
by construction and that says nothing. Reported: the backup and starter means the
allocator actually reads, the two straining coefficients, and held-out snap
accuracy for every position -- widening is global, so the cost lands on the
positions that were already right.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps
from ffmodel.models.season_opportunity import SeasonSnapShareModel

WATCH = ("depth_rank", "is_replacement_player", "qb_listed_starter")


def conditional_response(rows: pd.DataFrame) -> np.ndarray:
    share = pd.to_numeric(rows["snap_share"], errors="coerce").to_numpy(float)
    available = pd.to_numeric(
        rows["observed_availability"], errors="coerce"
    ).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(share / available, 1e-4, 1.0 - 1e-4)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025"))
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[0.35, 0.75, 1.5, 3.0])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path(
        f"scripts/validation_runs/snap_prior_{args.holdout}.json"
    )

    pr = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    train = pr[pr.season < args.holdout].copy()
    test = pr[pr.season == args.holdout].copy()

    report: dict[str, object] = {"holdout": args.holdout, "scales": {}}
    print(f"\nSNAP FEATURE PRIOR SWEEP, trained on <{args.holdout}, scored on "
          f"{args.holdout}\n")
    for scale in args.scales:
        model = SeasonSnapShareModel(feature_prior_scale=scale)
        model.fit(train, draws=args.draws, tune=args.draws, chains=4)

        beta = model.idata.posterior["beta"].stack(sample=("chain", "draw")).to_numpy()
        projection = np.asarray(model.feature_projection, dtype=float)
        implied = projection @ beta
        straining = {}
        for name in WATCH:
            if name not in model.feature_names:
                continue
            j = model.feature_names.index(name)
            prior_sd = scale * float(np.linalg.norm(projection[j]))
            straining[name] = {
                "implied_mean": float(implied[j].mean()),
                "posterior_sd": float(implied[j].std()),
                "prior_sd": prior_sd,
                "prior_sds_out": float(abs(implied[j].mean()) / max(prior_sd, 1e-9)),
            }

        prediction = model.predict_samples(test, seed=42)
        rows = prediction.rows.reset_index(drop=True)
        observed = conditional_response(rows)
        drawn = np.asarray(prediction.conditional_share, dtype=float)
        keep = (
            np.isfinite(observed)
            & pd.to_numeric(rows.get("snap_counts_observed"), errors="coerce")
            .fillna(0)
            .gt(0)
            .to_numpy()
        )
        starter = pd.to_numeric(rows["qb_listed_starter"], errors="coerce").fillna(0)
        quarterback = rows["position"].eq("QB").to_numpy()

        groups: dict[str, dict[str, float]] = {}
        bands = [
            ("QB starter", keep & quarterback & starter.eq(1).to_numpy()),
            ("QB backup", keep & quarterback & starter.ne(1).to_numpy()),
        ]
        bands += [
            (position, keep & rows["position"].eq(position).to_numpy())
            for position in ("RB", "WR", "TE")
        ]
        for label, mask in bands:
            if mask.sum() < 5:
                continue
            truth = observed[mask]
            sample = drawn[mask]
            groups[label] = {
                "n": int(mask.sum()),
                "observed": float(truth.mean()),
                "predicted": float(sample.mean(axis=1).mean()),
                "bias": float(sample.mean(axis=1).mean() - truth.mean()),
                "mae": float(np.abs(truth - sample.mean(axis=1)).mean()),
                "crps": float(empirical_crps(truth, sample).mean()),
            }

        report["scales"][str(scale)] = {"straining": straining, "groups": groups}
        print(f"  prior scale {scale}")
        for name, values in straining.items():
            print(
                f"    {name:24s} implied {values['implied_mean']:+7.3f} "
                f"({values['prior_sds_out']:.2f} prior sd)"
            )
        print(
            f"    {'group':12s} {'n':>5s} {'observed':>9s} {'predicted':>10s} "
            f"{'bias':>8s} {'MAE':>7s} {'CRPS':>7s}"
        )
        for label, values in groups.items():
            print(
                f"    {label:12s} {values['n']:>5d} {values['observed']:>9.3f} "
                f"{values['predicted']:>10.3f} {values['bias']:>+8.3f} "
                f"{values['mae']:>7.4f} {values['crps']:>7.4f}"
            )
        print("", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
