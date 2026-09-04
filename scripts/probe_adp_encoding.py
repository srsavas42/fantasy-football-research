"""Does re-encoding the ADP interaction survive the prior that broke it?

The interaction terms are informative — they cut logit snap-share error 4.11% in
a free fit on the submodel's own response — and they made the pipeline worse,
because the encoding is collinear and a shared ``Normal(0, 0.35)`` over every
coefficient cannot hold large opposing values. The block's median kept
coefficient fell from 1.13x to 0.25x.

Two encodings span exactly the same space:

* **deviations** (what was shipped): a shared rank slope plus one masked column
  per position, so each interaction coefficient is that position's *departure*
  from the common slope.
* **absolute**: one masked column per position and no shared slope, so each
  coefficient is that position's own slope.

At zero penalty these are the same model and give identical predictions. Under a
penalty they are not, because an isotropic Gaussian prior is not invariant to
reparameterisation: it prefers whichever encoding puts the truth closer to the
origin. If the per-position slopes are similar to each other, deviations are
small and the deviation encoding wins. If they differ a lot — which is the whole
premise of wanting an interaction — the deviations are large and the absolute
encoding wins.

This settles which, on the snap model's own response, own rows and own filter,
with ridge standing in for the Gaussian prior. Ridge *is* that prior: the
posterior mean of a Normal likelihood with a ``Normal(0, tau)`` coefficient prior
is the ridge solution at ``lambda = sigma^2 / tau^2``. Sweeping lambda covers
whatever the implied noise scale turns out to be, so the answer does not depend
on pinning it down.

Cheap on purpose. The last interaction went into the pipeline on the strength of
a probe against the wrong objective and cost two walk-forward arms to disprove.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.models.season_opportunity import SNAP_FEATURES, SeasonSnapShareModel

POSITIONS = ("QB", "RB", "WR", "TE")


def snap_rows(cache_dir: Path) -> pd.DataFrame:
    """The snap model's population and response, reproduced exactly."""
    rows = pd.read_pickle(cache_dir / "player_rows.pkl")
    prepared = SeasonSnapShareModel()._prepare(rows)
    observed = (
        pd.to_numeric(prepared.get("snap_counts_observed"), errors="coerce")
        .fillna(0)
        .gt(0)
    )
    share = pd.to_numeric(prepared.get("snap_share"), errors="coerce")
    availability = pd.to_numeric(prepared.get("observed_availability"), errors="coerce")
    valid = observed & share.gt(0) & availability.gt(0)
    out = prepared[valid].reset_index(drop=True)
    response = np.clip(
        share[valid].to_numpy(float) / availability[valid].to_numpy(float), 1e-4, 1 - 1e-4
    )
    out["snap_logit"] = np.log(response) - np.log(1 - response)
    return out


def columns(frame: pd.DataFrame, encoding: str) -> dict[str, np.ndarray]:
    base = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        for name in SNAP_FEATURES
        if name in frame
    }
    rank = pd.to_numeric(frame["adp_log_rank"], errors="coerce").to_numpy(float)
    drafted = pd.to_numeric(frame["adp_drafted"], errors="coerce").to_numpy(float)
    base["adp_position_log_rank"] = pd.to_numeric(
        frame["adp_position_log_rank"], errors="coerce"
    ).to_numpy(float)
    if encoding == "none":
        base["adp_log_rank"] = rank
        base["adp_drafted"] = drafted
        return base
    if encoding == "deviations":
        base["adp_log_rank"] = rank
        base["adp_drafted"] = drafted
        for position in POSITIONS[:-1]:
            mask = frame.position.eq(position).to_numpy(float)
            base[f"rank_x_{position}"] = mask * rank
            base[f"drafted_x_{position}"] = mask * drafted
        return base
    if encoding == "absolute":
        for position in POSITIONS:
            mask = frame.position.eq(position).to_numpy(float)
            base[f"rank_{position}"] = mask * rank
            base[f"drafted_{position}"] = mask * drafted
        return base
    raise ValueError(encoding)


def build(train, test, encoding):
    tr, te = columns(train, encoding), columns(test, encoding)
    names = sorted(tr)
    X, Z = [], []
    for name in names:
        values = tr[name]
        fill = np.nanmedian(values) if np.isfinite(values).any() else 0.0
        filled = np.where(np.isfinite(values), values, fill)
        mean, scale = filled.mean(), filled.std()
        scale = scale if scale > 1e-8 else 1.0
        X.append((filled - mean) / scale)
        other = te[name]
        X_test = np.where(np.isfinite(other), other, fill)
        Z.append((X_test - mean) / scale)
    # Position dummies and an intercept, left unpenalised, standing in for the
    # model's own intercept and position effect which do not share ``beta``'s
    # prior.
    free_tr = [np.ones(len(train))] + [
        train.position.eq(p).to_numpy(float) for p in POSITIONS[:-1]
    ]
    free_te = [np.ones(len(test))] + [
        test.position.eq(p).to_numpy(float) for p in POSITIONS[:-1]
    ]
    return (
        np.column_stack(X),
        np.column_stack(Z),
        np.column_stack(free_tr),
        np.column_stack(free_te),
    )


def ridge_fit(X, free, y, lam):
    """Ridge as augmented least squares, so a rank-deficient design is fine.

    Both encodings are deficient by construction — the position masks sum to the
    main effect, and roster_active and roster_reserve are complementary — so the
    normal equations are singular at zero penalty. ``lstsq`` returns the
    minimum-norm solution instead of raising, which is also what the model's own
    SVD projection does with a null direction.
    """
    design = np.column_stack([X, free])
    penalty = np.zeros(design.shape[1])
    penalty[: X.shape[1]] = np.sqrt(lam)
    augmented = np.vstack([design, np.diag(penalty)])
    target = np.concatenate([y, np.zeros(design.shape[1])])
    beta, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return beta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path("scripts/validation_runs/adp_encoding_probe.json")

    rows = snap_rows(args.cache_dir)
    lambdas = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
    results: dict[str, dict[str, float]] = {}
    for encoding in ("none", "deviations", "absolute"):
        for lam in lambdas:
            errors, n = 0.0, 0
            for holdout in args.holdouts:
                train = rows[rows.season < holdout]
                test = rows[rows.season == holdout]
                X, Z, free_tr, free_te = build(train, test, encoding)
                beta = ridge_fit(X, free_tr, train.snap_logit.to_numpy(float), lam)
                predicted = np.column_stack([Z, free_te]) @ beta
                errors += np.abs(predicted - test.snap_logit.to_numpy(float)).sum()
                n += len(test)
            results.setdefault(encoding, {})[str(lam)] = errors / n

    print("\nOUT-OF-SAMPLE LOGIT SNAP-SHARE MAE, POOLED OVER "
          f"{args.holdouts}\n")
    print(f"  {'lambda':>8s} " + " ".join(f"{e:>12s}" for e in results))
    for lam in lambdas:
        row = " ".join(f"{results[e][str(lam)]:>12.4f}" for e in results)
        print(f"  {lam:>8.0f} {row}")

    best = {e: min(v.values()) for e, v in results.items()}
    print("\n  best over lambda: " + ", ".join(f"{e} {v:.4f}" for e, v in best.items()))
    dev, absolute, none = best["deviations"], best["absolute"], best["none"]
    print(
        f"  absolute vs deviations at their own optima: "
        f"{(absolute - dev) / dev:+.2%}"
    )
    print(
        f"  each vs no interaction: deviations {(dev - none) / none:+.2%}, "
        f"absolute {(absolute - none) / none:+.2%}"
    )
    print(
        "\n  At lambda=0 the two encodings are the same model and must agree; "
        "any\n  divergence as lambda grows is the prior preferring one "
        "parameterisation."
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
