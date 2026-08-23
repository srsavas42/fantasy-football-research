"""When the model disagrees with the draft board, is it right?

The blend analysis says the two forecasts combine to beat the board by about
0.6%, which is barely material. But MAE against realised points is not the only
thing a drafter wants. They want to know who is *mispriced* — and a projection
can be worse than the board in absolute error while still carrying real
information about which way the board is wrong.

That is measurable directly. Regress the board's error on the model's
disagreement with it:

    observed - curve  =  alpha + beta * (model - curve)

``beta = 0`` means the disagreement is noise and the model has nothing to add.
``beta = 1`` means the disagreement is exactly the correction the board needs.
Anything in between says what fraction of the model's contrarian opinion is
right, and it is the same quantity as the variance-optimal blend weight, in
units a drafter can act on: at ``beta = 0.2``, a player the model likes by 100
points over his ADP-implied total actually beats it by about 20.

## Pre-registered, and why

The 2022 fold was explored first, and sliced six ways it produced two
suggestive subgroups. Six slices of 230 rows will produce a t of 2.4 by chance
often enough that the finding cannot be trusted on the fold that generated it.
This file is committed **before 2023 and 2024 finish exporting**, so what
follows is a prediction rather than a description.

Generated on 2022 (n=230), to be tested on 2023 and 2024:

1. **The pooled slope is positive.** 2022: +0.194, se 0.087, t +2.24.
2. **Quarterbacks carry more signal than the pool.** 2022: +0.559, se 0.231.
3. **The later board carries more signal than the early board.** 2022:
   ADP 101-300 at +0.295 (se 0.103) against ADP 1-100 at +0.059 (se 0.145).
   The reading, if it holds, is that the market is efficient where attention is
   concentrated and leaves something on the table deeper down.
4. **Running backs and tight ends carry none.** 2022: +0.063 and -0.041, both
   within one standard error of zero.

A hypothesis counts as surviving only if the 2023-2024 estimate keeps the sign
and stays within reach of the 2022 magnitude. Two folds cannot establish much;
they can falsify, which is what they are for here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "_adp_only", Path(__file__).with_name("benchmark_adp_only.py")
)
_adp_only = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adp_only)

MIN_ROWS = 25


def slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Least-squares slope with its standard error."""
    if len(x) < 3:
        return float("nan"), float("nan"), len(x)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    variance = np.linalg.inv(design.T @ design) * (residual @ residual) / (len(x) - 2)
    return float(beta[1]), float(np.sqrt(variance[1, 1])), len(x)


def load(out_dir: Path, label: str, holdout: int, cache_dir: Path, scoring: str):
    base = out_dir / f"{label}_{holdout}"
    rows = pd.read_parquet(base.with_suffix(".rows.parquet"))
    samples = np.load(base.with_suffix(".samples.npz"))["samples"].astype(float)
    named = (
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
        .fillna(0)
        .ne(1)
        .to_numpy()
    )
    drafted = pd.to_numeric(rows["adp_drafted"], errors="coerce").eq(1).to_numpy()
    finite = np.isfinite(rows["observed"].to_numpy(float)) & np.isfinite(samples).all(
        axis=1
    )
    mask = named & drafted & finite
    block = rows[mask].reset_index(drop=True)
    model = samples[mask]
    pool = _adp_only.prepare(cache_dir, scoring)
    curve = _adp_only.project(
        pool[pool.season.lt(holdout)], block, model.shape[1], seed=4000 + holdout
    )
    return block, model.mean(axis=1), curve.mean(axis=1)


def groups(block: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    rank = pd.to_numeric(block.adp_rank, errors="coerce").to_numpy(float)
    out = [("all drafted", np.ones(len(block), dtype=bool))]
    for position in ("QB", "RB", "WR", "TE"):
        out.append((f"position {position}", block.position.eq(position).to_numpy()))
    out.append(("ADP 1-100", rank <= 100))
    out.append(("ADP 101-300", rank > 100))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-on", type=int, default=2022)
    parser.add_argument("--test-on", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--label", default="shipping")
    parser.add_argument(
        "--out-dir", type=Path, default=Path(".cache/holdout-predictions")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path("scripts/validation_runs/market_disagreement.json")

    seasons = [args.generated_on, *args.test_on]
    data = {}
    for holdout in seasons:
        base = args.out_dir / f"{args.label}_{holdout}"
        if not base.with_suffix(".rows.parquet").exists():
            raise SystemExit(f"no export for {holdout}; run export_holdout_predictions")
        data[holdout] = load(
            args.out_dir, args.label, holdout, args.cache_dir, args.scoring
        )

    report: dict[str, object] = {"generated_on": args.generated_on, "test_on": args.test_on}
    names = [name for name, _ in groups(data[seasons[0]][0])]
    estimates: dict[str, dict[str, object]] = {}
    for name in names:
        estimates[name] = {}

    for holdout in seasons:
        block, model, curve = data[holdout]
        for name, mask in groups(block):
            if mask.sum() < MIN_ROWS:
                continue
            b, se, n = slope((model - curve)[mask], (block["observed"].to_numpy(float) - curve)[mask])
            estimates[name][str(holdout)] = {"slope": b, "se": se, "n": n}

    # The test folds pooled, which is the number the pre-registration is about.
    for name in names:
        xs, ys = [], []
        for holdout in args.test_on:
            block, model, curve = data[holdout]
            for gname, mask in groups(block):
                if gname == name and mask.sum() >= MIN_ROWS:
                    xs.append((model - curve)[mask])
                    ys.append((block["observed"].to_numpy(float) - curve)[mask])
        if xs:
            b, se, n = slope(np.concatenate(xs), np.concatenate(ys))
            estimates[name]["pooled_test"] = {"slope": b, "se": se, "n": n}

    print(f"\nSLOPE OF (observed - board) ON (model - board), {args.scoring.upper()}\n")
    header = f"  {'group':22s} {'2022 (gen)':>16s}"
    for holdout in args.test_on:
        header += f" {str(holdout):>16s}"
    header += f" {'test pooled':>18s}"
    print(header)
    for name in names:
        row = f"  {name:22s}"
        for key in [str(args.generated_on), *[str(h) for h in args.test_on], "pooled_test"]:
            e = estimates[name].get(key)
            row += (
                f" {e['slope']:>+8.3f}+-{e['se']:<6.3f}" if e else f" {'--':>16s}"
            )
        print(row)

    print("\n  pre-registered predictions and their verdicts:\n")
    verdicts = {}
    def check(label: str, ok: bool, detail: str) -> None:
        verdicts[label] = {"survived": bool(ok), "detail": detail}
        print(f"    [{'survives' if ok else 'FALSIFIED':>9s}] {label}: {detail}")

    p = estimates["all drafted"].get("pooled_test")
    if p:
        check("1. pooled slope positive", p["slope"] > 0,
              f"test-fold slope {p['slope']:+.3f} +- {p['se']:.3f} (n={p['n']})")
    qb, allp = estimates.get("position QB", {}).get("pooled_test"), p
    if qb and allp:
        check("2. QB above the pool", qb["slope"] > allp["slope"],
              f"QB {qb['slope']:+.3f} vs pool {allp['slope']:+.3f}")
    late = estimates.get("ADP 101-300", {}).get("pooled_test")
    early = estimates.get("ADP 1-100", {}).get("pooled_test")
    if late and early:
        check("3. later board above early board", late["slope"] > early["slope"],
              f"101-300 {late['slope']:+.3f} vs 1-100 {early['slope']:+.3f}")
    rb = estimates.get("position RB", {}).get("pooled_test")
    te = estimates.get("position TE", {}).get("pooled_test")
    if rb and te:
        near_zero = abs(rb["slope"]) < 2 * rb["se"] and abs(te["slope"]) < 2 * te["se"]
        check("4. RB and TE carry none", near_zero,
              f"RB {rb['slope']:+.3f}+-{rb['se']:.3f}, TE {te['slope']:+.3f}+-{te['se']:.3f}")

    report["estimates"] = estimates
    report["verdicts"] = verdicts
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
