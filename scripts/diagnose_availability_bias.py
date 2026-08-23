"""Where does the availability under-projection come from?

Projected games run 3.7% under observed pooled and 6.0% under on the drafted
pool, and the sign flips between drafted and undrafted for backs and receivers
(-6.3%/+5.0% and -8.8%/+4.5%). Two very different faults produce numbers like
that and they want opposite fixes.

**Misspecification.** The model cannot represent the relationship, so it is
biased even on the rows it was fitted to. An intercept correction helps.

**Generalization.** The model fits its training rows and misses on new ones,
either because the population shifted or because it shrinks too hard toward the
mean. An intercept correction papers over it and makes the undrafted worse,
since their bias has the opposite sign.

Fitting the availability layer alone and scoring it on *both* populations tells
them apart. Only this layer is fitted -- it is the whole question, and it costs
a minute rather than the quarter-hour a scoring fit takes.

Era drift is already ruled out: mean availability by season runs 0.799, 0.581,
0.785, 0.774, 0.672, 0.732, 0.670 over 2015-2021 against 0.669-0.700 across the
holdouts, so the training seasons are if anything *more* available than the
test ones and drift would bias the projection high.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.features.market import ADP_FEATURES
from ffmodel.models.season_availability import SeasonAvailabilityModel

POSITIONS = ("QB", "RB", "WR", "TE")


def summarise(name: str, rows: pd.DataFrame, projected: np.ndarray) -> dict:
    observed = pd.to_numeric(rows["games"], errors="coerce").to_numpy(float)
    keep = np.isfinite(projected) & np.isfinite(observed)
    p, o = projected[keep], observed[keep]
    if len(p) < 10:
        return {"population": name, "n": int(len(p))}
    return {
        "population": name,
        "n": int(len(p)),
        "projected": float(p.mean()),
        "observed": float(o.mean()),
        "bias": float((p - o).mean()),
        "bias_pct": float((p - o).mean() / o.mean()) if o.mean() else float("nan"),
    }


def report_for(label: str, rows: pd.DataFrame, projected: np.ndarray) -> dict:
    drafted = pd.to_numeric(
        rows.get("adp_drafted", pd.Series(0, index=rows.index)), errors="coerce"
    ).eq(1).to_numpy()
    out: dict[str, object] = {
        "label": label,
        "pooled": [
            summarise("all", rows, projected),
            summarise("drafted", rows[drafted], projected[drafted]),
            summarise("undrafted", rows[~drafted], projected[~drafted]),
        ],
        "positions": {},
    }
    for position in POSITIONS:
        at = rows["position"].eq(position).to_numpy()
        if at.sum() < 30:
            continue
        out["positions"][position] = [
            summarise("all", rows[at], projected[at]),
            summarise("drafted", rows[at & drafted], projected[at & drafted]),
            summarise("undrafted", rows[at & ~drafted], projected[at & ~drafted]),
        ]
    return out


def table(title: str, entries: list[dict]) -> None:
    print(f"  {title}")
    print(f"    {'population':11s} {'n':>5s} {'proj':>7s} {'obs':>7s} {'bias %':>8s}")
    for e in entries:
        if "bias" not in e:
            continue
        print(
            f"    {e['population']:11s} {e['n']:>5d} {e['projected']:>7.2f} "
            f"{e['observed']:>7.2f} {e['bias_pct']:>+7.1%}"
        )
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument(
        "--adp",
        action="store_true",
        help="give the availability regression the preseason draft board. The "
             "measured ADP ablation deliberately withheld it from this layer "
             "-- see _enable_market_adp_features -- so its null result says "
             "nothing about this arm",
    )
    parser.add_argument(
        "--position-slopes",
        action="store_true",
        help="give each position its own slope vector, drawn around a shared "
             "mean, instead of one vector for all four",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    label = args.label or (
        ("adp" if args.adp else "base") + ("_pos" if args.position_slopes else "")
    )
    output = args.output or Path(
        f"scripts/validation_runs/availability_bias_{args.holdout}_{label}.json"
    )

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    rows = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ]
    train = rows[rows.season < args.holdout].copy()
    test = rows[rows.season == args.holdout].copy()

    model = SeasonAvailabilityModel()
    if args.adp:
        missing = [name for name in ADP_FEATURES if name not in rows.columns]
        if missing:
            raise SystemExit(
                f"--adp needs {missing}, absent from {args.cache_dir}. Build "
                "them with scripts/augment_cache_features.py --feature market-adp"
            )
        model.extra_features = ADP_FEATURES
    model.position_varying_slopes = args.position_slopes
    model.fit(train, draws=args.draws, tune=args.draws, chains=4)
    print(f"availability layer fitted ({len(model.feature_names)} features)", flush=True)

    report = {"holdout": args.holdout, "adp": bool(args.adp), "arms": []}
    for label, frame, seed in (("in-sample (train)", train, 11), ("held out", test, 12)):
        prediction = model.predict_samples(frame, seed=seed)
        projected = prediction.games_active.mean(axis=1)
        entry = report_for(label, prediction.rows.reset_index(drop=True), projected)
        report["arms"].append(entry)

    print(f"\nAVAILABILITY, HOLDOUT {args.holdout}\n")
    for entry in report["arms"]:
        print(f"=== {entry['label']}")
        table("POOLED", entry["pooled"])
        for position, entries in entry["positions"].items():
            table(position, entries)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
