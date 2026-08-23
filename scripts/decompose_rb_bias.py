"""Where in the pipeline does the running-back under-projection come from?

Drafted running backs are projected about seventeen points light, on every
holdout, which is 12.8% of what they score. It is the largest position bias in
the package and the only one that holds its sign across folds -- quarterbacks
swing from +7.9 to -25.9 and average near zero, tight ends average +2.2.

Two explanations have already been ruled out.

**Omitted covariance.** ``E[XY] = E[X]E[Y] + Cov(X, Y)``, so a pipeline that
projects availability and per-game scoring separately and multiplies loses the
covariance term. For running backs that term is worth +2.1 points against a
-17.2 bias, because the correlation is only 0.071. It is not the mechanism.

**Conditioning on availability.** Splitting by whether a back played the season
shows -51 for the available and +12 for the injured, but any calibrated
projection averaging over availability shows that pattern by construction. It is
not diagnostic.

What the marginals say is narrower and more useful: the observed availability
mean times the observed per-game mean composes to 132.3, and the model projects
about 117. So the model is not failing to combine its components, it is
projecting at least one of them too low. This finds which.

The volume layer exposes every intermediate quantity -- games active, carries,
targets, and the shares they come from -- so each can be scored against what
actually happened. Only the volume pipeline is fitted, not efficiency, because
the question is about opportunity rather than what was done with it.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

POSITIONS = ("QB", "RB", "WR", "TE")


def summarise(name: str, projected: np.ndarray, observed: np.ndarray) -> dict:
    finite = np.isfinite(projected) & np.isfinite(observed)
    p, o = projected[finite], observed[finite]
    return {
        "component": name,
        "n": int(finite.sum()),
        "projected": float(p.mean()),
        "observed": float(o.mean()),
        "bias": float((p - o).mean()),
        "bias_pct": float((p - o).mean() / o.mean()) if o.mean() else float("nan"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=2024)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025-adp2")
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--position", default="RB")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path(
        f"scripts/validation_runs/rb_bias_{args.holdout}.json"
    )

    pr = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    tr = pd.read_pickle(args.cache_dir / "team_rows.pkl")
    train = SeasonAverageData(
        tr[tr.season < args.holdout].copy(), pr[pr.season < args.holdout].copy()
    )
    test = SeasonAverageData(
        tr[tr.season == args.holdout].copy(), pr[pr.season == args.holdout].copy()
    )
    pipeline = SeasonAverageVolumePipeline()
    pipeline.fit(train, draws=args.draws, tune=args.draws, chains=4)
    print("volume layer fitted", flush=True)
    prediction = pipeline.predict_samples(test, seed=42)

    rows = prediction.player_rows.reset_index(drop=True)
    named = (
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce")
        .fillna(0)
        .ne(1)
        .to_numpy()
    )
    drafted = pd.to_numeric(rows.get("adp_drafted"), errors="coerce").eq(1).to_numpy()
    report: dict[str, object] = {"holdout": args.holdout, "positions": {}}

    for position in POSITIONS:
        mask = named & drafted & rows["position"].eq(position).to_numpy()
        if mask.sum() < 10:
            continue
        entries = []
        # Exposure first: everything downstream is a rate multiplied by it.
        entries.append(
            summarise(
                "games",
                prediction.games_active.mean(axis=1)[mask],
                pd.to_numeric(rows["games"], errors="coerce").to_numpy()[mask],
            )
        )
        for name, projected, column in (
            ("carries", prediction.carries, "rush_att"),
            ("targets", prediction.targets, "targets"),
            ("carry_share", prediction.carry_share, None),
            ("target_share", prediction.target_share, None),
            ("snap_share", prediction.snap_share, "snap_share"),
        ):
            if column is None or column not in rows:
                if column is not None:
                    continue
                # Shares have no direct observed column; skip rather than invent
                # a denominator that would not match the model's own.
                continue
            entries.append(
                summarise(
                    name,
                    np.asarray(projected, dtype=float).mean(axis=1)[mask],
                    pd.to_numeric(rows[column], errors="coerce").to_numpy()[mask],
                )
            )
        report["positions"][position] = entries

    print(f"\nHOLDOUT {args.holdout}, DRAFTED POOL, VOLUME COMPONENTS\n")
    for position, entries in report["positions"].items():
        print(f"  {position}")
        print(
            f"    {'component':14s} {'n':>4s} {'projected':>10s} {'observed':>9s} "
            f"{'bias':>9s} {'bias %':>8s}"
        )
        for e in entries:
            print(
                f"    {e['component']:14s} {e['n']:>4d} {e['projected']:>10.2f} "
                f"{e['observed']:>9.2f} {e['bias']:>+9.2f} {e['bias_pct']:>+7.1%}"
            )
        print()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
