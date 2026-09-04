"""Does the Beta-Binomial's mean-variance link describe these responses?

Every proportion response in the layer uses `BetaBinomial(n, mu, c)` with a
single global concentration. That likelihood implies

    Var(latent rate) = mu * (1 - mu) / (c + 1)

so the latent variance is a *fixed fraction* of `mu(1-mu)`, the same fraction
for every player at every level of the mean. One global `c` is the assumption
that this fraction is constant. It is testable without fitting anything.

Bin on the lagged mean, and within each bin split the observed variance of the
realized rate into the part binomial sampling at each player's own exposure
would produce and the remainder:

    total  = Var(y / n)
    binom  = mean( mu(1-mu) / n )
    latent = total - binom

then back out the `c` that remainder implies. If the link holds, `c` is flat
across bins. If `c` moves by an order of magnitude, one global concentration
cannot be right anywhere except by accident.

A negative `latent` is not a bug in the arithmetic. It means the observed spread
is *smaller* than independent coin-flipping at that mean would give -- the
signature of a bin with no durable player-to-player difference left in it at
all, plus sampling noise in the estimate of the variance itself.

Rows whose numerator exceeds their own exposure are excluded and counted. They
are not a rate, and including them dominates the variance of whatever bin they
land in: seven such rows moved catch rate's lowest bin from `c = 66` to
`c = 4.5` before they were removed. See `tests/test_efficiency_label_scope.py`
for what produces them.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.models.efficiency_season_average import EFFICIENCY_MODEL_BY_TARGET

# Measurement floor, not the model's. A rate on five targets carries almost no
# information about the variance being estimated.
MIN_EXPOSURE = 20
BINS = 5

RESPONSES = ("rec_catch_rate", "rec_td_rate", "rush_td_rate")


def link_table(rows: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    out = rows[rows["position"].isin(spec.positions)].copy()
    for column in (spec.numerator, spec.exposure, spec.prior_feature):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[
        out[spec.exposure].ge(MIN_EXPOSURE)
        & out[spec.numerator].notna()
        & out[spec.prior_feature].notna()
    ]
    before = len(out)
    out = out[out[spec.numerator].le(out[spec.exposure])]
    dropped = before - len(out)

    out["rate"] = out[spec.numerator] / out[spec.exposure]
    out["bin"] = pd.qcut(out[spec.prior_feature], BINS, labels=False, duplicates="drop")

    records = []
    for index, block in out.groupby("bin", observed=True):
        exposure = block[spec.exposure].to_numpy(float)
        mu = float(block[spec.prior_feature].mean())
        total = float(block["rate"].var(ddof=1))
        binomial = float(np.mean(mu * (1.0 - mu) / exposure))
        latent = total - binomial
        records.append(
            {
                "bin": f"Q{int(index) + 1}",
                "n": int(len(block)),
                "mean_exposure": float(exposure.mean()),
                "mu": mu,
                "total_var": total,
                "binomial_part": binomial,
                "latent_var": latent,
                "latent_share_pct": 100.0 * latent / total if total > 0 else np.nan,
                "implied_c": (mu * (1.0 - mu) / latent - 1.0) if latent > 0 else np.inf,
            }
        )
    return pd.DataFrame(records).set_index("bin"), dropped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--player-rows", type=Path, default=Path(".cache/player_rows_2014_2025.pkl")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.player_rows.exists():
        raise SystemExit(f"no frame at {args.player_rows}")
    rows = pd.read_pickle(args.player_rows)

    results = {}
    for target in RESPONSES:
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        table, dropped = link_table(rows, target)
        finite = table["implied_c"][np.isfinite(table["implied_c"])]
        spread = float(finite.max() / finite.min()) if len(finite) > 1 else np.nan
        print(f"\n=== {target} ===")
        print(
            f"  model uses one global c, prior centred at "
            f"{spec.prior_concentration:g}; dropped {dropped} rows with "
            "numerator > exposure"
        )
        print(table.round(5).to_string())
        if np.isfinite(spread):
            print(
                f"  implied c spans {finite.min():.0f} to {finite.max():.0f}"
                f"  ->  {spread:.1f}x across the mean's range"
            )
        if (table["latent_var"] <= 0).any():
            bins = ", ".join(table.index[table["latent_var"] <= 0])
            print(
                f"  {bins}: observed spread is at or below independent binomial "
                "noise -- no durable player difference left in that bin"
            )
        results[target] = {
            "dropped_rows": dropped,
            "implied_c_spread": spread,
            "bins": table.reset_index().to_dict("records"),
        }

    print(
        "\nReading it: a flat implied c means one global concentration is fine. "
        "A\nlarge spread means the mean-variance link is doing the wrong thing, "
        "and the\ncheapest correction is a concentration that varies with the "
        "mean or the\nposition -- still a Beta-Binomial -- rather than a "
        "different likelihood family."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
