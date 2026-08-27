"""Is a role change knowable before it happens?

The error attribution found the model seven points low in the week a player's
role steps up, and unbiased two weeks later. That is a lag, not a tuning problem:
the promotion enters the usage averages only as it is produced. The question this
answers is whether anything published *before* that game would have said so.

This is a ceiling measurement, deliberately run before any feature is added to
the model. Three things have to hold before the injury report and the depth chart
are worth wiring in, and each can fail on its own:

1. **The signal has to fire on the right rows.** A flag that marks 30% of the
   panel and catches 30% of promotions has told us nothing. Lift -- how much more
   often a promotion happens when the flag is up -- is the statistic, not
   coverage.
2. **The rows it fires on have to be where the error is.** A signal can be
   perfectly predictive of a role change that costs nothing.
3. **It has to survive being lagged.** The depth chart for week ``w`` is a
   scraped artefact placed against the next game. If the contemporaneous version
   is worth a lot and the strictly-lagged one nothing, the difference is probably
   revision rather than information, and the honest number is the lagged one.

The output is a lift table and a bias table, not a model.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly.features import add_features, relevant_population
from ffmodel.weekly.frame import load_panel
from ffmodel.weekly.market import attach_adp
from ffmodel.weekly.news import add_news_features
from ffmodel.weekly.nextweek import Hurdle


def _shipped() -> Hurdle:
    return Hurdle(
        use_team=True, use_matchup=True, use_phase=True,
        use_script=True, use_adp=True, by_position=True,
    )


def role_delta(frame: pd.DataFrame) -> pd.Series:
    """Realised share of the team's carries/targets minus the lagged share."""
    is_back = frame["position"].eq("RB").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        realised = np.where(
            is_back,
            np.divide(
                frame["rush_att"].to_numpy(float), frame["team_rush_att"].to_numpy(float),
                out=np.full(len(frame), np.nan),
                where=frame["team_rush_att"].to_numpy(float) > 0,
            ),
            np.divide(
                frame["targets"].to_numpy(float), frame["team_targets"].to_numpy(float),
                out=np.full(len(frame), np.nan),
                where=frame["team_targets"].to_numpy(float) > 0,
            ),
        )
    lagged = np.where(
        is_back,
        pd.to_numeric(frame["prior_rush_share_recent"], errors="coerce"),
        pd.to_numeric(frame["prior_target_share_recent"], errors="coerce"),
    )
    return pd.Series(realised - lagged, index=frame.index)


def signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    promoted = pd.to_numeric(frame["depth_promoted"], errors="coerce").fillna(0.0)
    promoted_lag = pd.to_numeric(frame["depth_promoted_lagged"], errors="coerce").fillna(0.0)
    rank = pd.to_numeric(frame["depth_rank"], errors="coerce")
    return {
        "someone ahead of him is out": frame["ahead_out"].eq(1.0),
        "someone ahead of him was out last week": frame["ahead_out_lagged"].eq(1.0),
        "promoted on the depth chart": promoted.ge(1.0),
        "promoted last week (lagged)": promoted_lag.ge(1.0),
        "listed first at his position": rank.eq(1.0),
        "he is questionable or worse": frame["inj_questionable_or_worse"].eq(1.0),
        "two or more out at his position": frame["position_group_out"].ge(2.0),
        "ahead-out AND he is healthy": frame["ahead_out"].eq(1.0)
        & frame["inj_questionable_or_worse"].eq(0.0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--features", type=Path, default=Path(".cache/weekly_features_2016_2025.pkl")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    frame = (
        pd.read_pickle(args.features)
        if args.features.exists()
        else add_features(attach_adp(load_panel(range(2016, 2026))))
    )
    frame = add_news_features(frame)

    coverage = {
        "injury report present": float(frame["inj_status"].gt(0).mean()),
        "depth rank present": float(frame["depth_rank"].notna().mean()),
    }
    print("feed coverage over the whole panel:")
    for name, value in coverage.items():
        print(f"  {name:28s} {value:.3f}")

    blocks, bias_rows = [], []
    for holdout in args.holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        if train.empty or test.empty:
            continue
        keep = relevant_population(test).to_numpy(bool) | pd.to_numeric(
            test["adp_drafted"], errors="coerce"
        ).eq(1).to_numpy()
        test = test[keep].reset_index(drop=True)

        model = _shipped().fit(train, train["points"].to_numpy(float))
        samples = model.predict_samples(test, draws=args.draws, seed=holdout)
        observed = test["points"].to_numpy(float)
        error = samples.mean(axis=1) - observed

        grew = (role_delta(test) >= args.threshold).to_numpy()
        base = float(grew.mean())
        for name, mask in signals(test).items():
            flag = mask.to_numpy(bool)
            if flag.sum() < 30:
                continue
            hit = float(grew[flag].mean())
            blocks.append(
                {
                    "signal": name,
                    "n_flagged": int(flag.sum()),
                    "pct_of_rows": 100.0 * flag.mean(),
                    "p_role_grew": hit,
                    "base_rate": base,
                    "lift": hit / base if base > 0 else np.nan,
                    "recall": float(flag[grew].mean()),
                    "model_bias": float(error[flag].mean()),
                    "model_mae": float(np.abs(error[flag]).mean()),
                    "observed": float(observed[flag].mean()),
                }
            )
        bias_rows.append(
            {
                "holdout": holdout,
                "role_grew_n": int(grew.sum()),
                "role_grew_bias": float(error[grew].mean()),
                "share_of_grew_flagged_by_ahead_out": float(
                    test["ahead_out"].eq(1.0).to_numpy()[grew].mean()
                ),
                "share_of_grew_flagged_by_any": float(
                    (
                        test["ahead_out"].eq(1.0)
                        | pd.to_numeric(test["depth_promoted"], errors="coerce")
                        .fillna(0)
                        .ge(1.0)
                    ).to_numpy()[grew].mean()
                ),
            }
        )

    table = pd.DataFrame(blocks)
    pooled = (
        table.assign(
            **{
                c: table[c] * table["n_flagged"]
                for c in table.columns
                if c not in ("signal", "n_flagged")
            }
        )
        .groupby("signal", as_index=False)
        .sum()
    )
    for column in pooled.columns:
        if column not in ("signal", "n_flagged"):
            pooled[column] = pooled[column] / pooled["n_flagged"]
    pooled = pooled.sort_values("lift", ascending=False)

    print(f"\n{'=' * 108}\ndoes the signal fire where the role changes?\n{'=' * 108}")
    print(
        pooled[
            [
                "signal", "n_flagged", "pct_of_rows", "p_role_grew", "base_rate",
                "lift", "recall", "observed", "model_bias", "model_mae",
            ]
        ].round(3).to_string(index=False)
    )

    print(f"\n{'=' * 60}\nhow much of the role growth is flagged at all?\n{'=' * 60}")
    print(pd.DataFrame(bias_rows).round(3).to_string(index=False))

    print(
        "\nReading it: `lift` is how much more often a role grows when the flag "
        "is up.\n`recall` is the share of role growths the flag catches. A signal "
        "needs both --\nhigh lift with 2% recall fixes almost nothing -- and it "
        "needs `model_bias` to be\nnegative on its rows, which is what says the "
        "error is actually there. Compare\neach contemporaneous signal against "
        "its lagged twin: if the lag destroys it, the\nsignal was revision, not "
        "information."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "coverage": coverage,
                    "signals": pooled.to_dict("records"),
                    "folds": bias_rows,
                },
                indent=2,
                default=str,
            ),
            "utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
