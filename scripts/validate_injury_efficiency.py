"""Walk-forward screen for injury covariates in the efficiency layer.

The last layer injury has never been tested in. Every efficiency response reads
prior_availability, prior_snap_share, age, experience, team_change and
cold_start, and no reserve or injury flag.

Expectations should be low and are worth stating before the run rather than
after. The descriptive screen put every injury-efficiency correlation under
|r| = 0.06, with the sign against recurrence *positive* -- yards per carry
+0.053, yards per target +0.037 -- which reads as survivorship rather than
support, since a player who keeps getting hurt and keeps playing is good enough
to be kept. Only last season's absence gave a mild negative on receiving,
-0.056 yards per target and -0.048 catch rate. For comparison, recurrence
against carry role was -0.141 after controls and still failed its holdout.

So this is run to close the question rather than in expectation of a win, and a
null here means the layer is genuinely uninformed by injury rather than that
nobody looked.

**That expectation was wrong.** On 2023/2024/2025 the reserve flag clears the
gate on rec_yards_per_target, and it is the only arm in this whole line of work
that does:

    overall (n~411)     MAE -0.37%   CRPS -0.26%   3/3 folds
    reserve  (n~50)     MAE -2.48%   CRPS -1.85%   3/3 folds
    recurrent (n~214)   MAE -0.38%   CRPS -0.31%   3/3 folds

Material against the 0.25% floor and winning every fold, with the effect
concentrated where it should be: a reserve player's yards per target is
predicted 2.5% better when the model knows he was on a reserve list.

Two things keep this modest rather than a headline. The overall CRPS gain of
0.26% sits barely over the floor, so MAE is carrying the claim. And recurrence
adds nothing here -- -0.06% at two folds of three -- which matches the
descriptive sign being positive and survivorship-shaped, and means ``both`` is
just the reserve flag with a passenger.

The scope is also one response. rec_catch_rate cannot take covariates at all and
rushing efficiency was descriptively flat, so this is not "injury helps
efficiency" but "the reserve flag helps yards per target".

Restricted to the receiving responses. That is where the descriptive sign was
negative at all; rushing efficiency was flat against every injury marker, so an
arm on it would be testing a hypothesis the data never suggested.

    python scripts/validate_injury_efficiency.py --holdouts 2023
    python scripts/validate_injury_efficiency.py --merge a.json b.json c.json
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

from ffmodel.evaluation.metrics import empirical_crps
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    PosteriorSeasonEfficiencyModel,
)

# Only the responses whose design actually admits covariates. rec_catch_rate
# fits with a single term -- its mean mode returns an empty design on purpose,
# the exposure weight having already entered through the prior -- so every arm
# on it returns byte-identical numbers. Scoring it would report four arms
# agreeing as evidence of a null when it is evidence of nothing at all.
RESPONSES = ("rec_yards_per_target",)

ARMS = {
    "baseline": (),
    "reserve": ("roster_reserve",),
    "recurrence": ("prior_injury_episode_count_3yr",),
    "both": ("roster_reserve", "prior_injury_episode_count_3yr"),
}

MATERIAL = 0.0025


def _metrics(observed, samples):
    mean = samples.mean(axis=1)
    return {
        "mae": float(np.abs(observed - mean).mean()),
        "rmse": float(np.sqrt(np.mean((observed - mean) ** 2))),
        "crps": float(empirical_crps(observed, samples).mean()),
        "n": int(len(observed)),
    }


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name in ("roster_reserve", "prior_injury_episode_count_3yr"):
        out[name] = pd.to_numeric(
            out.get(name, pd.Series(0.0, index=out.index)), errors="coerce"
        ).fillna(0.0)
    return out


def _evaluate(train, test, response, features, *, fit_kwargs, seed):
    spec = EFFICIENCY_MODEL_BY_TARGET[response]
    model = PosteriorSeasonEfficiencyModel(spec=spec, extra_features=tuple(features))
    model.fit(train, **fit_kwargs)
    prediction = model.predict_samples(test, seed=seed)
    # EfficiencyRatePrediction carries the future-season draws on ``rate``;
    # ``mean`` is the posterior location and would score the fit, not the
    # forecast.
    rows = prediction.rows
    samples = np.asarray(prediction.rate, dtype=float)
    observed = pd.to_numeric(rows[response], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(observed) & np.isfinite(samples).all(axis=1)
    observed, samples = observed[keep], samples[keep]
    out = {"overall": _metrics(observed, samples), "features": len(model.feature_names)}

    reserve = pd.to_numeric(
        rows.get("roster_reserve"), errors="coerce"
    ).fillna(0).gt(0).to_numpy()[keep]
    if reserve.sum() >= 5:
        out["reserve"] = _metrics(observed[reserve], samples[reserve])
    episodes = pd.to_numeric(
        rows.get("prior_injury_episode_count_3yr"), errors="coerce"
    ).fillna(0).to_numpy()[keep]
    if (episodes >= 2).sum() >= 5:
        out["recurrent"] = _metrics(observed[episodes >= 2], samples[episodes >= 2])
    del model, prediction, samples
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = [h for h in args.holdouts if str(h) in folds]
    for response in RESPONSES:
        print(f"\n{'=' * 78}\n{response} vs baseline\n{'=' * 78}")
        for population in ("overall", "reserve", "recurrent"):
            rows = []
            for arm in ARMS:
                values = [
                    folds[str(h)][response][arm][population]
                    for h in holdouts
                    if population in folds[str(h)][response][arm]
                ]
                if not values:
                    continue
                rows.append(
                    {
                        "arm": arm,
                        "n": int(np.mean([v["n"] for v in values])),
                        **{
                            m: float(np.mean([v[m] for v in values]))
                            for m in ("mae", "crps")
                        },
                    }
                )
            if not rows:
                continue
            table = pd.DataFrame(rows).set_index("arm")
            base = table.loc["baseline"]
            for metric in ("mae", "crps"):
                table[f"{metric}_delta"] = (table[metric] - base[metric]) / base[metric]
            for arm in ARMS:
                scored = [
                    h for h in holdouts if population in folds[str(h)][response][arm]
                ]
                wins = sum(
                    folds[str(h)][response][arm][population]["crps"]
                    < folds[str(h)][response]["baseline"][population]["crps"]
                    for h in scored
                )
                table.loc[arm, "crps_folds_won"] = f"{wins}/{len(scored)}"
            print(f"\n-- {population} (n~{int(table['n'].iloc[0])}) --")
            print(
                table[["mae", "crps", "mae_delta", "crps_delta", "crps_folds_won"]]
                .to_string(
                    float_format=lambda v: f"{v:.5f}" if abs(v) > 1e-3 else f"{v:+.2%}"
                )
            )
    print(f"\nmateriality floor {MATERIAL:.2%}; a smaller move is not a result")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    print(f"wrote {args.report_json}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/injury_efficiency.json")
    )
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        for path in args.merge:
            folds.update(json.loads(path.read_text("utf-8"))["folds"])
        args.holdouts = sorted(int(key) for key in folds)
        return _report({"holdouts": args.holdouts, "folds": folds}, args)

    player_rows = _prepare(pd.read_pickle(args.cache_dir / "player_rows.pkl"))
    fit_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    report: dict[str, object] = {"holdouts": args.holdouts, "folds": {}}
    for holdout in args.holdouts:
        train = player_rows[player_rows.season.lt(holdout)].copy()
        test = player_rows[player_rows.season.eq(holdout)].copy()
        fold: dict = {}
        for response in RESPONSES:
            fold[response] = {}
            for arm, features in ARMS.items():
                fold[response][arm] = _evaluate(
                    train, test, response, features,
                    fit_kwargs=fit_kwargs, seed=args.seed,
                )
                block = fold[response][arm]["overall"]
                print(
                    f"{holdout} {response:22s} {arm:11s} CRPS {block['crps']:.6f}  "
                    f"MAE {block['mae']:.6f}  feat {fold[response][arm]['features']}",
                    flush=True,
                )
        report["folds"][str(holdout)] = fold
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
