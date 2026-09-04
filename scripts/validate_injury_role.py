"""Walk-forward screen for reserve status in the role allocation layer.

The share models read prior_availability, prior_snap_share, age, experience,
team_change and cold_start. None of those says *why* a player's playing time was
short, and the descriptive screen in ``screen_injury_role_efficiency.py`` says
the reason matters: a week-1 reserve player takes 26% fewer carries and 17%
fewer targets per game than his own prior role implies, against a healthy
population, on 2015-2025.

That is a residual gap, not a model failure, and the difference is the reason
this script exists. The layer is fitted, so it may already absorb the gap
through ``prior_availability``, which is correlated with being hurt. Only a
holdout can say whether the flag adds anything the fit does not already have.

**Playing time is held fixed across arms.** Both arms are given the observed
snap share rather than a predicted one, so neither is rewarded or punished for
the availability layer's accuracy and the only difference between them is
whether the allocation knows the player was on a reserve list. Letting
availability vary would fold two questions into one number.

Arms:

``baseline``        today's covariates
``reserve``         the bare flag, kept as the rejected reference
``reserve-x-role``  the flag plus its interaction with holding a real role
``recurrence``      prior injury episode count
``both``            the corrected flag and recurrence together

Scored overall and on the reserve population, because a flag that fires on 4% of
rows cannot move a pooled average even when it is right about those rows.

**Result on 2023/2024/2025: nothing clears the gate, across four encodings.**

    carry, overall          reserve +0.81% MAE   reserve-x-role +0.81%
                            recurrence +2.76%    both +2.30%     0-1/3 folds
    carry, reserve_with_role (n~31)
                            reserve-x-role -1.66% MAE, -1.49% CRPS, 2/3 folds
    carry, recurrent_with_role (n~79)
                            recurrence +6.76% MAE, +6.27% CRPS, 0/3 folds

The only arm that helps anywhere is the flag interacted with holding a real
role, on the thirty-odd carry rows a season that are actually at risk: -1.66%
MAE, above the materiality floor, on two folds of three. The gate is every fold,
so it does not pass, and it is worse everywhere else including overall. On a
population that small, two of three is what a coin does.

**Recurrence failed hardest, and it was the arm with the best prior case.** Its
partial correlation with carry role was -0.141 after controlling snaps, age and
experience, and its population is large rather than diluted. In the holdout it
is worse than baseline everywhere and *worst on its own population*, 0 of 3
folds. A partial correlation that survives controls is not the same quantity as
predictive value over a fitted model that already carries a role prior, and this
is the cleanest demonstration of that gap in this file.

Taken with the earlier arms, four encodings of injury -- a bare flag, the flag
split by reserve kind, the flag interacted with role, and recurrence -- have now
failed at this layer. The descriptive gaps are real and repeatedly do not
survive, because the allocation already conditions on prior role and observed
playing time, which is most of what an injury changes. The remaining untested
place for this signal is upstream in the snap model, not here.

Folds run one per process; twelve fits in one process exhausts this container.

    python scripts/validate_injury_role.py --holdouts 2023 --report-json out.json
    python scripts/validate_injury_role.py --merge out1.json out2.json out3.json
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

from ffmodel.evaluation.metrics import point_and_distribution
from ffmodel.models.season_availability import RESERVE_KIND_FEATURES
from ffmodel.models.volume_season_average import SeasonRosterShareModel

ARMS = {
    "baseline": (),
    # The bare flag, kept as the rejected reference rather than deleted: the
    # corrected arms are only interpretable against what they correct.
    "reserve": ("roster_reserve",),
    # The flag restricted to players with something to lose. 68% of flagged rows
    # held under 2% of their team's work the season before, and among those the
    # flag costs nothing (+0.463 against +0.401 for healthy fringe players)
    # while among role-holders it costs 0.2 to 0.4 in log share. One coefficient
    # over both is what the bare arm fitted.
    "reserve-x-role": ("roster_reserve", "reserve_holds_role"),
    # Recurrence, which survived the controls that killed the flag: partial
    # correlation with carry role is -0.162 raw and -0.141 after snaps, age and
    # experience together. It also avoids the dilution by construction, because
    # episode count rises with role rather than falling with it.
    "recurrence": ("prior_injury_episode_count_3yr",),
    "both": (
        "roster_reserve",
        "reserve_holds_role",
        "prior_injury_episode_count_3yr",
    ),
}

STREAMS = {"carry": "carry_share", "target": "target_share"}

# Where a reserve flag starts to mean "a role is at risk" rather than "this is a
# fringe roster body". Read off the banding in
# ``scripts/screen_reserve_flag_dilution.py``: the sign of the reserve effect
# turns between the 2-8% and 8-20% bands.
ROLE_THRESHOLD = 0.08


def _derive(frame: pd.DataFrame, stream: str) -> pd.DataFrame:
    """Add the interaction, per stream, without touching the cache.

    Stream-specific on purpose: a back's carry role and a receiver's target role
    are different things to have at risk, and pooling them would put a receiver's
    target share in a carry regression.
    """
    out = frame.copy()
    prior = pd.to_numeric(
        out.get(f"prior_{stream}_share", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    reserve = pd.to_numeric(
        out.get("roster_reserve", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    out["reserve_holds_role"] = reserve * prior.ge(ROLE_THRESHOLD).astype(float)
    out["prior_injury_episode_count_3yr"] = pd.to_numeric(
        out.get("prior_injury_episode_count_3yr", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    return out

MATERIAL = 0.0025


def _evaluate(train, test, stream, features, *, fit_kwargs, seed):
    model = SeasonRosterShareModel(stream=stream, extra_features=tuple(features))
    model.fit(train, **fit_kwargs)
    # The exposure has to be one column per *posterior* draw, which is chains
    # times draws and not the --draws argument.
    sizes = model.idata.posterior.sizes
    draws = int(sizes["chain"]) * int(sizes["draw"])
    # Hand the allocation the playing time that actually happened, so the arms
    # differ only in whether they know the reason for it.
    prepared = model._design(test)["rows"]
    snaps = pd.to_numeric(
        prepared.get("snap_share", pd.Series(np.nan, index=prepared.index)),
        errors="coerce",
    ).fillna(
        pd.to_numeric(prepared.get("observed_availability"), errors="coerce")
    ).fillna(0.5).clip(1e-5, 1.0).to_numpy(dtype=float)
    prediction = model.predict_share_samples(
        test, snap_samples=np.repeat(snaps[:, None], draws, axis=1), seed=seed
    )
    rows = prediction.rows
    observed = pd.to_numeric(
        rows[STREAMS[stream]], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    samples = prediction.shares

    out = {
        "overall": point_and_distribution(observed, samples),
        "features": len(model.feature_names),
    }
    reserve = pd.to_numeric(
        rows.get("roster_reserve"), errors="coerce"
    ).fillna(0).gt(0).to_numpy()
    if reserve.sum() >= 5:
        out["reserve"] = point_and_distribution(observed[reserve], samples[reserve])
    # The population the screen measured the gap on: someone who actually holds
    # a role, rather than the long tail of zero-share roster filler that
    # dominates a pooled average and moves for other reasons.
    holds_role = reserve & (observed > 0.02)
    if holds_role.sum() >= 5:
        out["reserve_with_role"] = point_and_distribution(
            observed[holds_role], samples[holds_role]
        )
    # The recurrence population, which is a different and much larger set than
    # the reserve one: episode count rises with role, so this is not the thin
    # slice the bare flag was diluted into.
    episodes = pd.to_numeric(
        rows.get("prior_injury_episode_count_3yr"), errors="coerce"
    ).fillna(0).to_numpy()
    recurrent = (episodes >= 2) & (observed > 0.02)
    if recurrent.sum() >= 5:
        out["recurrent_with_role"] = point_and_distribution(
            observed[recurrent], samples[recurrent]
        )
    del model, prediction, samples
    gc.collect()
    return out


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = [h for h in args.holdouts if str(h) in folds]
    for stream in STREAMS:
        print(f"\n{'=' * 78}\n{stream} share vs baseline\n{'=' * 78}")
        for population in ("overall", "reserve", "reserve_with_role", "recurrent_with_role"):
            rows = []
            for arm in ARMS:
                values = [
                    folds[str(h)][stream][arm][population]
                    for h in holdouts
                    if population in folds[str(h)][stream][arm]
                ]
                if not values:
                    continue
                rows.append(
                    {
                        "arm": arm,
                        "n": int(np.mean([v["n"] for v in values])),
                        **{
                            m: float(np.mean([v[m] for v in values]))
                            for m in ("mae", "crps", "coverage_80")
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
                scored = [h for h in holdouts if population in folds[str(h)][stream][arm]]
                wins = sum(
                    folds[str(h)][stream][arm][population]["crps"]
                    < folds[str(h)][stream]["baseline"][population]["crps"]
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
    parser.add_argument("--report-json", type=Path, default=Path("reports/injury_role.json"))
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        for path in args.merge:
            folds.update(json.loads(path.read_text("utf-8"))["folds"])
        args.holdouts = sorted(int(key) for key in folds)
        return _report({"holdouts": args.holdouts, "folds": folds}, args)

    player_rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    missing = [c for c in RESERVE_KIND_FEATURES if c not in player_rows]
    if missing:
        raise SystemExit(
            f"{args.cache_dir} is missing {missing}; every arm would score as "
            "the baseline. Rebuild with scripts/build_projection_cache.py"
        )

    fit_kwargs = {"draws": args.draws, "tune": args.tune, "chains": args.chains}
    report: dict[str, object] = {"holdouts": args.holdouts, "folds": {}}
    for holdout in args.holdouts:
        train = player_rows[player_rows.season.lt(holdout)].copy()
        test = player_rows[player_rows.season.eq(holdout)].copy()
        if train.empty or test.empty:
            raise SystemExit(f"holdout {holdout} has no train or test rows")
        fold: dict = {}
        for stream in STREAMS:
            fold[stream] = {}
            # Derived per stream, and on both frames, so the fit and the score
            # see the same column rather than a train-only feature.
            train_s, test_s = _derive(train, stream), _derive(test, stream)
            for arm, features in ARMS.items():
                fold[stream][arm] = _evaluate(
                    train_s, test_s, stream, features,
                    fit_kwargs=fit_kwargs, seed=args.seed,
                )
                block = fold[stream][arm]["overall"]
                print(
                    f"{holdout} {stream:7s} {arm:14s} CRPS {block['crps']:.6f}  "
                    f"MAE {block['mae']:.6f}",
                    flush=True,
                )
        report["folds"][str(holdout)] = fold
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
