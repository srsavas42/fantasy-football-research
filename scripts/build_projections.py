"""Out-of-sample projections for every player-week, for the league agent to use.

The agent has been ranking players on exponentially weighted scoring and the
draft board. That is roughly the naive baseline the weekly model beats by 11.6%
CRPS, and the ablation said where the gap shows: the agent captures 47% of the
available waiver band but only 10% of the start/sit band. Start/sit is where
projection quality binds. This is what feeds the projection in.

Two horizons, because the agent makes two different decisions:

**Next week** answers the lineup question -- who scores most this Sunday. The
shipped hurdle, fitted per position with team, matchup, phase defence, game
script, ADP and the pre-game injury and depth feeds.

**Rest of season** answers the waiver question, and it is the one the agent had
no way to ask before. "Should I add this player" is not about Sunday; it is about
what he is worth for the remaining weeks, which is exactly the response
:func:`ffmodel.weekly.restofseason.add_rest_of_season_target` defines -- points
from week ``w`` to the end, with the player's club's remaining games as the
offset so a mid-season exit is handled honestly.

The rest-of-season projection is the **blend** the weekly document ships: the
direct total with phase and ADP, combined with the draft-board rank curve at a
weight estimated per horizon. The weight is the variance-optimal one -- the slope
of ``observed - curve`` on ``model - curve`` -- and it has to be estimated on
predictions the model has not seen, so the most recent training season is held
out to measure it and both forecasts are then refitted on everything. Measured
weights run about 0.5 early and 1.0 late, which is the board earning its keep in
September and giving it back as the season supplies usage the board never saw.

The blend covers **drafted** players at the four skill positions, and only those.
The rank curve is fitted per position on the skill panel, and its weight is
estimated on drafted rows, because the board is the only thing it knows. An
undrafted player is placed at the deepest rank the curve ever saw -- a fallback,
not a forecast -- so blending it in would replace most of his projection with a
replacement-level constant. That is not hypothetical: applied to everybody it
costs 12.9% MAE overall and 24.4% early, while gaining 1.9% on the population it
was fitted for. Undrafted players are the waiver wire, which is the decision this
projection exists to serve. Kickers and defenses keep the unblended specialist
total.

Kickers and defenses get the same two horizons from the specialist ladders, so
all six startable positions project on one scale.

**Every projection is made without the week it projects, and without the season
it lands in.** Each holdout season is predicted by a model fitted on strictly
earlier seasons, the same walk-forward the weekly work is validated under. That
costs the first two seasons -- a fit needs two prior ones -- so projections begin
in 2018 and the agent's training window shortens accordingly. Fitting once on
everything would be faster, much better looking, and worthless: the agent would
be reading projections that had seen the games they project.

    python scripts/build_projections.py --seasons 2018 2019 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.weekly import FEATURES_CACHE
from ffmodel.weekly.market import (
    HORIZON_BUCKETS,
    WeeklyRankCurve,
    bucket_labels,
    fit_blend_weights,
)
from ffmodel.weekly.nextweek import Hurdle
from ffmodel.weekly.restofseason import OFFSET, TARGET, DirectTotal, add_rest_of_season_target
from ffmodel.weekly.specialists import (
    DEFENSE_HISTORY_FEATURES,
    KICKER_HISTORY_FEATURES,
    SpecialistDirectTotal,
    SpecialistModel,
)

CACHE = Path(".cache/league_projections.parquet")
MIN_TRAIN_SEASONS = 2

# The shipped next-week configuration, as `docs/weekly-modeling-2026-08.md`
# describes it and `scripts/validate_weekly_baselines.py` scores it.
SHIPPED_WEEKLY = dict(
    use_team=True, use_matchup=True, use_phase=True, use_script=True,
    use_adp=True, use_news=True, use_snaps=True, use_recent=True,
    use_pedigree=True, use_charting=True, by_position=True,
)


# The quantiles carried alongside the mean. A projection's *spread* is a
# different fact from its level and the two decisions want it differently: a
# lineup behind late wants the p90, a lineup ahead wants the p10, and a waiver
# claim on a player nobody has seen play is a bet on the top of his range.
QUANTILES = (10, 50, 90)


def _summary(estimator, frame: pd.DataFrame, draws: int, seed: int) -> dict:
    """Mean and quantiles from one pass of the sampler.

    Drawn once and summarised, rather than sampled separately per statistic, so
    the mean and the quantiles describe the same predictive distribution rather
    than three neighbouring ones.
    """
    samples = np.asarray(estimator.predict_samples(frame, draws, seed), float)
    out = {"mean": samples.mean(axis=1)}
    for q, values in zip(QUANTILES, np.percentile(samples, QUANTILES, axis=1)):
        out[f"p{q}"] = values
    return out


def _mean(estimator, frame: pd.DataFrame, draws: int, seed: int) -> np.ndarray:
    """The predictive mean, from the same sampler the evaluation scores."""
    return _summary(estimator, frame, draws, seed)["mean"]


def _fit_predict(make, train, test, target, draws, seed, label):
    """One walk-forward fit, or nothing if it cannot be made honestly.

    Returns the full summary -- mean plus quantiles -- or ``None``.
    """
    values = pd.to_numeric(train[target], errors="coerce")
    usable = np.isfinite(values)
    if usable.sum() < 50 or train["season"].nunique() < MIN_TRAIN_SEASONS:
        return None
    try:
        model = make().fit(train[usable], values[usable].to_numpy(float))
        return _summary(model, test, draws, seed)
    except Exception as error:  # noqa: BLE001 -- reported, not swallowed
        print(f"    !! {label} failed: {type(error).__name__}: {error}")
        return None


def _blended_rest(build_model, train, test, draws, seed):
    """The rest-of-season total, blended with the draft-board rank curve.

    Returns ``(values, weights)``. Falls back to the unblended model when the
    curve cannot be fitted -- too few drafted player-seasons, or a panel with no
    board at all -- because a blend with a curve that does not exist is just the
    model, and saying so is better than failing.
    """
    try:
        weights = fit_blend_weights(
            train, build_model, TARGET, draws=draws, seed=seed
        )
        curve = WeeklyRankCurve(per_game=False, offset=OFFSET).fit(
            train, train["points"].to_numpy(float)
        )
    except (ValueError, KeyError) as error:
        print(f"    .. no rank curve ({type(error).__name__}); leaving it unblended")
        return None, {}

    model = build_model().fit(train, train[TARGET].to_numpy(float))
    model_summary = _summary(model, test, draws, seed)
    curve_summary = _summary(curve, test, draws, seed)

    labels = bucket_labels(test["week"].to_numpy(float))
    weight = np.ones(len(test), float)
    for name, _, _ in HORIZON_BUCKETS:
        weight[labels == name] = weights.get(name, 1.0)

    # **Only where the board has something to say.** The curve is a statement
    # about a draft board, and the blend weight is estimated on drafted players
    # alone; an unranked player is placed at the deepest rank the curve ever
    # saw, which is a fallback rather than a forecast. Blending that in at a
    # weight of 0.68 -- what the early-season weight implies -- replaces two
    # thirds of an undrafted player's projection with a replacement-level
    # constant.
    #
    # Measured across 2018-2025, doing it everywhere costs 12.9% MAE overall and
    # 24.4% early, while gaining 1.9% on the drafted population it was fitted
    # for. And undrafted players are precisely the waiver wire, which is the
    # decision this projection exists to serve.
    drafted = pd.to_numeric(test.get("adp_drafted"), errors="coerce").eq(1).to_numpy()
    weight = np.where(drafted, weight, 1.0)

    # curve + w * (model - curve): w = 1 is the model, w = 0 is the board.
    #
    # Applied to each statistic separately, which is a location shift of the
    # whole predictive distribution rather than a true quantile blend. The exact
    # object -- quantiles of the mixture of two predictive distributions -- is a
    # different and worse-behaved thing, and blending the summaries keeps the
    # quantiles consistent with the mean they are reported beside, which is what
    # a downstream feature needs.
    blended = {
        name: np.maximum(
            curve_summary[name] + weight * (model_summary[name] - curve_summary[name]),
            0.0,
        )
        for name in model_summary
    }
    return blended, weights


def _spread(block: pd.DataFrame, horizon: str, summary: dict | None) -> None:
    """Write one horizon's mean and quantiles into the output block."""
    stem = "projection" if horizon == "week" else "ros_projection"
    names = {"mean": stem, **{f"p{q}": f"{stem}_p{q}" for q in QUANTILES}}
    for key, column in names.items():
        block[column] = None if summary is None else summary[key]


def project_panel(
    panel: pd.DataFrame, seasons, *, weekly, rest, draws: int, label: str,
    blend: bool = True,
) -> pd.DataFrame:
    """Walk-forward projections for one panel, both horizons."""
    frame = add_rest_of_season_target(panel)
    rows = []
    for season in seasons:
        train = frame[frame["season"] < season]
        test = frame[frame["season"] == season]
        if test.empty or train["season"].nunique() < MIN_TRAIN_SEASONS:
            print(f"  {label} {season}: skipped, only "
                  f"{train['season'].nunique()} prior season(s)")
            continue
        started = time.time()
        block = test[["season", "week", "player_key"]].copy()
        _spread(
            block,
            "week",
            _fit_predict(
                weekly, train, test, "points", draws, season,
                f"{label} next-week {season}",
            ),
        )
        blended, weights = (None, {})
        if blend:
            usable = np.isfinite(pd.to_numeric(train[TARGET], errors="coerce"))
            blended, weights = _blended_rest(
                rest, train[usable], test, draws, season
            )
        if blended is not None:
            print(f"    blend weights {', '.join(f'{k} {v:.2f}' for k, v in weights.items())}")
        else:
            blended = _fit_predict(
                rest, train, test, TARGET, draws, season, f"{label} ROS {season}"
            )
        _spread(block, "ros", blended)
        block["games_remaining"] = pd.to_numeric(
            test[OFFSET], errors="coerce"
        ).to_numpy(float)
        rows.append(block)
        print(
            f"  {label} {season}: {len(block)} rows in {time.time() - started:.0f}s "
            f"(trained on {train['season'].nunique()} seasons)"
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", type=int, nargs="+",
        default=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    )
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--output", type=Path, default=CACHE)
    parser.add_argument("--skip-specialists", action="store_true")
    parser.add_argument(
        "--no-blend", action="store_true",
        help="use the bare direct total for rest of season instead of blending "
             "it with the draft-board rank curve",
    )
    args = parser.parse_args(argv)

    frames = []

    skill = pd.read_pickle(FEATURES_CACHE)
    print(f"skill panel {len(skill)} rows")
    frames.append(
        project_panel(
            skill, args.seasons,
            weekly=lambda: Hurdle(name="shipped", **SHIPPED_WEEKLY),
            rest=lambda: DirectTotal(
                name="direct-total+phase+adp",
                use_team=True, use_phase=True, use_adp=True,
            ),
            draws=args.draws, label="skill", blend=not args.no_blend,
        )
    )

    if not args.skip_specialists:
        for path, label, history in (
            (".cache/weekly_kickers_2016_2025.pkl", "kicker", KICKER_HISTORY_FEATURES),
            (".cache/weekly_defenses_2016_2025.pkl", "defense", DEFENSE_HISTORY_FEATURES),
        ):
            panel = pd.read_pickle(path)
            print(f"{label} panel {len(panel)} rows")
            hurdle = label == "kicker"
            frames.append(
                project_panel(
                    panel, args.seasons,
                    weekly=lambda h=history, u=hurdle: SpecialistModel(
                        name="shipped", history=h, use_market=True, use_hurdle=u
                    ),
                    rest=lambda h=history: SpecialistDirectTotal(
                        name="direct-total", history=h
                    ),
                    draws=args.draws, label=label, blend=False,
                )
            )

    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out = out.drop_duplicates(["season", "week", "player_key"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    print(f"\nwrote {args.output}: {len(out)} player-weeks")
    print(out.groupby("season")[["projection", "ros_projection"]].agg(
        ["count", "mean"]
    ).round(2).to_string())

    # A projection that does not track what happened is a bug rather than a weak
    # model, so it is worth seeing before anything is trained on it. Each horizon
    # is checked against *its own* response: next week against that week's
    # points, rest of season against the remaining total.
    panels = [skill] + ([] if args.skip_specialists else [
        pd.read_pickle(".cache/weekly_kickers_2016_2025.pkl"),
        pd.read_pickle(".cache/weekly_defenses_2016_2025.pkl"),
    ])
    truth = add_rest_of_season_target(pd.concat(panels, ignore_index=True))
    merged = out.merge(
        truth[["season", "week", "player_key", "position", "points", TARGET]],
        on=["season", "week", "player_key"], how="inner",
    )
    print("\nagainst what actually happened:")
    for column, response in (("projection", "points"), ("ros_projection", TARGET)):
        good = merged[merged[column].notna() & merged[response].notna()]
        if not len(good):
            continue
        # Restricted to players a lineup decision is actually between: the panel
        # is mostly deep bench, and a correlation dominated by players nobody
        # would start says nothing about the decision being made.
        live = good[good[response] > 0]
        # Pearson on ranks rather than `method="spearman"`, which reaches for
        # scipy; this environment does not have it and the definition is the
        # same.
        rho = (
            live.groupby("position")
            .apply(lambda g: g[column].rank().corr(g[response].rank()))
            .mean()
        )
        print(
            f"  {column:16s} r = {good[column].corr(good[response]):+.3f}"
            f"   within-position rho = {rho:+.3f}   ({len(good)} rows)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
