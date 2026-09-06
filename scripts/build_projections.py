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
offset so a mid-season exit is handled honestly. The direct total with phase and
ADP, which is the documented shipped configuration.

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


def _mean(estimator, frame: pd.DataFrame, draws: int, seed: int) -> np.ndarray:
    """The predictive mean, from the same sampler the evaluation scores."""
    samples = estimator.predict_samples(frame, draws, seed)
    return np.asarray(samples, float).mean(axis=1)


def _fit_predict(make, train, test, target, draws, seed, label):
    """One walk-forward fit, or nothing if it cannot be made honestly."""
    values = pd.to_numeric(train[target], errors="coerce")
    usable = np.isfinite(values)
    if usable.sum() < 50 or train["season"].nunique() < MIN_TRAIN_SEASONS:
        return None
    try:
        model = make().fit(train[usable], values[usable].to_numpy(float))
        return _mean(model, test, draws, seed)
    except Exception as error:  # noqa: BLE001 -- reported, not swallowed
        print(f"    !! {label} failed: {type(error).__name__}: {error}")
        return None


def project_panel(
    panel: pd.DataFrame, seasons, *, weekly, rest, draws: int, label: str
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
        block["projection"] = _fit_predict(
            weekly, train, test, "points", draws, season, f"{label} next-week {season}"
        )
        block["ros_projection"] = _fit_predict(
            rest, train, test, TARGET, draws, season, f"{label} ROS {season}"
        )
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
            draws=args.draws, label="skill",
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
                    draws=args.draws, label=label,
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
