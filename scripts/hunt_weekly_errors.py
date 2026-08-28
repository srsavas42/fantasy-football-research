"""Error sources beyond role change and absence.

Those two account for roughly 56% of absolute error and both have a named cause.
This looks for what is left, in four places where a weekly fantasy model is
known to go wrong for reasons that have nothing to do with role:

**Touchdown dependence.** Points from touchdowns are lumpy and largely
unrepeatable at the player-week level, while yards and receptions are not. A
player whose recent scoring came disproportionately from the end zone has an
inflated history, and a model averaging his points inherits the inflation. If
this is real, bias rises with the touchdown share of his recent points -- a
regression effect that a yards-and-targets model would not have.

**Calibration across the projection range.** A model can be unbiased pooled and
wrong at both ends, over-projecting the players it likes most and under-
projecting the ones it likes least. That is the shrinkage signature, it is
invisible in a pooled bias, and it matters more than the pooled number because
lineup decisions are made at the top of the range.

**Game-script surprise.** The closing line is in the model, so the question is
not whether script matters but what happens when the market is *wrong*: a game
expected to be close that turns into a blowout rewrites both teams' second half.
This is unforecastable by construction and the point is to size it, since it puts
a floor under how well any weekly model can do.

**Position and archetype.** Where the remaining error concentrates once the
above are accounted for.

The first, third and fourth segments are defined using the outcome, which is
correct for attribution and would be leakage in a feature. Nothing here is a
feature.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.metrics import empirical_crps
from ffmodel.weekly.features import relevant_population
from ffmodel.weekly.nextweek import Hurdle


def _shipped() -> Hurdle:
    return Hurdle(
        use_team=True, use_matchup=True, use_phase=True, use_script=True,
        use_adp=True, use_news=True, use_snaps=True, by_position=True,
    )


def touchdown_share(frame: pd.DataFrame) -> pd.Series:
    """Share of a player's *recent* points that came from touchdowns.

    Built from the lagged points average against a lagged touchdown rate, so the
    segment describes what his history looked like going in rather than what he
    did in the week being scored.
    """
    scored = (
        pd.to_numeric(frame["rec_td"], errors="coerce").fillna(0.0)
        + pd.to_numeric(frame["rush_td"], errors="coerce").fillna(0.0)
    )
    work = pd.DataFrame(
        {
            "player_key": frame["player_key"].to_numpy(),
            "season": frame["season"].to_numpy(),
            "week": frame["week"].to_numpy(),
            "td_points": 6.0 * scored.to_numpy(),
            "points": pd.to_numeric(frame["points"], errors="coerce")
            .fillna(0.0)
            .to_numpy(),
        }
    ).sort_values(["player_key", "season", "week"])
    grouped = work.groupby("player_key", sort=False)
    alpha = 1.0 - 0.5 ** (1.0 / 4.0)
    td = grouped["td_points"].apply(
        lambda s: s.ewm(alpha=alpha, adjust=True).mean().shift(1)
    ).droplevel(0)
    total = grouped["points"].apply(
        lambda s: s.ewm(alpha=alpha, adjust=True).mean().shift(1)
    ).droplevel(0)
    ratio = (td / total.replace(0.0, np.nan)).reindex(work.index)
    return pd.Series(
        ratio.to_numpy(), index=frame.index[np.argsort(np.argsort(work.index))]
    ).reindex(frame.index)


def actual_team_score(seasons) -> pd.DataFrame:
    """(season, week, team) -> points the club actually scored.

    Not to be confused with the panel's ``team_points``, which is the sum of its
    skill players' *fantasy* points. Comparing that against a Vegas total is
    comparing two different units, and doing so silently marked 99.5% of rows as
    "beat the implied total".
    """
    from ffmodel.data import ingest

    schedule = ingest.load_schedules(sorted({int(s) for s in seasons}))
    if schedule.empty or "home_score" not in schedule.columns:
        return pd.DataFrame(columns=["season", "week", "team", "team_score"])
    if "game_type" in schedule.columns:
        schedule = schedule[schedule["game_type"] == "REG"]
    frames = []
    for side, points in (("home_team", "home_score"), ("away_team", "away_score")):
        block = schedule[["season", "week", side, points]].rename(
            columns={side: "team", points: "team_score"}
        )
        frames.append(block)
    out = pd.concat(frames, ignore_index=True)
    out["team_score"] = pd.to_numeric(out["team_score"], errors="coerce")
    for column in ("season", "week"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna().astype({"season": int, "week": int})


def build_segments(frame: pd.DataFrame, projected: np.ndarray) -> dict:
    td = touchdown_share(frame)
    spread = pd.to_numeric(frame["spread"], errors="coerce")
    total = pd.to_numeric(frame["game_total"], errors="coerce")
    # What the game actually turned into, against what the market expected.
    scores = actual_team_score(frame["season"].unique())
    merged = frame[["season", "week", "team"]].merge(
        scores, on=["season", "week", "team"], how="left"
    )
    realised = pd.Series(merged["team_score"].to_numpy(), index=frame.index)
    surprise = realised - pd.to_numeric(frame["implied_team_total"], errors="coerce")
    rank = pd.Series(projected, index=frame.index).rank(pct=True)

    segments = {
        "all": pd.Series(True, index=frame.index),
        "TD share of history < 20%": td.lt(0.20),
        "TD share 20-40%": td.between(0.20, 0.40),
        "TD share 40-60%": td.between(0.40, 0.60),
        "TD share > 60%": td.gt(0.60),
        "projected bottom 20%": rank.le(0.20),
        "projected 20-60%": rank.between(0.20, 0.60),
        "projected 60-90%": rank.between(0.60, 0.90),
        "projected top 10%": rank.gt(0.90),
        "offence beat its implied total by 10+": surprise.ge(10.0),
        "offence missed its implied total by 10+": surprise.le(-10.0),
        "big favourite (spread <= -7)": spread.le(-7.0),
        "big underdog (spread >= +7)": spread.ge(7.0),
        "high total (>= 48)": total.ge(48.0),
        "low total (<= 41)": total.le(41.0),
    }
    for name in ("QB", "RB", "WR", "TE"):
        segments[f"position {name}"] = frame["position"].eq(name)
    return segments


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(".cache/weekly_features_news_2016_2025.pkl"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    frame = pd.read_pickle(args.features)
    blocks = []
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
        projected = samples.mean(axis=1)
        crps = empirical_crps(observed, samples)
        total_error = float(np.abs(projected - observed).sum())

        for name, mask in build_segments(test, projected).items():
            want = mask.fillna(False).to_numpy(bool)
            if want.sum() < 40:
                continue
            error = projected[want] - observed[want]
            blocks.append(
                {
                    "segment": name,
                    "n": int(want.sum()),
                    "share_of_abs_error": 100.0 * np.abs(error).sum() / total_error,
                    "observed": float(observed[want].mean()),
                    "projected": float(projected[want].mean()),
                    "bias": float(error.mean()),
                    "mae": float(np.abs(error).mean()),
                    "crps": float(crps[want].mean()),
                }
            )

    table = pd.DataFrame(blocks)
    pooled = (
        table.assign(
            **{
                c: table[c] * table["n"]
                for c in table.columns
                if c not in ("segment", "n")
            }
        )
        .groupby("segment", as_index=False)
        .sum()
    )
    for column in pooled.columns:
        if column not in ("segment", "n"):
            pooled[column] = pooled[column] / pooled["n"]
    pooled = pooled.sort_values("share_of_abs_error", ascending=False)

    print("=" * 96)
    print(pooled.round(3).to_string(index=False))
    print(
        "\nReading it: `bias` is projected minus observed, so positive is "
        "over-projection.\nA monotone trend down the TD-share rows is the "
        "regression effect; a bias that\nflips sign between the bottom and top "
        "projection deciles is shrinkage; the\nimplied-total rows size what no "
        "weekly model can forecast."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(pooled.to_dict("records"), indent=2, default=str), "utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
