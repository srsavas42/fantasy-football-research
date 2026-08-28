"""Where does the weekly model's error actually come from, and why early?

The model beats a naive draft board comfortably from week 5 on and struggles in
weeks 1-4. "It has less data early" is true and useless -- it does not say which
players it gets wrong, in which direction, or whether the same failure is present
later and merely diluted. This attributes the error to segments, each defined
from what is knowable at decision time, and reports the signed bias as well as
the magnitude, because over- and under-prediction call for opposite fixes.

Four hypotheses, each given its own segmentation:

**History depth.** A player with no career weeks has no history features at all;
one with four has noisy ones. If this is the story, error concentrates on thin
history and disappears once a player has a season behind him.

**The board's low end.** If the model over-predicts players the market ranked
late or declined to rank, bias is positive and rises as ADP rank rises. This is
the classic failure of a history model in week 1: last year's part-time producer
is projected to repeat a role he has already lost.

**Absence and return.** Players coming back from missed time, and players who
miss time. The availability half is fitted on appearance history alone, with no
injury report, so a returning starter is the case it is least equipped for.

**Role inheritance -- the handcuff.** The one this design is structurally worst
at. Every usage feature is lagged, so a backup whose starter goes down carries a
backup's history into the week he inherits the job. The model cannot know, and
the question is how much that costs and how long it takes to correct.

Segments are diagnostic, not features. Several are defined using the outcome or
using facts the model does not read (whether the starter was active), which is
exactly the point: they measure what the model is blind to.
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
from ffmodel.weekly import FEATURES_CACHE, PANEL_CACHE
from ffmodel.weekly.features import add_features, relevant_population
from ffmodel.weekly.frame import load_panel
from ffmodel.weekly.market import WeeklyRankCurve, attach_adp
from ffmodel.weekly.nextweek import Hurdle

EARLY = 4


def _shipped() -> Hurdle:
    return Hurdle(
        use_team=True,
        use_matchup=True,
        use_phase=True,
        use_script=True,
        use_adp=True,
        by_position=True,
    )


def lead_back_out(frame: pd.DataFrame) -> pd.Series:
    """Was this team's established lead rusher absent this week?

    The lead is whoever carried the largest lagged share of the team's carries,
    so the label is formed from history rather than from the week being scored.
    Whether he then played is a fact about the current week -- known on a Sunday
    morning, but *not* read by the model -- which is what makes this a measure of
    what the model is blind to rather than a feature it failed to use.
    """
    backs = frame[frame["position"].eq("RB")].copy()
    backs["share"] = pd.to_numeric(
        backs["prior_rush_share_recent"], errors="coerce"
    ).fillna(0.0)
    order = backs.sort_values(
        ["season", "week", "team", "share"], ascending=[True, True, True, False]
    )
    lead = order.groupby(["season", "week", "team"], as_index=False).head(1)
    # A "lead" with no established share is not a lead; his absence tells us
    # nothing about anybody else's role.
    lead = lead[lead["share"] >= 0.35]
    lead = lead[["season", "week", "team", "player_key", "played"]].rename(
        columns={"player_key": "lead_key", "played": "lead_played"}
    )
    merged = frame[["season", "week", "team", "player_key"]].merge(
        lead, on=["season", "week", "team"], how="left"
    )
    out = (
        merged["lead_key"].notna()
        & merged["lead_played"].eq(0)
        & merged["player_key"].ne(merged["lead_key"])
        & frame["position"].eq("RB").to_numpy()
    )
    return pd.Series(out.to_numpy(), index=frame.index)


def previous_team(frame: pd.DataFrame) -> pd.Series:
    """The team he finished the previous season on, or NA."""
    last = (
        frame.sort_values(["player_key", "season", "week"])
        .groupby(["player_key", "season"], as_index=False)
        .last()[["player_key", "season", "team"]]
    )
    last["season"] = last["season"] + 1
    last = last.rename(columns={"team": "prev_team"})
    merged = frame[["player_key", "season", "team"]].merge(
        last, on=["player_key", "season"], how="left"
    )
    return pd.Series(merged["prev_team"].to_numpy(), index=frame.index)


def build_segments(frame: pd.DataFrame) -> dict[str, pd.Series]:
    prior_games = pd.to_numeric(frame["prior_games"], errors="coerce").fillna(0.0)
    rank = pd.to_numeric(frame["adp_rank"], errors="coerce")
    drafted = pd.to_numeric(frame["adp_drafted"], errors="coerce").eq(1)
    gap = pd.to_numeric(frame["weeks_since_played"], errors="coerce")
    prev = previous_team(frame)

    # Realised share against the lagged one: did the role actually change?
    with np.errstate(divide="ignore", invalid="ignore"):
        realised = np.where(
            frame["position"].eq("RB"),
            np.divide(
                frame["rush_att"].to_numpy(float),
                frame["team_rush_att"].to_numpy(float),
                out=np.full(len(frame), np.nan),
                where=frame["team_rush_att"].to_numpy(float) > 0,
            ),
            np.divide(
                frame["targets"].to_numpy(float),
                frame["team_targets"].to_numpy(float),
                out=np.full(len(frame), np.nan),
                where=frame["team_targets"].to_numpy(float) > 0,
            ),
        )
    lagged = np.where(
        frame["position"].eq("RB"),
        pd.to_numeric(frame["prior_rush_share_recent"], errors="coerce"),
        pd.to_numeric(frame["prior_target_share_recent"], errors="coerce"),
    )
    delta = pd.Series(realised - lagged, index=frame.index)

    return {
        "all": pd.Series(True, index=frame.index),
        "no career history": prior_games.eq(0),
        "thin history (1-7 games)": prior_games.between(1, 7),
        "some history (8-16)": prior_games.between(8, 16),
        "established (17+)": prior_games.ge(17),
        "changed team since last season": prev.notna() & prev.ne(frame["team"]),
        "undrafted by the board": ~drafted,
        "ADP 1-36": drafted & rank.le(36),
        "ADP 37-84": drafted & rank.between(37, 84),
        "ADP 85-150": drafted & rank.between(85, 150),
        "ADP 151+": drafted & rank.gt(150),
        "returning after 1 week out": gap.eq(2),
        "returning after 2+ weeks out": gap.ge(3),
        "did not play": frame["played"].eq(0),
        "handcuff: lead back out": lead_back_out(frame),
        "role grew (share +10pt)": delta.ge(0.10),
        "role shrank (share -10pt)": delta.le(-0.10),
    }


def summarize(
    frame: pd.DataFrame,
    observed: np.ndarray,
    model: np.ndarray,
    curve: np.ndarray,
    segments: dict[str, pd.Series],
) -> pd.DataFrame:
    model_mean = model.mean(axis=1)
    curve_mean = curve.mean(axis=1)
    model_crps = empirical_crps(observed, model)
    curve_crps = empirical_crps(observed, curve)
    total_error = float(np.abs(model_mean - observed).sum())

    rows = []
    for name, mask in segments.items():
        want = mask.to_numpy(bool)
        if want.sum() < 30:
            continue
        error = model_mean[want] - observed[want]
        rows.append(
            {
                "segment": name,
                "n": int(want.sum()),
                "share_of_rows": 100.0 * want.mean(),
                "share_of_abs_error": 100.0 * np.abs(error).sum() / total_error,
                "observed": float(observed[want].mean()),
                "projected": float(model_mean[want].mean()),
                "bias": float(error.mean()),
                "mae": float(np.abs(error).mean()),
                "crps": float(model_crps[want].mean()),
                "adp_bias": float((curve_mean[want] - observed[want]).mean()),
                "adp_crps": float(curve_crps[want].mean()),
            }
        )
    return pd.DataFrame(rows)


def role_change_profile(
    frame: pd.DataFrame, observed: np.ndarray, model: np.ndarray
) -> pd.DataFrame:
    """How long does the model take to price a role it did not see coming?

    Every usage feature is an exponentially weighted average of what a player has
    already done, so a promotion enters the features only as it is produced. This
    finds the week a player's role actually stepped up and follows the model's
    bias forward from it: the size of the miss in the first week is what the lag
    costs, and the rate it decays is how fast the average catches up.

    An event is the first week in a season where a player's realised share of his
    team's carries (backs) or targets (everyone else) exceeds his lagged share by
    ten points *and* reaches twenty percent, so a genuine promotion is separated
    from a one-week spike in a role he never held.
    """
    work = frame.copy().reset_index(drop=True)
    work["error"] = model.mean(axis=1) - observed
    is_back = work["position"].eq("RB").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        realised = np.where(
            is_back,
            np.divide(
                work["rush_att"].to_numpy(float), work["team_rush_att"].to_numpy(float),
                out=np.full(len(work), np.nan), where=work["team_rush_att"].to_numpy(float) > 0,
            ),
            np.divide(
                work["targets"].to_numpy(float), work["team_targets"].to_numpy(float),
                out=np.full(len(work), np.nan), where=work["team_targets"].to_numpy(float) > 0,
            ),
        )
    lagged = np.where(
        is_back,
        pd.to_numeric(work["prior_rush_share_recent"], errors="coerce"),
        pd.to_numeric(work["prior_target_share_recent"], errors="coerce"),
    )
    work["event"] = (realised - lagged >= 0.10) & (realised >= 0.20)

    rows = []
    for (_, _), block in work.groupby(["player_key", "season"], sort=False):
        block = block.sort_values("week")
        events = block.index[block["event"].to_numpy()]
        if not len(events):
            continue
        start = block.loc[events[0], "week"]
        after = block[block["week"] >= start]
        for offset, (_, row) in enumerate(after.iterrows()):
            if offset > 5:
                break
            rows.append({"weeks_since": offset, "error": row["error"],
                         "observed": row["points"]})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return (
        out.groupby("weeks_since")
        .agg(n=("error", "size"), bias=("error", "mean"),
             mae=("error", lambda s: s.abs().mean()),
             observed=("observed", "mean"))
        .reset_index()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--draws", type=int, default=600)
    parser.add_argument(
        "--features", type=Path, default=FEATURES_CACHE
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--population",
        choices=["relevant", "drafted-or-relevant"],
        default="relevant",
        help=(
            "'relevant' needs 4 prior appearances and so excludes every rookie "
            "by construction; 'drafted-or-relevant' adds anyone the board ranked, "
            "which is the only way to see players with no history at all"
        ),
    )
    args = parser.parse_args(argv)

    frame = (
        pd.read_pickle(args.features)
        if args.features.exists()
        else add_features(attach_adp(load_panel(range(2016, 2026))))
    )

    blocks = {"early (weeks 1-4)": [], "later (weeks 5+)": []}
    for holdout in args.holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        if train.empty or test.empty:
            continue
        keep = relevant_population(test).to_numpy(bool)
        if args.population == "drafted-or-relevant":
            keep = keep | pd.to_numeric(
                test["adp_drafted"], errors="coerce"
            ).eq(1).to_numpy()
        test = test[keep].reset_index(drop=True)
        target = train["points"].to_numpy(float)

        model = _shipped().fit(train, target)
        curve = WeeklyRankCurve(per_game=True).fit(train, target)
        model_samples = model.predict_samples(test, draws=args.draws, seed=holdout)
        curve_samples = curve.predict_samples(test, draws=args.draws, seed=holdout)

        observed = test["points"].to_numpy(float)
        week = pd.to_numeric(test["week"], errors="coerce").to_numpy(float)
        segments = build_segments(test)
        for label, want in (
            ("early (weeks 1-4)", week <= EARLY),
            ("later (weeks 5+)", week > EARLY),
        ):
            inside = {k: v[want] for k, v in segments.items()}
            blocks[label].append(
                summarize(
                    test[want],
                    observed[want],
                    model_samples[want],
                    curve_samples[want],
                    inside,
                )
            )

    profiles = []
    for holdout in args.holdouts:
        train = frame[frame["season"] < holdout]
        test = frame[frame["season"] == holdout]
        if train.empty or test.empty:
            continue
        keep = relevant_population(test).to_numpy(bool)
        if args.population == "drafted-or-relevant":
            keep = keep | pd.to_numeric(
                test["adp_drafted"], errors="coerce"
            ).eq(1).to_numpy()
        test = test[keep].reset_index(drop=True)
        model = _shipped().fit(train, train["points"].to_numpy(float))
        samples = model.predict_samples(test, draws=args.draws, seed=holdout)
        got = role_change_profile(test, test["points"].to_numpy(float), samples)
        if not got.empty:
            profiles.append(got)

    payload = {}
    for label, frames in blocks.items():
        if not frames:
            continue
        joined = pd.concat(frames)
        # Row-count weighted across folds.
        pooled = (
            joined.assign(**{
                c: joined[c] * joined["n"]
                for c in joined.columns
                if c not in ("segment", "n")
            })
            .groupby("segment", as_index=False)
            .sum()
        )
        for column in pooled.columns:
            if column not in ("segment", "n"):
                pooled[column] = pooled[column] / pooled["n"]
        pooled = pooled.sort_values("share_of_abs_error", ascending=False)
        payload[label] = pooled.to_dict("records")

        print(f"\n{'=' * 100}\n{label}\n{'=' * 100}")
        show = pooled[
            [
                "segment",
                "n",
                "share_of_abs_error",
                "observed",
                "projected",
                "bias",
                "mae",
                "crps",
                "adp_bias",
                "adp_crps",
            ]
        ]
        print(show.round(3).to_string(index=False))

    if profiles:
        joined = pd.concat(profiles)
        pooled = (
            joined.assign(**{c: joined[c] * joined["n"] for c in ("bias", "mae", "observed")})
            .groupby("weeks_since", as_index=False)
            .sum()
        )
        for column in ("bias", "mae", "observed"):
            pooled[column] = pooled[column] / pooled["n"]
        payload["role_change_profile"] = pooled.to_dict("records")
        print(f"\n{'=' * 100}\ncatching up to a role change\n{'=' * 100}")
        print(pooled.round(3).to_string(index=False))

    print(
        "\nReading it: `bias` is projected minus observed, so positive is "
        "over-projection.\n`share_of_abs_error` is what fraction of the total "
        "absolute error the segment\naccounts for -- a segment can have a large "
        "MAE and still not matter if it is\ntiny. `adp_bias` and `adp_crps` are "
        "the draft board on the same rows, so a\nsegment where the model is worse "
        "than the board is visible directly."
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
