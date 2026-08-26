"""How much mean reversion does the efficiency layer actually apply?

Four of the ten scoring responses -- catch rate, receiving touchdown rate,
rushing touchdown rate and fumble-lost rate -- ship with
``POSTERIOR_MEAN_MODE`` set to ``"prior"``. In that mode
``PosteriorSeasonEfficiencyModel._prior_mean`` links the lagged feature,
re-adds the centre it just removed, and inverts the link, so the conditional
mean it hands the simulator *is* ``prior_<response>`` exactly. The Beta
distribution around it moves the spread, never the location.

So for those responses the whole of the pipeline's regression to the mean is
the empirical-Bayes step in ``features/season_efficiency.py``:

    shrunk = (numerator + K * pooled) / (denominator + K)

with ``K`` a constant chosen per response in ``EFFICIENCY_SPECS``. That gives
the lagged season an effective persistence of ``den / (den + K)`` -- a number
set by usage and a hand-picked pseudo-count, never by the year-over-year
regression it is standing in for.

This script measures the persistence the data asks for and compares it with the
persistence the shipped policy applies. Three quantities per response:

- **the slope the data wants**, from a walk-forward weighted regression of the
  realized season ``Y+1`` rate on the shipped ``shrunk`` feature. A shipped
  policy of "use the feature as the mean" is the claim that this slope is 1.
- **held-out error in touchdowns**, not in rate space, because rate error is
  only interesting once multiplied by opportunity. Scored at the player's
  realized season ``Y+1`` exposure so the volume layer is held out of it.
- **bias by quintile of the lagged feature**, which is where a missing slope
  shows up: too persistent means the top of the board is projected high and the
  bottom low, with the pooled average hiding both.

The challenger is deliberately the smallest possible one -- an intercept and a
slope on the feature the model already builds. It is not the efficiency-v2
posterior-regression arm, which added the whole covariate block at once and lost
0/3 folds on these responses; see docs/efficiency-v2-validation.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.data import load_player_weeks
from ffmodel.features import crossseason
from ffmodel.features.season_efficiency import (
    EFFICIENCY_BY_NAME,
    player_season_efficiency,
)
from ffmodel.features.volume import MODEL_POSITIONS, normalize_model_positions
from ffmodel.models.efficiency_season_average import (
    EFFICIENCY_MODEL_BY_TARGET,
    POSTERIOR_MEAN_MODE,
)

# Fantasy points per event, for reading a rate error as something a drafter
# would notice. Yardage responses are excluded from the points column because
# theirs is not a per-event constant.
POINTS_PER_EVENT = {
    "pass_td_rate": 4.0,
    "rec_td_rate": 6.0,
    "rush_td_rate": 6.0,
    "rec_catch_rate": 1.0,
}

# Responses whose conditional mean is a rate the simulator draws counts from.
MEASURED = (
    "rec_td_rate",
    "rush_td_rate",
    "pass_td_rate",
    "rec_catch_rate",
)

# Minimum opportunity in *both* seasons for a pair to be scored. This is a
# measurement floor, not the model's ``min_exposure``: a rate computed on five
# targets carries no information about the slope being estimated, and keeping
# such rows would attenuate every number here toward zero.
MIN_EXPOSURE = {
    "rec_td_rate": 40,
    "rush_td_rate": 60,
    "pass_td_rate": 150,
    "rec_catch_rate": 40,
    # Reported only in the contamination section below.
    "rec_yards_per_target": 40,
    "rush_yards_per_carry": 60,
}


def transitions(efficiency: pd.DataFrame, target: str) -> pd.DataFrame:
    """Pair each player-season's shipped lagged feature with what happened next.

    The feature column is the pipeline's own ``shrunk_<target>``, so this
    measures the quantity the model actually consumes rather than a
    reconstruction of it.
    """
    spec = EFFICIENCY_BY_NAME[target]
    model_spec = EFFICIENCY_MODEL_BY_TARGET[target]
    denominator = spec.denominator
    columns = [
        "season",
        "player_key",
        "position",
        f"shrunk_{target}",
        spec.numerator,
        denominator,
    ]
    if "availability" in efficiency:
        columns.append("availability")
    rows = efficiency[columns].copy()
    rows = rows[rows["position"].isin(model_spec.positions)]

    later = rows[["season", "player_key", spec.numerator, denominator]].copy()
    later["season"] -= 1
    later = later.rename(
        columns={spec.numerator: "numerator_next", denominator: "exposure_next"}
    )
    paired = rows.merge(later, on=["season", "player_key"], how="inner")

    floor = MIN_EXPOSURE[target]
    paired = paired[
        paired[denominator].ge(floor) & paired["exposure_next"].ge(floor)
    ].copy()
    paired["feature"] = pd.to_numeric(paired[f"shrunk_{target}"], errors="coerce")
    paired["rate_next"] = paired["numerator_next"] / paired["exposure_next"]
    paired["residual"] = paired["rate_next"] - paired["feature"]
    if "availability" not in paired:
        paired["availability"] = np.nan
    return paired.dropna(subset=["feature", "rate_next"]).reset_index(drop=True)


def weighted_line(feature: np.ndarray, response: np.ndarray, weight: np.ndarray):
    """Exposure-weighted intercept and slope."""
    scale = weight / weight.mean()
    design = np.column_stack([np.ones(len(feature)), feature])
    left = design.T @ (design * scale[:, None])
    right = design.T @ (response * scale)
    return np.linalg.solve(left, right)


def walk_forward(paired: pd.DataFrame, points: float | None) -> dict:
    """Score the shipped policy against a fitted slope, one fold per season.

    Each fold fits on transitions strictly earlier than the one it scores, so
    the challenger never sees its own holdout. The shipped arm needs no fit --
    it is the feature itself.
    """
    folds = []
    for season in sorted(paired["season"].unique()):
        train = paired[paired["season"] < season]
        test = paired[paired["season"] == season]
        if len(train) < 300 or len(test) < 30:
            continue
        line = weighted_line(
            train["feature"].to_numpy(float),
            train["rate_next"].to_numpy(float),
            train["exposure_next"].to_numpy(float),
        )
        exposure = test["exposure_next"].to_numpy(float)
        actual = test["rate_next"].to_numpy(float)
        shipped = test["feature"].to_numpy(float)
        fitted = line[0] + line[1] * shipped
        folds.append(
            {
                "season": int(season),
                "n": int(len(test)),
                "slope": float(line[1]),
                "shipped_mae": float(np.mean(np.abs((shipped - actual) * exposure))),
                "fitted_mae": float(np.mean(np.abs((fitted - actual) * exposure))),
                "shipped_bias": float(np.mean((shipped - actual) * exposure)),
            }
        )
    if not folds:
        return {}
    frame = pd.DataFrame(folds)
    total = int(frame["n"].sum())

    def pooled(column: str) -> float:
        return float((frame[column] * frame["n"]).sum() / total)

    shipped_mae = pooled("shipped_mae")
    fitted_mae = pooled("fitted_mae")
    out = {
        "folds": len(frame),
        "n": total,
        "fitted_slope": pooled("slope"),
        "shipped_event_mae": shipped_mae,
        "fitted_event_mae": fitted_mae,
        "delta_pct": 100.0 * (fitted_mae - shipped_mae) / shipped_mae,
        "folds_improved": int((frame["fitted_mae"] < frame["shipped_mae"]).sum()),
        "shipped_event_bias": pooled("shipped_bias"),
    }
    if points is not None:
        out["shipped_points_gap"] = points * (shipped_mae - fitted_mae)
    return out


def quintiles(paired: pd.DataFrame, points: float | None) -> pd.DataFrame:
    """Shipped bias by quintile of the lagged feature.

    Exposure is the realized next season's, so this isolates the rate policy:
    every row is asked only "given the opportunities he actually got, how many
    touchdowns does last year's shrunk rate predict".
    """
    out = paired.copy()
    out["quintile"] = pd.qcut(
        out["feature"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
    )
    out["projected"] = out["feature"] * out["exposure_next"]
    out["gap"] = out["projected"] - out["numerator_next"]
    table = out.groupby("quintile", observed=True).agg(
        n=("gap", "size"),
        lagged_rate=("feature", "mean"),
        realized_rate=("rate_next", "mean"),
        projected_events=("projected", "mean"),
        actual_events=("numerator_next", "mean"),
        event_gap=("gap", "mean"),
    )
    if points is not None:
        table["points_gap"] = table["event_gap"] * points
    return table


def season_games(weeks: pd.DataFrame) -> pd.DataFrame:
    """Weeks with a stat line, as a share of the season, per player-season.

    A proxy for availability rather than the roster truth: a player who is
    active but records nothing is missed. That understates a blocking tight
    end's games and is immaterial here, because the contamination question is
    asked only of players with 40+ opportunities in both seasons.
    """
    out = normalize_model_positions(weeks)
    out = out[out["position"].isin(MODEL_POSITIONS)].copy()
    if "player_id" not in out:
        out["player_id"] = pd.NA
    out["player_key"] = crossseason.player_key(out)
    games = (
        out.groupby(["season", "player_key"])["week"]
        .nunique()
        .rename("games")
        .reset_index()
    )
    games = games.join(out.groupby("season")["week"].max().rename("team_weeks"), on="season")
    games["availability"] = games["games"] / games["team_weeks"]
    return games


def contamination(efficiency: pd.DataFrame, target: str) -> pd.DataFrame:
    """Efficiency-prior residual by how much of season Y the player missed.

    If injury games corrupted the per-touch responses, the residual would trend
    with availability -- a hurt season's rate would be systematically low, and
    the prior built from it systematically low with it. A flat column is the
    evidence that the ratio-of-totals decomposition plus the exposure-weighted
    shrinkage already handles it.
    """
    paired = transitions(efficiency, target)
    paired = paired[paired["availability"].notna()]
    if paired.empty:
        return pd.DataFrame()
    paired["bucket"] = pd.cut(
        paired["availability"],
        [0.0, 0.55, 0.75, 0.90, 1.01],
        labels=["under 55%", "55-75%", "75-90%", "90%+"],
    )
    table = paired.groupby("bucket", observed=True).agg(
        n=("residual", "size"),
        mean_residual=("residual", "mean"),
        sd_residual=("residual", "std"),
    )
    table["bias_pct_of_mean"] = 100.0 * table["mean_residual"] / paired["rate_next"].mean()
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(1999, 2026)))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    cache = args.cache_dir / "season_efficiency.pkl"
    games_cache = args.cache_dir / "player_season_games.pkl"
    if cache.exists() and games_cache.exists():
        efficiency = pd.read_pickle(cache)
        games = pd.read_pickle(games_cache)
    else:
        weeks = load_player_weeks(args.seasons)
        efficiency = player_season_efficiency(weeks)
        games = season_games(weeks)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        efficiency.to_pickle(cache)
        games.to_pickle(games_cache)
    efficiency = efficiency.merge(
        games[["season", "player_key", "availability"]],
        on=["season", "player_key"],
        how="left",
    )

    summary = {}
    for target in MEASURED:
        paired = transitions(efficiency, target)
        points = POINTS_PER_EVENT.get(target)
        spec = EFFICIENCY_BY_NAME[target]
        result = walk_forward(paired, points)
        if not result:
            continue
        exposure = paired[spec.denominator].to_numpy(float)
        result["mean_mode"] = POSTERIOR_MEAN_MODE[target]
        result["shrinkage_pseudocount"] = spec.prior_opportunities
        result["implied_persistence"] = float(
            np.mean(exposure / (exposure + spec.prior_opportunities))
        )
        summary[target] = result

        print(f"\n=== {target}  (mean mode: {POSTERIOR_MEAN_MODE[target]}) ===")
        print(
            f"  shrinkage K = {spec.prior_opportunities:g} {spec.denominator}"
            f"  ->  effective persistence on the raw lagged rate "
            f"{result['implied_persistence']:.3f}"
        )
        asserted = (
            "shipped policy asserts 1.000"
            if POSTERIOR_MEAN_MODE[target] == "prior"
            else "reference only: this response's mean is a fitted ridge, so "
            "the shipped arm below is the lagged feature alone, not what ships"
        )
        print(
            f"  slope the data wants on the shipped feature: "
            f"{result['fitted_slope']:.3f}   ({asserted})"
        )
        print(
            f"  held-out event MAE  shipped {result['shipped_event_mae']:.3f}"
            f"  vs fitted {result['fitted_event_mae']:.3f}"
            f"   ({result['delta_pct']:+.2f}%, "
            f"{result['folds_improved']}/{result['folds']} folds)"
        )
        print(quintiles(paired, points).round(3).to_string())

    print("\n\n=== Do injury-shortened seasons corrupt the efficiency prior? ===")
    print("Residual of the realized season Y+1 rate around the shipped lagged")
    print("feature, split by how much of season Y the player was available for.")
    for target in ("rec_td_rate", "rush_td_rate", "rec_yards_per_target",
                   "rec_catch_rate", "rush_yards_per_carry"):
        if target not in EFFICIENCY_BY_NAME:
            continue
        table = contamination(efficiency, target)
        if table.empty:
            continue
        print(f"\n{target}")
        print(table.round(4).to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
