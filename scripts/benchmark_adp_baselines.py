"""Is 38% relative error good, bad, or just what this problem costs?

The ADP evaluation reported 38.3% MAE against realized points on the drafted
pool. A percentage with nothing to compare it to is not a result, so this scores
two projections that need no model at all on exactly the same players:

* **last season.** The player's own prior-season points, carried forward. The
  oldest projection there is, and a hard one to beat in a sport with this much
  year-to-year persistence.
* **ADP itself.** The market's own forecast, converted to points by learning the
  rank-to-points curve on earlier seasons and applying it to the holdout's
  ranks. This is what a drafter implicitly believes when they take a player at
  their ADP.

Both are honest baselines in the sense that matters: they use only information
available before the season, and they are scored on the same rows with the same
metric.

A third reference runs the other way. Instead of asking how little a projection
can know, it asks how much would be enough:

* **the availability floor.** An oracle handed every player's realized per-game
  scoring rate -- perfect knowledge of how good he turned out to be -- and told
  nothing about who gets hurt. It projects the pool's average availability for
  everyone, so it carries no bias and its whole error is missed games. Nothing
  that improves rate modelling can score below it.

A model that cannot beat "last season" is not earning its complexity. A model
that beats ADP is finding something the market has not priced. Neither
comparison is flattered here -- rookies and anyone without a prior season are
dropped from the last-season baseline's pool and from the model's, so the
comparison is like for like.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
from ffmodel.simulation.scoring import fantasy_points

MODEL_POSITIONS = ("QB", "RB", "WR", "TE")


def normalise(name: str) -> str:
    text = str(name).lower()
    text = re.sub(r"\s+(jr|sr|ii|iii|iv|v)\.?$", "", text)
    return re.sub(r"[^a-z]", "", text)


def season_points(rows: pd.DataFrame, scoring: str) -> pd.Series:
    return fantasy_points(observed_scoring_rows(rows.reset_index(drop=True)), scoring)


def load_adp(season: int, directory: Path) -> pd.DataFrame:
    adp = pd.read_csv(directory / f"FantasyPros_{season}_Overall_ADP_Rankings.csv")
    parsed = adp["Player (Bye)"].astype(str).str.strip().str.extract(
        r"^(?P<name>.*?)\s+(?P<team>[A-Z]{2,3})\s*\(\w+\)$"
    )
    adp["adp_name"] = parsed["name"].fillna(adp["Player (Bye)"].astype(str).str.strip())
    adp["adp_position"] = adp["POS"].astype(str).str.extract(r"^([A-Z]+)")[0]
    adp["adp_rank"] = pd.to_numeric(adp["Rank"], errors="coerce")
    adp["key"] = adp["adp_name"].map(normalise)
    return adp


def report(name: str, observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "projection": name,
        "n": int(len(observed)),
        "mae": float(np.abs(error).mean()),
        "mae_pct": float(np.abs(error).mean() / max(observed.mean(), 1e-9)),
        "bias": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": float(np.corrcoef(observed, predicted)[0, 1]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--adp-dir", type=Path, default=Path("ADP"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-wf-2025"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or Path(
        f"scripts/validation_runs/adp_baselines_{args.season}.json"
    )

    pr = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    pr = pr[pr.position.isin(MODEL_POSITIONS)].copy()
    pr["key"] = pr["player_name"].map(normalise)
    pr["points"] = np.nan
    for season, block in pr.groupby("season"):
        pr.loc[block.index, "points"] = season_points(block, args.scoring).to_numpy()

    named = (
        pd.to_numeric(pr.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    )
    holdout = pr[pr.season.eq(args.season) & named & pr.points.notna()].copy()

    adp = load_adp(args.season, args.adp_dir)
    drafted = adp[adp.adp_rank.le(args.top) & adp.adp_position.isin(MODEL_POSITIONS)]
    ranks = dict(zip(drafted["key"], drafted["adp_rank"]))
    holdout["adp_rank"] = holdout["key"].map(ranks)
    pool = holdout[holdout.adp_rank.notna()].copy()

    # Last season, carried forward. Only for players who have one.
    #
    # Summed by player rather than indexed by row: a player traded mid-season
    # has one row per stint, and his season total is the sum of them. Indexing
    # by row raises on the duplicate key, which is the good failure -- silently
    # taking the first stint would have credited a traded player with half a
    # season and called it his prior-year baseline.
    prior = (
        pr[pr.season.eq(args.season - 1)].groupby("key", as_index=True)["points"].sum()
    )
    pool["last_season"] = pool["key"].map(prior)

    # ADP converted to points, with the curve learned on earlier seasons only.
    # Fitting it on the holdout would let the baseline see the answer.
    history = []
    for season in sorted(s for s in pr.season.unique() if s < args.season):
        try:
            past_adp = load_adp(int(season), args.adp_dir)
        except FileNotFoundError:
            continue
        past_ranks = dict(
            zip(
                past_adp[past_adp.adp_position.isin(MODEL_POSITIONS)]["key"],
                past_adp[past_adp.adp_position.isin(MODEL_POSITIONS)]["adp_rank"],
            )
        )
        block = pr[pr.season.eq(season) & named & pr.points.notna()].copy()
        block["adp_rank"] = block["key"].map(past_ranks)
        history.append(block[block.adp_rank.notna()][["adp_rank", "points"]])
    curve = pd.concat(history, ignore_index=True)
    # Points fall off smoothly with rank; a log fit is the standard shape and
    # needs no tuning, which keeps this a baseline rather than a second model.
    coefficients = np.polyfit(np.log(curve["adp_rank"]), curve["points"], 1)
    pool["adp_points"] = np.polyval(coefficients, np.log(pool["adp_rank"]))

    model_path = Path(
        f"scripts/validation_runs/adp_top{args.top}_{args.season}.json"
    )
    model = json.loads(model_path.read_text())["groups"][f"ADP top {args.top}"]

    # "Last season" only exists for players who have one, so it is scored on a
    # subset. The model has to be re-scored on that same subset or the two
    # numbers are describing different populations -- and the dropped players
    # are rookies, the hardest rows in the pool, so the mismatch would flatter
    # the baseline rather than fail loudly.
    row_path = model_path.with_suffix(".rows.csv")
    if row_path.exists():
        per_row = pd.read_csv(row_path)
        pool = pool.merge(
            per_row[["key", "predicted"]], on="key", how="left", validate="one_to_one"
        )

    complete = pool[pool.last_season.notna()]
    results = [
        report("last season", complete["points"].to_numpy(), complete["last_season"].to_numpy()),
        report("ADP curve", complete["points"].to_numpy(), complete["adp_points"].to_numpy()),
    ]
    if "predicted" in complete and complete["predicted"].notna().all():
        results.insert(
            0,
            report(
                "the model",
                complete["points"].to_numpy(),
                complete["predicted"].to_numpy(),
            ),
        )
    everyone = report("ADP curve (all matched)", pool["points"].to_numpy(), pool["adp_points"].to_numpy())

    # How much of the error is not forecastable at all?
    #
    # This oracle is handed every player's realized per-game scoring rate --
    # perfect knowledge of how good he was, which no preseason projection has --
    # and is told nothing about who gets hurt. It projects the pool's average
    # availability for everyone, so it is unbiased by construction and its error
    # is availability alone. Whatever it scores is a floor: the part of the
    # relative error that better rate modelling cannot touch.
    games = pd.to_numeric(pool["games"], errors="coerce")
    team_games = pd.to_numeric(pool["team_games"], errors="coerce")
    playable = games.gt(0) & team_games.gt(0)
    rate = (pool.loc[playable, "points"] / games[playable]).to_numpy()
    slate = team_games[playable].to_numpy(float)
    availability = float((games[playable] / team_games[playable]).mean())
    floor = report(
        "availability floor",
        pool.loc[playable, "points"].to_numpy(),
        rate * slate * availability,
    )
    floor["mean_availability"] = availability

    print(f"\n{args.season} {args.scoring.upper()}, ADP top {args.top}\n")
    print(f"  {'projection':26s} {'n':>4s} {'MAE':>8s} {'MAE %':>7s} {'bias':>8s} "
          f"{'RMSE':>8s} {'corr':>6s}")
    print(f"  {'the model':26s} {model['n']:>4d} {model['mae']:>8.2f} "
          f"{model['mae_pct']:>6.1%} {model['bias']:>+8.2f} {'-':>8s} {'-':>6s}"
          "   <- on all matched rows")
    for row in [everyone]:
        print(f"  {row['projection']:26s} {row['n']:>4d} {row['mae']:>8.2f} "
              f"{row['mae_pct']:>6.1%} {row['bias']:>+8.2f} {row['rmse']:>8.2f} "
              f"{row['correlation']:>6.3f}")
    print(f"  {floor['projection']:26s} {floor['n']:>4d} {floor['mae']:>8.2f} "
          f"{floor['mae_pct']:>6.1%} {floor['bias']:>+8.2f} {floor['rmse']:>8.2f} "
          f"{floor['correlation']:>6.3f}   <- perfect rates, no injury foresight")
    print(f"\n  restricted to players with a prior season (n={len(complete)}):")
    for row in results:
        print(f"  {row['projection']:26s} {row['n']:>4d} {row['mae']:>8.2f} "
              f"{row['mae_pct']:>6.1%} {row['bias']:>+8.2f} {row['rmse']:>8.2f} "
              f"{row['correlation']:>6.3f}")

    payload = {
        "season": args.season,
        "top": args.top,
        "scoring": args.scoring,
        "model": model,
        "baselines": results + [everyone, floor],
        "adp_curve_coefficients": [float(c) for c in coefficients],
        "curve_training_rows": int(len(curve)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
