"""Does the pipeline's structure earn its complexity?

The shipping model is a chain: availability, then roster-share volume, then
per-response efficiency, then a coherent stat-line simulation, then points. Each
layer is fitted, tested and promoted on its own terms. None of that is evidence
that the *chain* beats predicting the answer directly from the same inputs, and
that comparison has never been run.

This is the flat control. One regression, all preseason-safe inputs, season PPR
points straight out -- no availability model, no share allocation, no efficiency
responses, no simulation.

Features are safe by construction rather than by inspection: everything named
``prior_*`` (lagged by the feature layer), everything named ``adp_*`` (a
preseason draft board), and a whitelist of demographics and roster status. No
current-season column can reach it, which is the only way to make the comparison
mean anything.

The estimator is ridge with the penalty chosen on a held-out slice of the
training seasons, and spread comes from a residual pool local in predicted
value -- the same way ``RankCurve`` builds its distribution, so the CRPS
comparison is not decided by one side having a better uncertainty model bolted
on. A gradient-boosted challenger would be the stronger flat model and is not
available in this environment; the repository's own comments refer to an
optional XGBoost challenger elsewhere, so a tree ensemble is the obvious next
version of this control if it ever matters.

**The arms decompose a confound rather than conflating it.** The shipped
pipeline sets ``market_adp_features = False`` -- it never sees the draft board.
A flat model handed every preseason column does see it, and ADP is the single
most predictive input available: |r| = 0.72 against season points, against 0.61
for the best play-by-play feature. A two-way comparison would measure "flat model
with ADP against pipeline without it", and the answer would invite being read as
a statement about architecture.

    pipeline       the shipped chain, no ADP
    flat_no_adp    ridge on preseason columns with every adp_* removed
    flat_ridge     the same ridge with the board included
    flat_gbm       gradient boosting on every preseason column
    adp            the rank curve alone

    blend          the pipeline joined to the board at 0.316, which is what
                   project_season.py actually projects

Two questions, kept apart. Among the arms without ADP -- pipeline, flat_no_adp,
gbm_no_adp -- the comparison is about architecture: does a chain of availability,
volume, efficiency and simulation beat one regression on the same inputs.

Among the arms with it -- blend, flat_ridge, flat_gbm -- the comparison is about
the deliverable. A flat model carrying adp_* columns must be measured against
the pipeline *after* it is joined to the board, not against the raw pipeline,
because the raw pipeline is not what ships and does not have the board. That
join is a constant-weight mixture rather than a refit, since refitting the
weight here would use the holdout.

    python scripts/validate_flat_baseline.py --holdouts 2023
    python scripts/validate_flat_baseline.py --merge a.json b.json c.json
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

from ffmodel.evaluation.efficiency_posterior import observed_scoring_rows
from ffmodel.evaluation.metrics import empirical_crps, interval_coverage
from ffmodel.features.season_average import SeasonAverageData
from ffmodel.models.market_blend import RankCurve, blend_samples
from ffmodel.models.season_scoring import SeasonAverageScoringPipeline
from ffmodel.simulation.scoring import fantasy_points

DEMOGRAPHIC_FEATURES = (
    "age", "experience", "team_change", "cold_start", "depth_rank",
    "draft_target_prior", "draft_carry_prior", "draft_pass_prior",
    "combine_forty", "athletic_score", "team_games",
    "roster_reserve", "roster_injured_reserve", "roster_pup", "roster_nfi",
    "suspended_games", "mandatory_missed_games",
)
TIERS = (("top50", 1, 50), ("51_150", 51, 150), ("151_300", 151, 300), ("drafted", 1, 400))
# What actually ships. project_season.py projects the blend, not the raw
# pipeline, so a flat model carrying ADP among its features must be compared
# against the pipeline *after* it is joined to the board.
BLEND_WEIGHT = 0.316
# Local residual pool for the flat model's spread, in predicted-value space.
POOL_WINDOW = 25.0
MIN_POOL = 40


def preseason_features(rows: pd.DataFrame, *, use_adp: bool = True) -> list[str]:
    """Everything a preseason forecast may see, by naming convention."""
    names = [c for c in rows.columns if c.startswith("prior_")]
    if use_adp:
        names += [c for c in rows.columns if c.startswith("adp_")]
    names += [c for c in DEMOGRAPHIC_FEATURES if c in rows.columns]
    return list(dict.fromkeys(names))


class FlatBoost:
    """Gradient boosting on the same columns, with the same residual spread.

    The stronger flat challenger. If the pipeline's structure is worth having it
    should beat a learner that can find interactions a ridge cannot.
    """

    def __init__(self, *, use_adp: bool = True, seed: int = 0):
        self.use_adp = use_adp
        self.seed = seed
        self.names: list[str] = []
        self.model = None
        self.residuals = np.array([])
        self.fitted_values = np.array([])

    def fit(self, rows: pd.DataFrame, y: np.ndarray) -> "FlatBoost":
        from xgboost import XGBRegressor

        self.names = preseason_features(rows, use_adp=self.use_adp)
        x = rows[self.names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        self.model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0,
            random_state=self.seed, n_jobs=4,
        )
        self.model.fit(x, np.asarray(y, dtype=float))
        self.fitted_values = self.model.predict(x)
        self.residuals = np.asarray(y, dtype=float) - self.fitted_values
        return self

    def predict_samples(self, rows: pd.DataFrame, draws: int, seed: int = 0) -> np.ndarray:
        x = rows.reindex(columns=self.names).apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        centre = self.model.predict(x)
        rng = np.random.default_rng(seed)
        out = np.empty((len(centre), draws), dtype=float)
        for i, value in enumerate(centre):
            near = np.abs(self.fitted_values - value) <= POOL_WINDOW
            pool = self.residuals[near] if near.sum() >= MIN_POOL else self.residuals
            out[i] = value + rng.choice(pool, size=draws, replace=True)
        return np.maximum(out, 0.0)


class FlatRidge:
    """Ridge on standardised features, with a local residual pool for spread."""

    def __init__(self, alphas=(3.0, 10.0, 30.0, 100.0, 300.0, 1000.0), *, use_adp=True):
        self.alphas = alphas
        self.use_adp = use_adp
        self.names: list[str] = []
        self.mean = self.scale = self.beta = None
        self.intercept = 0.0
        self.residuals = np.array([])
        self.fitted = np.array([])
        self.alpha = None

    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        block = rows.reindex(columns=self.names).apply(pd.to_numeric, errors="coerce")
        filled = block.fillna(pd.Series(self.fill, index=self.names))
        return ((filled.to_numpy(dtype=float) - self.mean) / self.scale)

    def fit(self, rows: pd.DataFrame, y: np.ndarray, *, seed: int = 0) -> "FlatRidge":
        self.names = preseason_features(rows, use_adp=self.use_adp)
        block = rows[self.names].apply(pd.to_numeric, errors="coerce")
        self.fill = block.median().fillna(0.0).to_dict()
        filled = block.fillna(pd.Series(self.fill))
        raw = filled.to_numpy(dtype=float)
        self.mean = raw.mean(axis=0)
        scale = raw.std(axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        x = (raw - self.mean) / self.scale
        y = np.asarray(y, dtype=float)

        # Penalty chosen on a held-out slice of the training rows, never on the
        # holdout season.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y))
        cut = int(len(y) * 0.8)
        tr, va = order[:cut], order[cut:]
        best = (None, np.inf)
        for alpha in self.alphas:
            beta, intercept = self._solve(x[tr], y[tr], alpha)
            error = np.abs(y[va] - (x[va] @ beta + intercept)).mean()
            if error < best[1]:
                best = (alpha, error)
        self.alpha = best[0]
        self.beta, self.intercept = self._solve(x, y, self.alpha)
        self.fitted = x @ self.beta + self.intercept
        self.residuals = y - self.fitted
        return self

    @staticmethod
    def _solve(x, y, alpha):
        centre = y.mean()
        gram = x.T @ x + alpha * np.eye(x.shape[1])
        beta = np.linalg.solve(gram, x.T @ (y - centre))
        return beta, centre

    def predict_samples(self, rows: pd.DataFrame, draws: int, seed: int = 0) -> np.ndarray:
        centre = self._matrix(rows) @ self.beta + self.intercept
        rng = np.random.default_rng(seed)
        out = np.empty((len(centre), draws), dtype=float)
        for i, value in enumerate(centre):
            near = np.abs(self.fitted - value) <= POOL_WINDOW
            pool = self.residuals[near] if near.sum() >= MIN_POOL else self.residuals
            out[i] = value + rng.choice(pool, size=draws, replace=True)
        return np.maximum(out, 0.0)


def _metrics(observed: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.abs(observed - samples.mean(axis=1)).mean()),
        "crps": float(empirical_crps(observed, samples).mean()),
        "coverage_80": float(interval_coverage(observed, samples, level=0.8)["coverage"]),
        "n": int(len(observed)),
    }


def _run_fold(holdout: int, cache_dir: Path, *, draws, tune, chains, seed):
    player_rows = pd.read_pickle(cache_dir / "player_rows.pkl")
    team_rows = pd.read_pickle(cache_dir / "team_rows.pkl")
    player_rows = player_rows[player_rows.season.lt(2026)]
    team_rows = team_rows[team_rows.season.lt(2026)]
    train = SeasonAverageData(
        team_rows[team_rows.season.lt(holdout)].copy(),
        player_rows[player_rows.season.lt(holdout)].copy(),
    )
    test = SeasonAverageData(
        team_rows[team_rows.season.eq(holdout)].copy(),
        player_rows[player_rows.season.eq(holdout)].copy(),
    )

    pipeline = SeasonAverageScoringPipeline()
    kwargs = {"draws": draws, "tune": tune, "chains": chains}
    pipeline.fit(train, volume_sample_kwargs=kwargs, efficiency_sample_kwargs=kwargs)
    prediction = pipeline.predict_samples(test, seed=seed)
    rows = prediction.player_rows.reset_index(drop=True)
    observed = fantasy_points(observed_scoring_rows(rows), "ppr").to_numpy(dtype=float)
    model = np.asarray(prediction.fantasy_points["ppr"], dtype=float)
    n_draws = model.shape[1]

    train_rows = train.player_rows.reset_index(drop=True)
    train_points = fantasy_points(
        observed_scoring_rows(train_rows), "ppr"
    ).to_numpy(dtype=float)
    usable = np.isfinite(train_points)
    flat = FlatRidge().fit(train_rows[usable], train_points[usable], seed=seed)
    flat_samples = flat.predict_samples(rows, draws=n_draws, seed=seed + 3)
    no_adp = FlatRidge(use_adp=False).fit(
        train_rows[usable], train_points[usable], seed=seed
    )
    no_adp_samples = no_adp.predict_samples(rows, draws=n_draws, seed=seed + 4)
    boost = FlatBoost(seed=seed).fit(train_rows[usable], train_points[usable])
    boost_samples = boost.predict_samples(rows, draws=n_draws, seed=seed + 5)
    no_adp_boost = FlatBoost(use_adp=False, seed=seed).fit(
        train_rows[usable], train_points[usable]
    )
    no_adp_boost_samples = no_adp_boost.predict_samples(
        rows, draws=n_draws, seed=seed + 6
    )
    curve = RankCurve().fit(train_rows[usable], train_points[usable])
    adp_samples = curve.predict_samples(rows, draws=n_draws, seed=seed + 7)

    blend = blend_samples(model, adp_samples, BLEND_WEIGHT, seed=seed + 11)
    rank = pd.to_numeric(rows.get("adp_rank"), errors="coerce").to_numpy(float)
    keep = np.isfinite(observed) & np.isfinite(model).all(axis=1)
    fold = {"alpha": float(flat.alpha), "features": len(flat.names), "tiers": {}}
    for name, low, high in TIERS:
        mask = keep & np.isfinite(rank) & (rank >= low) & (rank <= high)
        mask &= np.isfinite(adp_samples).all(axis=1)
        if mask.sum() < 25:
            continue
        fold["tiers"][name] = {
            "pipeline": _metrics(observed[mask], model[mask]),
            "flat_ridge": _metrics(observed[mask], flat_samples[mask]),
            "flat_no_adp": _metrics(observed[mask], no_adp_samples[mask]),
            "flat_gbm": _metrics(observed[mask], boost_samples[mask]),
            "gbm_no_adp": _metrics(observed[mask], no_adp_boost_samples[mask]),
            "blend": _metrics(observed[mask], blend[mask]),
            "adp": _metrics(observed[mask], adp_samples[mask]),
        }
    everyone = keep & np.isfinite(flat_samples).all(axis=1)
    fold["tiers"]["all_rostered"] = {
        "pipeline": _metrics(observed[everyone], model[everyone]),
        "flat_ridge": _metrics(observed[everyone], flat_samples[everyone]),
        "flat_no_adp": _metrics(observed[everyone], no_adp_samples[everyone]),
        "flat_gbm": _metrics(observed[everyone], boost_samples[everyone]),
        "gbm_no_adp": _metrics(observed[everyone], no_adp_boost_samples[everyone]),
        "blend": _metrics(observed[everyone], blend[everyone]),
    }
    del pipeline, prediction, model, flat_samples, no_adp_samples, boost_samples
    gc.collect()
    return fold


def _report(report: dict, args) -> int:
    folds = report["folds"]
    holdouts = sorted(int(k) for k in folds)
    alphas = [folds[str(h)]["alpha"] for h in holdouts]
    features = [folds[str(h)]["features"] for h in holdouts]
    print(f"\n{'=' * 86}\nflat regression against the pipeline and the board\n{'=' * 86}")
    print(f"  ridge penalties chosen per fold: {alphas}; {features[0]} features")
    arms = ("pipeline", "flat_no_adp", "gbm_no_adp", "adp", "blend",
            "flat_ridge", "flat_gbm")
    for metric in ("mae", "crps"):
        print(f"\n  {metric.upper()}")
        print(f"  {'tier':13} {'n':>5} " + "".join(f"{a:>13}" for a in arms))
        print("  " + "-" * (20 + 13 * len(arms)))
        for name in [t[0] for t in TIERS] + ["all_rostered"]:
            blocks = [folds[str(h)]["tiers"][name] for h in holdouts
                      if name in folds[str(h)]["tiers"]]
            if not blocks:
                continue
            n = int(np.mean([b["pipeline"]["n"] for b in blocks]))
            cells = []
            for arm in arms:
                values = [b[arm][metric] for b in blocks if arm in b]
                cells.append(f"{np.mean(values):13.2f}" if values else f"{'--':>13}")
            print(f"  {name:13} {n:5d}" + "".join(cells))
    print("\n  no ADP:   pipeline | flat_no_adp | gbm_no_adp  -- the architecture question")
    print("  with ADP: blend    | flat_ridge  | flat_gbm    -- the deliverable question")
    print("  blend is what ships: the pipeline joined to the board at 0.316.")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, default=str), "utf-8")
    print(f"\nwrote {args.report_json}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-2026"))
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-json", type=Path, default=Path("reports/flat_baseline.json")
    )
    parser.add_argument("--merge", type=Path, nargs="+", default=None)
    args = parser.parse_args(argv)

    if args.merge:
        folds: dict = {}
        for path in args.merge:
            folds.update(json.loads(path.read_text("utf-8"))["folds"])
        return _report({"folds": folds}, args)

    report: dict[str, object] = {"folds": {}}
    for holdout in args.holdouts:
        report["folds"][str(holdout)] = _run_fold(
            holdout, args.cache_dir,
            draws=args.draws, tune=args.tune, chains=args.chains, seed=args.seed,
        )
        print(f"holdout {holdout} done", flush=True)
    return _report(report, args)


if __name__ == "__main__":
    raise SystemExit(main())
