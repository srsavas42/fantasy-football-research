"""The NGS screen again, against the design the model actually has.

The first pass controlled only for the shrunk prior rate, and that was the wrong
baseline. Every efficiency spec carries an ``advanced_features`` block that is
*not* empty, and for the receiving responses it already contains the play-by-play
versions of the two fields the first screen flagged:

    prior_rec_air_yards_per_target    aDOT, from pbp
    prior_rec_yac_per_reception       YAC, from pbp

So "aDOT adds 2.8% to catch rate" was measured against a model that already
knows a player's aDOT by another name. The honest question is whether the
tracking version says anything past the charting version, and that needs the
whole existing design regressed out first, not just the prior.

The design differs by response, and the difference decides what a hit is worth:

    posterior   prior + exposure + base + volume + advanced (+ teammate, reserve)
                rec_yards_per_target
    ridge       the same block, fitted outside the likelihood
                every pass_* response, rush_yards_per_carry
    persistence prior only -- an empty covariate design on purpose
                rec_catch_rate, rec_td_rate, rush_td_rate

A hit on a persistence response is not a feature that can be added; it is an
argument for changing that response's mean mode, which is a different and much
larger change. Those are scored here and labelled, not proposed.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

import nflreadpy as nfl

from ffmodel.models.efficiency_season_average import (
    BASE_EFFICIENCY_FEATURES,
    EFFICIENCY_MODEL_BY_TARGET,
    PERSISTENCE_MEAN_MODE,
    POSTERIOR_MEAN_MODE,
    RESERVE_EFFICIENCY_TARGETS,
    TEAMMATE_QUALITY_TARGETS,
)

pd.set_option("display.width", 240)
CACHE = "/home/user/fantasy-football-research/.cache/ffmodel-2026"
SEASONS = list(range(2016, 2026))

NGS_SPECS = {
    "receiving": {
        "exposure": "targets",
        "metrics": (
            "avg_cushion", "avg_separation", "avg_intended_air_yards",
            "percent_share_of_intended_air_yards", "avg_yac",
            "avg_expected_yac", "avg_yac_above_expectation",
        ),
    },
    "rushing": {
        "exposure": "rush_attempts",
        "metrics": (
            "efficiency", "percent_attempts_gte_eight_defenders",
            "avg_time_to_los", "rush_yards_over_expected_per_att",
            "rush_pct_over_expected",
        ),
    },
    "passing": {
        "exposure": "attempts",
        "metrics": (
            "avg_time_to_throw", "avg_completed_air_yards",
            "avg_intended_air_yards", "avg_air_yards_differential",
            "aggressiveness", "avg_air_yards_to_sticks",
            "expected_completion_percentage",
            "completion_percentage_above_expectation", "avg_air_distance",
        ),
    },
}

RESPONSES = (
    ("rec_yards_per_target", "receiving", "targets"),
    ("rec_catch_rate", "receiving", "targets"),
    ("rec_td_rate", "receiving", "targets"),
    ("rush_yards_per_carry", "rushing", "rush_att"),
    ("rush_td_rate", "rushing", "rush_att"),
    ("pass_yards_per_attempt", "passing", "pass_att"),
    ("pass_completion_rate", "passing", "pass_att"),
    ("pass_td_rate", "passing", "pass_att"),
)


def mean_mode(target: str) -> str:
    return PERSISTENCE_MEAN_MODE.get(target) or POSTERIOR_MEAN_MODE[target]


def control_columns(target: str) -> list[str]:
    """The covariates the shipping pipeline already hands this response."""
    spec = EFFICIENCY_MODEL_BY_TARGET[target]
    if mean_mode(target) == "persistence":
        return [spec.prior_feature]
    names = [spec.prior_feature, spec.prior_exposure, spec.volume_feature]
    names += list(BASE_EFFICIENCY_FEATURES)
    names += list(spec.advanced_features)
    if target in TEAMMATE_QUALITY_TARGETS:
        names.append("teammate_qb_quality_signal")
    if target in RESERVE_EFFICIENCY_TARGETS:
        names.append("roster_reserve")
    return list(dict.fromkeys(names))


def season_ngs(stat_type: str, *, lag: bool = True) -> pd.DataFrame:
    spec = NGS_SPECS[stat_type]
    expo = spec["exposure"]
    d = nfl.load_nextgen_stats(seasons=SEASONS, stat_type=stat_type).to_pandas()
    d = d[d.season_type.eq("REG") & d.week.gt(0)].copy()
    d[expo] = pd.to_numeric(d[expo], errors="coerce").fillna(0.0)
    frames = []
    for metric in spec["metrics"]:
        v = pd.to_numeric(d[metric], errors="coerce")
        ok = v.notna() & d[expo].gt(0)
        agg = (
            pd.DataFrame({
                "season": d.season, "player_id": d.player_gsis_id,
                "num": (v * d[expo]).where(ok, 0.0), "den": d[expo].where(ok, 0.0),
            })
            .groupby(["season", "player_id"], as_index=False)[["num", "den"]].sum()
        )
        agg[metric] = np.where(agg.den.gt(0), agg.num / agg.den, np.nan)
        frames.append(agg[["season", "player_id", metric]])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["season", "player_id"], how="outer")
    if lag:
        merged["season"] = merged.season + 1
    return merged


def residualise(y: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    """Least-squares residual of y on the controls, median-filled as the model does."""
    x = controls.apply(pd.to_numeric, errors="coerce")
    x = x.loc[:, x.notna().any() & (x.std(ddof=0) > 1e-8)]
    design = [np.ones(len(y))]
    for name in x.columns:
        values = x[name]
        design.append(values.fillna(values.median()).to_numpy(dtype=float))
        if values.isna().any():
            design.append(values.isna().to_numpy(dtype=float))
    matrix = np.column_stack(design)
    beta, *_ = np.linalg.lstsq(matrix, y.to_numpy(dtype=float), rcond=None)
    return y.to_numpy(dtype=float) - matrix @ beta


def main() -> int:
    pr = pd.read_pickle(f"{CACHE}/player_rows.pkl")
    p = pr[pr.season.isin(SEASONS)].copy()
    p = p[pd.to_numeric(p.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]
    ngs = {group: season_ngs(group) for group in NGS_SPECS}

    for target, group, expo_col in RESPONSES:
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        mode = mean_mode(target)
        controls = [c for c in control_columns(target) if c in p.columns]
        missing = [c for c in control_columns(target) if c not in p.columns]
        m = p.merge(ngs[group], on=["season", "player_id"], how="left")
        y = pd.to_numeric(m[target], errors="coerce")
        q = pd.to_numeric(m[spec.prior_feature], errors="coerce")
        e = pd.to_numeric(m[expo_col], errors="coerce").fillna(0)
        keep = y.notna() & q.notna() & e.ge(spec.min_exposure)
        keep &= m["position"].astype(str).str.upper().isin(spec.positions)
        metrics = list(NGS_SPECS[group]["metrics"])
        keep &= m[metrics].notna().any(axis=1)
        sub = m[keep]
        if len(sub) < 60:
            print(f"\n{target}: only {len(sub)} covered rows, skipped")
            continue

        ys = y[keep]
        base_resid = residualise(ys, sub[controls])
        prior_resid = residualise(ys, sub[[spec.prior_feature]])
        explained = 1 - base_resid.var() / ys.to_numpy(dtype=float).var()
        print(f"\n{'=' * 100}")
        print(f"{target}   mode={mode}   n={len(sub)}   "
              f"controls={len(controls)}"
              + (f"  (absent from cache: {missing})" if missing else ""))
        print(f"  existing design explains {explained:5.1%} of this response; "
              f"residual sd {base_resid.std():.4f} vs response sd {ys.std():.4f}")
        if mode == "persistence":
            print("  NOTE: this response fits an empty covariate design on purpose; "
                  "any hit below is an argument for a mode change, not a feature")
        print(f"{'=' * 100}")

        rows = []
        for metric in metrics:
            x = pd.to_numeric(sub[metric], errors="coerce")
            ok = x.notna().to_numpy()
            if ok.sum() < 60:
                continue
            r_prior, _ = stats.pearsonr(x[ok], prior_resid[ok])
            r_full, p_full = stats.pearsonr(x[ok], base_resid[ok])
            rows.append({
                "metric": metric, "n": int(ok.sum()),
                "vs_prior_only": r_prior, "vs_full_design": r_full,
                "p": p_full, "r2_add": r_full ** 2,
            })
        table = pd.DataFrame(rows).sort_values("r2_add", ascending=False)
        table["verdict"] = np.where(
            (table.p < 0.01) & (table.r2_add > 0.01), "adds", ""
        )
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
