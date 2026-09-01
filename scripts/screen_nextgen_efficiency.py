"""Do Next Gen Stats tell the efficiency layer anything it does not already know?

The efficiency layer has never seen a tracking number. Every response reads
prior_availability, prior_snap_share, age, experience, team_change and
cold_start, plus the shrunk prior rate; ``load_nextgen_stats`` is called only by
the weekly pipeline and by the raw ingest layer, and reaches the season-average
feature builder nowhere.

This is a better prospect than the two team-level ideas that just failed. Those
died on a shared defect -- a team number is mostly the player standing in front
of it, and what survives that does not survive a year. NGS is charted per player,
so neither objection applies by construction, and several of its fields describe
something the response cannot express on its own:

    aDOT               a deep threat's yards per target is structurally high and
                       his catch rate structurally low; the model currently has
                       to infer depth of target from the rate it is predicting
    separation         route-running quality, upstream of both catch rate and
                       yards per target
    CPOE               completion percentage over expectation, the standard
                       skill-versus-situation split for a passer
    YAC over expected  the receiver's own half of yards per target
    RYOE (player)      the same statistic whose *team* aggregate failed, on the
                       unit it was actually charted for
    stacked boxes      the defensive front a back runs against, which is the
                       part of "opponent strength" that does not average out

The bar is not whether an NGS field correlates with the response. It obviously
does -- aDOT and yards per target are nearly the same measurement. The bar is
whether it says anything *beyond the prior rate the model already carries*, so
every field is scored twice: raw against the next season's rate, and again
against that rate's residual after the shrunk prior has been regressed out. The
second number is the one that decides.

Coverage is the standing constraint. NGS charts a qualifying subset, not the
league: roughly 200 receivers a season against the ~400 the receiving responses
fit on, and 80-85 backs. Passing is nearly complete. The join rate is reported
per response before any correlation, because a feature present on half the fit
population is a different proposition from one present on all of it.

**This screen's control was too weak and its headline is wrong.** It regresses
out only the shrunk prior rate, but every spec carries a non-empty
``advanced_features`` block, and the receiving one already contains the
play-by-play aDOT and YAC this screen "discovers" in the tracking feed.
``screen_nextgen_beyond_design.py`` redoes it against the design each response
actually has; read that one for the verdict. What survives there is nothing on
any response that already fits covariates, and the one real signal --
aDOT against catch rate -- turns out to be a pbp field the model has and
discards, not a tracking field it lacks.

Kept as written because the two-control comparison is the point: the gap
between this screen's numbers and the next one's is the size of the
contaminated-baseline error, and it is large. aDOT against
rec_yards_per_target reads r = +0.080 here and -0.016 there.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

import nflreadpy as nfl

pd.set_option("display.width", 240)
CACHE = "/home/user/fantasy-football-research/.cache/ffmodel-2026"
SEASONS = list(range(2016, 2026))

# metric -> the exposure it should be weighted by when weeks are summed.
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

# response -> (NGS group, prior rate column, exposure column, min exposure)
RESPONSES = (
    ("rec_yards_per_target", "receiving", "prior_rec_yards_per_target", "targets", 20),
    ("rec_catch_rate", "receiving", "prior_rec_catch_rate", "targets", 20),
    ("rec_td_rate", "receiving", "prior_rec_td_rate", "targets", 20),
    ("rush_yards_per_carry", "rushing", "prior_rush_yards_per_carry", "rush_att", 20),
    ("rush_td_rate", "rushing", "prior_rush_td_rate", "rush_att", 20),
    ("pass_yards_per_attempt", "passing", "prior_pass_yards_per_attempt", "pass_att", 50),
    ("pass_completion_rate", "passing", "prior_pass_completion_rate", "pass_att", 50),
    ("pass_td_rate", "passing", "prior_pass_td_rate", "pass_att", 50),
)


def season_ngs(stat_type: str) -> pd.DataFrame:
    """Exposure-weighted season aggregate of the weekly charting.

    The published season rows (``week == 0``) are a shorter leaderboard than the
    weekly ones -- 115 receivers against 202 in 2023 -- so the weeks are summed
    here instead. A metric is a per-play average, so it is re-weighted by the
    exposure of the weeks it was charted in rather than averaged flat.
    """
    spec = NGS_SPECS[stat_type]
    expo = spec["exposure"]
    d = nfl.load_nextgen_stats(seasons=SEASONS, stat_type=stat_type).to_pandas()
    d = d[d.season_type.eq("REG") & d.week.gt(0)].copy()
    d[expo] = pd.to_numeric(d[expo], errors="coerce").fillna(0.0)
    out = {}
    for metric in spec["metrics"]:
        v = pd.to_numeric(d[metric], errors="coerce")
        ok = v.notna() & d[expo].gt(0)
        num = (v * d[expo]).where(ok, 0.0)
        den = d[expo].where(ok, 0.0)
        frame = pd.DataFrame(
            {"season": d.season, "player_id": d.player_gsis_id, "num": num, "den": den}
        )
        agg = frame.groupby(["season", "player_id"], as_index=False)[["num", "den"]].sum()
        agg[metric] = np.where(agg.den.gt(0), agg.num / agg.den, np.nan)
        out[metric] = agg[["season", "player_id", metric]]
    merged = out[spec["metrics"][0]]
    for metric in spec["metrics"][1:]:
        merged = merged.merge(out[metric], on=["season", "player_id"], how="outer")
    totals = d.groupby(["season", "player_id"], as_index=False).size() if False else None
    expo_total = (
        pd.DataFrame({"season": d.season, "player_id": d.player_gsis_id, "e": d[expo]})
        .groupby(["season", "player_id"], as_index=False)["e"].sum()
        .rename(columns={"e": f"ngs_{expo}"})
    )
    return merged.merge(expo_total, on=["season", "player_id"], how="left")


def main() -> int:
    pr = pd.read_pickle(f"{CACHE}/player_rows.pkl")
    p = pr[pr.season.isin(SEASONS)].copy()
    p = p[pd.to_numeric(p.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]

    print("=== building the prior-season NGS frames ===")
    ngs = {}
    for group in NGS_SPECS:
        frame = season_ngs(group)
        # lag one season: the covariate for season s is what he did in s-1.
        frame = frame.copy()
        frame["season"] = frame.season + 1
        ngs[group] = frame
        print(f"  {group:10} {len(frame):5d} player-seasons, "
              f"{frame.player_id.nunique()} players")

    for response, group, prior_col, expo_col, floor in RESPONSES:
        frame = ngs[group]
        metrics = NGS_SPECS[group]["metrics"]
        m = p.merge(frame, on=["season", "player_id"], how="left")
        y = pd.to_numeric(m[response], errors="coerce")
        q = pd.to_numeric(m[prior_col], errors="coerce")
        e = pd.to_numeric(m[expo_col], errors="coerce").fillna(0)
        fit_pop = y.notna() & q.notna() & e.ge(floor)
        if response.endswith("_rate"):
            fit_pop &= y.ge(0)
        else:
            fit_pop &= y.gt(0.1) & q.gt(0.1)
        covered = fit_pop & m[list(metrics)].notna().any(axis=1)
        print(f"\n{'=' * 96}\n{response}   fit population {int(fit_pop.sum())}, "
              f"NGS on {int(covered.sum())} ({covered.sum() / max(fit_pop.sum(), 1):.0%})"
              f"\n{'=' * 96}")
        if covered.sum() < 60:
            print("  too few covered rows to screen")
            continue

        # The number to beat: what the prior rate alone already explains.
        sub = m[covered]
        ys, qs = y[covered], q[covered]
        r_prior, _ = stats.pearsonr(qs, ys)
        # Residual of the response after the model's own prior signal.
        slope, intercept = np.polyfit(qs, ys, 1)
        resid = ys - (slope * qs + intercept)
        print(f"  prior rate alone: r={r_prior:+.3f}  r2={r_prior**2:5.1%}  "
              f"(residual sd {resid.std():.4f} vs response sd {ys.std():.4f})")

        rows = []
        for metric in metrics:
            x = pd.to_numeric(sub[metric], errors="coerce")
            ok = x.notna()
            if ok.sum() < 60:
                continue
            r_raw, p_raw = stats.pearsonr(x[ok], ys[ok])
            r_res, p_res = stats.pearsonr(x[ok], resid[ok])
            # persistence of the metric itself, on the same population
            rows.append({
                "metric": metric,
                "n": int(ok.sum()),
                "r_raw": r_raw,
                "r_resid": r_res,
                "p_resid": p_res,
                "r2_add": r_res ** 2,
            })
        if not rows:
            print("  no metric had enough coverage")
            continue
        table = pd.DataFrame(rows).sort_values("r2_add", ascending=False)
        table["flag"] = np.where(
            (table.p_resid < 0.01) & (table.r2_add > 0.01), "  <== adds", ""
        )
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print(f"\n{'=' * 96}\npersistence of each NGS metric (can last season's value be used at all?)\n{'=' * 96}")
    for group in NGS_SPECS:
        cur = season_ngs(group)
        nxt = cur.copy(); nxt["season"] = nxt.season + 1
        j = cur.merge(nxt, on=["season", "player_id"], how="inner", suffixes=("", "_prior"))
        print(f"\n-- {group} --")
        for metric in NGS_SPECS[group]["metrics"]:
            pair = j[[metric, f"{metric}_prior"]].dropna()
            if len(pair) < 40:
                print(f"  {metric:42} too few pairs ({len(pair)})")
                continue
            r, pv = stats.pearsonr(pair[f"{metric}_prior"], pair[metric])
            print(f"  {metric:42} r={r:+.3f}  r2={r*r:5.1%}  n={len(pair):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
