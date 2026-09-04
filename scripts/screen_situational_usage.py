"""Where a player's touches come from, not just how many he gets.

Two responses in this layer are asked to predict a rate whose *situation* the
model cannot see. A back who runs on third-and-one and at the goal line will
have a low yards per carry and a high touchdown rate, and a back who runs on
first-and-ten between the twenties will have the opposite, and the pipeline is
handed both as the same statistic. Same on the other side: a deep receiver has a
high yards per target, a high touchdown rate and a *low* catch rate.

That structure is what makes this worth testing rather than another correlation
hunt. The hypothesis predicts opposite signs on responses that share a player,
so it can be wrong in a way a single number cannot:

    short-yardage carries    yards per carry DOWN, rushing TD rate UP
    deep targets             yards per target UP, receiving TD rate UP,
                             catch rate DOWN

What the cache has and does not. Receiving already carries an aDOT --
``prior_rec_air_yards_per_target`` -- so the deep-target arm is partly a test of
whether the *distribution* of depth adds to its mean: a receiver at aDOT 10 may
be running ten-yard outs, or half screens and half go routes, and those are
different players. Rushing carries nothing situational at all. There is no
yards-to-go, no yardline, no goal-line share; ``late_carry_share`` is
fourth-quarter usage, not short yardage. So the rushing arm is testing a
covariate the layer has never had in any form.

Route tree proper is not available. No nflverse feed classifies routes -- pbp
has ``pass_length`` and ``air_yards`` and nothing finer, and FTN's charting has
screens and play-action but no route labels. Depth distribution is the honest
substitute and is named as such rather than dressed up as a route feature.

Controls are the design each response actually fits, following
``screen_nextgen_beyond_design.py``: the previous screen's headline evaporated
when it was scored against the real covariate block instead of the prior alone,
and the same trap is open here, since the receiving block already holds aDOT.
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
SEASONS = list(range(2014, 2026))

RUSH_FEATURES = (
    "rush_ydstogo_mean", "rush_yardline_mean", "rush_inside10_share",
    "rush_inside5_share", "rush_short_yardage_share", "rush_goal_to_go_share",
    "rush_late_down_share",
)
REC_FEATURES = (
    "rec_adot_pbp", "rec_deep_share", "rec_screen_share",
    "rec_intermediate_share", "rec_yardline_mean", "rec_inside10_share",
    "rec_air_yards_sd",
)

# response -> (feature group, the sign the hypothesis predicts, or 0 for none)
HYPOTHESIS = {
    "rush_yards_per_carry": ("rush", {"rush_short_yardage_share": -1,
                                      "rush_inside5_share": -1,
                                      "rush_ydstogo_mean": +1,
                                      "rush_yardline_mean": +1}),
    "rush_td_rate": ("rush", {"rush_short_yardage_share": +1,
                              "rush_inside5_share": +1,
                              "rush_inside10_share": +1,
                              "rush_yardline_mean": -1}),
    "rec_yards_per_target": ("rec", {"rec_deep_share": +1, "rec_adot_pbp": +1,
                                     "rec_screen_share": -1}),
    "rec_td_rate": ("rec", {"rec_deep_share": +1, "rec_inside10_share": +1}),
    "rec_catch_rate": ("rec", {"rec_deep_share": -1, "rec_adot_pbp": -1,
                               "rec_screen_share": +1}),
}


def situational_features() -> pd.DataFrame:
    """Per player-season usage context from play-by-play, lagged one season."""
    rush_rows, rec_rows = [], []
    for season in SEASONS:
        try:
            pbp = nfl.load_pbp(seasons=[season]).to_pandas()
        except Exception as exc:  # a season the feed will not serve yet
            print(f"  {season}: {type(exc).__name__} {str(exc)[:60]}")
            continue
        pbp = pbp[pbp.season_type.eq("REG")]
        ydstogo = pd.to_numeric(pbp.ydstogo, errors="coerce")
        yardline = pd.to_numeric(pbp.yardline_100, errors="coerce")
        down = pd.to_numeric(pbp.down, errors="coerce")
        gtg = pd.to_numeric(pbp.goal_to_go, errors="coerce").fillna(0)

        r = pbp.rush_attempt.eq(1) & pbp.rusher_player_id.notna()
        rush = pd.DataFrame({
            "season": season, "player_id": pbp.rusher_player_id[r],
            "ydstogo": ydstogo[r], "yardline": yardline[r],
            "inside10": yardline[r].le(10).astype(float),
            "inside5": yardline[r].le(5).astype(float),
            "short": ydstogo[r].le(2).astype(float),
            "gtg": gtg[r].astype(float),
            "late_down": down[r].isin((3, 4)).astype(float),
        })
        agg = rush.groupby(["season", "player_id"]).agg(
            rush_plays=("ydstogo", "size"),
            rush_ydstogo_mean=("ydstogo", "mean"),
            rush_yardline_mean=("yardline", "mean"),
            rush_inside10_share=("inside10", "mean"),
            rush_inside5_share=("inside5", "mean"),
            rush_short_yardage_share=("short", "mean"),
            rush_goal_to_go_share=("gtg", "mean"),
            rush_late_down_share=("late_down", "mean"),
        ).reset_index()
        rush_rows.append(agg)

        # Targets, including incompletions: the depth of a target is charted
        # whether or not it was caught, and restricting to receptions would
        # select on the very outcome catch rate is the response for.
        t = pbp.pass_attempt.eq(1) & pbp.receiver_player_id.notna()
        air = pd.to_numeric(pbp.air_yards, errors="coerce")[t]
        rec = pd.DataFrame({
            "season": season, "player_id": pbp.receiver_player_id[t],
            "air": air, "yardline": yardline[t],
            "deep": air.ge(20).astype(float),
            "screen": air.le(0).astype(float),
            "intermediate": air.between(10, 19).astype(float),
            "inside10": yardline[t].le(10).astype(float),
        })
        agg = rec.groupby(["season", "player_id"]).agg(
            rec_plays=("air", "size"),
            rec_adot_pbp=("air", "mean"),
            rec_air_yards_sd=("air", "std"),
            rec_deep_share=("deep", "mean"),
            rec_screen_share=("screen", "mean"),
            rec_intermediate_share=("intermediate", "mean"),
            rec_yardline_mean=("yardline", "mean"),
            rec_inside10_share=("inside10", "mean"),
        ).reset_index()
        rec_rows.append(agg)
        del pbp

    rush = pd.concat(rush_rows, ignore_index=True)
    rec = pd.concat(rec_rows, ignore_index=True)
    out = rush.merge(rec, on=["season", "player_id"], how="outer")
    out["season"] = out.season + 1  # last season's usage, this season's row
    return out


def mean_mode(target: str) -> str:
    return PERSISTENCE_MEAN_MODE.get(target) or POSTERIOR_MEAN_MODE[target]


def control_columns(target: str) -> list[str]:
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


def residualise(y: pd.Series, controls: pd.DataFrame) -> np.ndarray:
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
    print("=== building situational usage from play-by-play ===")
    feats = situational_features()
    print(f"  {len(feats)} player-seasons, "
          f"{feats.rush_plays.notna().sum()} with carries, "
          f"{feats.rec_plays.notna().sum()} with targets")

    pr = pd.read_pickle(f"{CACHE}/player_rows.pkl")
    p = pr[pr.season.isin(SEASONS)].copy()
    p = p[pd.to_numeric(p.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]
    m = p.merge(feats, on=["season", "player_id"], how="left")

    print("\n=== does the pbp aDOT reproduce the one already in the cache? ===")
    both = m[["rec_adot_pbp", "prior_rec_air_yards_per_target"]].dropna()
    r, _ = stats.pearsonr(both.rec_adot_pbp, both.prior_rec_air_yards_per_target)
    print(f"  r={r:+.3f}  n={len(both)}   (a low number here means one of them is wrong)")

    print(f"\n{'=' * 104}\npersistence: is last season's usage context stable enough to use?\n{'=' * 104}")
    raw = feats.copy(); raw["season"] = raw.season - 1
    nxt = raw.copy(); nxt["season"] = nxt.season + 1
    j = raw.merge(nxt, on=["season", "player_id"], how="inner", suffixes=("", "_prior"))
    for name in RUSH_FEATURES + REC_FEATURES:
        floor = "rush_plays" if name.startswith("rush") else "rec_plays"
        pair = j[[name, f"{name}_prior", floor, f"{floor}_prior"]].dropna()
        pair = pair[pair[floor].ge(20) & pair[f"{floor}_prior"].ge(20)]
        if len(pair) < 40:
            print(f"  {name:28} too few pairs ({len(pair)})")
            continue
        r, _ = stats.pearsonr(pair[f"{name}_prior"], pair[name])
        print(f"  {name:28} r={r:+.3f}  r2={r*r:5.1%}  n={len(pair):>5}")

    for target, (group, signs) in HYPOTHESIS.items():
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        features = RUSH_FEATURES if group == "rush" else REC_FEATURES
        controls = [c for c in control_columns(target) if c in m.columns]
        y = pd.to_numeric(m[target], errors="coerce")
        q = pd.to_numeric(m[spec.prior_feature], errors="coerce")
        e = pd.to_numeric(m[spec.exposure], errors="coerce").fillna(0)
        keep = (
            y.notna() & q.notna() & e.ge(spec.min_exposure)
            & m["position"].astype(str).str.upper().isin(spec.positions)
            & m[list(features)].notna().any(axis=1)
        )
        sub = m[keep]
        if len(sub) < 60:
            print(f"\n{target}: only {len(sub)} covered rows, skipped")
            continue
        ys = y[keep]
        base = residualise(ys, sub[controls])
        print(f"\n{'=' * 104}")
        print(f"{target}   mode={mean_mode(target)}   n={len(sub)}   "
              f"controls={len(controls)}   "
              f"design explains {1 - base.var() / ys.to_numpy(float).var():5.1%}")
        print(f"{'=' * 104}")
        rows = []
        for name in features:
            x = pd.to_numeric(sub[name], errors="coerce")
            ok = x.notna().to_numpy()
            if ok.sum() < 60:
                continue
            r, pv = stats.pearsonr(x[ok], base[ok])
            want = signs.get(name, 0)
            rows.append({
                "feature": name, "n": int(ok.sum()), "r": r, "p": pv,
                "r2_add": r * r,
                "predicted": {1: "+", -1: "-", 0: ""}[want],
                "as_predicted": "yes" if want and np.sign(r) == want else
                                ("NO" if want else ""),
            })
        table = pd.DataFrame(rows).sort_values("r2_add", ascending=False)
        table["verdict"] = np.where((table.p < 0.01) & (table.r2_add > 0.01), "adds", "")
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
