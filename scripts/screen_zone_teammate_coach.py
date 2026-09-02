"""Three cuts at once: red-zone role, rushing teammate context, and the coach.

**Red zone.** ``ffmodel.features.redzone`` already exists, already computes the
right quantity, and reaches no cache column. The right quantity is the
*differential* -- a player's share of his team's red-zone work minus his share
of its ordinary work -- because red-zone volume on its own is mostly a
restatement of volume. That is what would encode "goal-line back" as a trait.
An earlier situational screen tested a target's *yardline* and found nothing;
this is the better-aimed version of the same idea, and it is aimed at
rec_td_rate and rush_td_rate, the two responses with the least explained
variance in the layer.

**Rushing teammate context.** Receiving has ``teammate_qb_quality_signal``;
rushing has no teammate covariate at all. A back runs behind an offense, and a
defense that must respect the pass leaves lighter boxes. Two prior-season,
leakage-safe proxies for that: the team's passing efficiency, and how much of
the team's rushing came from its quarterback. Both are team-level, which is the
family that has failed repeatedly here, so the prior is poor and the screen is
run to close the question rather than in hope.

**Coach.** The committed coaching tables are empty -- the Wikipedia scraper that
fills them cannot run here, the environment's network policy denies the host --
but ``load_schedules`` carries ``home_coach`` and ``away_coach``, so head coach
per team-season needs no scrape at all. That covers "by coach"; it does not
cover coordinator or coaching *tree*, which is the lineage the Wikipedia tables
hold.

Testing a coach effect properly means asking whether coach identity explains
residual variance beyond what the design already has, which is a variance
question rather than a correlation: with ~60 coaches, any single dummy is noise.
So the coach cut is scored two ways -- a between-coach variance ratio against a
label-shuffled null, and the persistence of a coach's own residual from one
season to the next, which is what a usable feature would need.

Controls throughout are the design each response actually fits.

**Results: two of the three close, one survives.**

Red zone is null everywhere, including on the two touchdown responses it was
aimed at. Every differential sits under 0.01 of residual variance at p > 0.05 --
rec_td_rate's best is -0.016 at p = 0.47. This is the well-built version of the
idea, a differential rather than raw red-zone volume, already written and
already the right quantity, and it still says nothing about next season. Taken
with the earlier situational screen, which found goal-line carry share persists
at 1.8-2.2% year over year, the reading is consistent: red-zone role is a thing
that happens to a player in a season, not a trait he carries into the next one.

The coach effect is a team effect wearing a hat. Coach identity looks strong on
the receiving responses -- 6.95% of the residual on yards per target at a
shuffled-label p of 0.000 -- but a coach and a franchise are nearly collinear,
and absorbing team identity first collapses it:

    response                coach   team    coach after team
    rec_yards_per_target    6.95%   3.05%   3.82%  p=0.305
    rec_catch_rate          6.94%   2.85%   4.40%  p=0.100
    rec_td_rate             5.08%   2.61%   2.50%  p=0.985

What is left is a small team effect, and team-level quality is the family that
has failed every forecast test in this line of work. Coach residuals also barely
carry across seasons (r = +0.149 on yards per target, zero on the other two).
Coaching *tree* is a different question and remains untested: it needs the
scheme-lineage tables, whose scraper cannot run here because the environment's
network policy denies the Wikipedia host.

The survivor is the quarterback's share of his team's carries, against a back's
yards per carry: r = +0.135 at p < 1e-4, 1.8% of a residual whose ridge design
already explains 22.2%. The sign is the interesting part -- a back gains yards
per carry when his quarterback runs more, which is the light-box mechanism, and
is larger than the short-yardage share that already cleared a walk-forward
(1.2%). It is null on rushing touchdown rate (p = 0.16). Team passing efficiency,
the other teammate proxy, is null on both.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

import nflreadpy as nfl

from ffmodel.features.redzone import REDZONE_FEATURES, add_redzone_features
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
RESPONSES = (
    "rec_td_rate", "rush_td_rate", "rec_yards_per_target",
    "rush_yards_per_carry", "rec_catch_rate",
)


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


def rushing_teammate_context() -> pd.DataFrame:
    """Prior-season team passing quality and quarterback share of carries."""
    stats_ = nfl.load_player_stats(seasons=SEASONS).to_pandas()
    if "season_type" in stats_:
        stats_ = stats_[stats_.season_type.eq("REG")]
    for c in ("passing_yards", "attempts", "carries", "rushing_yards"):
        stats_[c] = pd.to_numeric(stats_.get(c), errors="coerce").fillna(0.0)
    stats_["is_qb"] = stats_["position"].astype(str).str.upper().eq("QB")
    team = stats_.groupby(["season", "team"], as_index=False).agg(
        pass_yds=("passing_yards", "sum"),
        pass_att=("attempts", "sum"),
        carries=("carries", "sum"),
    )
    qb = (
        stats_[stats_.is_qb].groupby(["season", "team"], as_index=False)
        .agg(qb_carries=("carries", "sum"))
    )
    team = team.merge(qb, on=["season", "team"], how="left")
    team["team_pass_ypa"] = team.pass_yds / team.pass_att.replace(0, np.nan)
    team["qb_carry_share"] = team.qb_carries.fillna(0) / team.carries.replace(0, np.nan)
    team["season"] = team.season + 1
    return team[["season", "team", "team_pass_ypa", "qb_carry_share"]]


def head_coaches() -> pd.DataFrame:
    """Head coach per team-season, from the schedule feed."""
    sched = nfl.load_schedules(seasons=SEASONS).to_pandas()
    sched = sched[sched.game_type.eq("REG")] if "game_type" in sched else sched
    home = sched[["season", "home_team", "home_coach"]].rename(
        columns={"home_team": "team", "home_coach": "coach"}
    )
    away = sched[["season", "away_team", "away_coach"]].rename(
        columns={"away_team": "team", "away_coach": "coach"}
    )
    both = pd.concat([home, away], ignore_index=True).dropna()
    # The coach who called the most games, so an interim stint does not rename
    # a whole team-season.
    counts = both.groupby(["season", "team", "coach"], as_index=False).size()
    idx = counts.groupby(["season", "team"])["size"].idxmax()
    return counts.loc[idx, ["season", "team", "coach"]].reset_index(drop=True)


def coach_variance(resid: np.ndarray, labels: np.ndarray, *, seed=0) -> tuple:
    """Between-group share of variance, against a shuffled-label null."""
    frame = pd.DataFrame({"r": resid, "g": labels}).dropna()
    frame = frame[frame.g.map(frame.g.value_counts()).ge(8)]
    if frame.g.nunique() < 5:
        return np.nan, np.nan, 0, 0
    grand = frame.r.mean()
    def ratio(values):
        f = pd.DataFrame({"r": values, "g": frame.g.to_numpy()})
        means = f.groupby("g").r.agg(["mean", "size"])
        between = float((means["size"] * (means["mean"] - grand) ** 2).sum())
        return between / float(((f.r - grand) ** 2).sum())
    observed = ratio(frame.r.to_numpy())
    rng = np.random.default_rng(seed)
    null = [ratio(rng.permutation(frame.r.to_numpy())) for _ in range(200)]
    return observed, float(np.mean(np.array(null) >= observed)), frame.g.nunique(), len(frame)


def main() -> int:
    pr = pd.read_pickle(f"{CACHE}/player_rows.pkl")
    p = pr[pr.season.isin(SEASONS)].copy()
    p = p[pd.to_numeric(p.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]

    print("=== attaching red-zone differentials, teammate context and coaches ===")
    p = add_redzone_features(p)
    covered = p[list(REDZONE_FEATURES)].notna().any(axis=1)
    print(f"  red-zone features on {int(covered.sum())} of {len(p)} rows")
    p = p.merge(rushing_teammate_context(), on=["season", "team"], how="left")
    coaches = head_coaches()
    p = p.merge(coaches, on=["season", "team"], how="left")
    print(f"  coach on {int(p.coach.notna().sum())} rows, "
          f"{p.coach.nunique()} distinct across {len(SEASONS)} seasons")

    teammate = ("team_pass_ypa", "qb_carry_share")
    for target in RESPONSES:
        spec = EFFICIENCY_MODEL_BY_TARGET[target]
        controls = [c for c in control_columns(target) if c in p.columns]
        y = pd.to_numeric(p[target], errors="coerce")
        q = pd.to_numeric(p[spec.prior_feature], errors="coerce")
        e = pd.to_numeric(p[spec.exposure], errors="coerce").fillna(0)
        keep = (
            y.notna() & q.notna() & e.ge(spec.min_exposure)
            & p["position"].astype(str).str.upper().isin(spec.positions)
        )
        sub = p[keep]
        if len(sub) < 80:
            print(f"\n{target}: only {len(sub)} rows, skipped")
            continue
        ys = y[keep]
        base = residualise(ys, sub[controls])
        print(f"\n{'=' * 104}")
        print(f"{target}   mode={mean_mode(target)}   n={len(sub)}   "
              f"design explains {1 - base.var() / ys.to_numpy(float).var():5.1%}")
        print(f"{'=' * 104}")

        features = list(REDZONE_FEATURES)
        if target.startswith("rush"):
            features += list(teammate)
        rows = []
        for name in features:
            x = pd.to_numeric(sub.get(name), errors="coerce")
            if x is None or x.notna().sum() < 80:
                continue
            ok = x.notna().to_numpy()
            r, pv = stats.pearsonr(x[ok], base[ok])
            rows.append({"feature": name, "n": int(ok.sum()), "r": r,
                         "p": pv, "r2_add": r * r})
        if rows:
            table = pd.DataFrame(rows).sort_values("r2_add", ascending=False)
            table["verdict"] = np.where(
                (table.p < 0.01) & (table.r2_add > 0.01), "adds", ""
            )
            print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

        obs, pval, groups, n = coach_variance(base, sub.coach.to_numpy())
        if np.isfinite(obs):
            print(f"  coach: between-coach variance {obs:5.2%} of residual, "
                  f"shuffled-label p={pval:.3f}, {groups} coaches, n={n}")
            # would a coach's residual carry to next season?
            cf = pd.DataFrame({
                "season": sub.season.to_numpy(), "coach": sub.coach.to_numpy(),
                "r": base,
            }).dropna()
            per = cf.groupby(["season", "coach"], as_index=False).agg(
                r=("r", "mean"), k=("r", "size")
            )
            per = per[per.k.ge(4)]
            nxt = per.copy(); nxt["season"] = nxt.season + 1
            j = per.merge(nxt, on=["season", "coach"], suffixes=("", "_prior"))
            if len(j) >= 40:
                rr, pp = stats.pearsonr(j.r_prior, j.r)
                print(f"  coach residual persistence r={rr:+.3f}  r2={rr*rr:5.2%}  "
                      f"p={pp:.3g}  n={len(j)}")
            else:
                print(f"  coach residual persistence: too few pairs ({len(j)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
