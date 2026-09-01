"""The blocking-isolating O-line metrics, which the last screen could not test.

The previous O-line screen failed on proxies rather than on the hypothesis. Team
rush yards over expected is computed from the team's own carries, so it conflates
the line with the back running behind it, and team sack rate is a QB statistic as
much as a blocking one -- a quarterback who holds the ball takes sacks behind a
good line. Both came back null and neither answered the question.

Pro Football Reference charts the split directly. Every carry is divided into
yards *before* contact and yards *after*, and the pre-contact half is the part
the line is responsible for: it is the hole, not the run. Same idea on the pass
side, where pressures allowed separate the protection from the sack, which is
partly the quarterback's.

    run blocking      team yards before contact per carry (PFR, weekly)
    pass protection   team pressure rate allowed (PFR, weekly)

ESPN's pass block and run block win rates are the metrics usually named here and
they are proprietary -- ESPN publishes the leaderboard, not the data, and no
nflverse feed carries them. Yards before contact is the closest public
substitute and is arguably the better one for this purpose anyway, since it is
measured in the units of the response.

Two constructions matter and both are here. Team yards before contact is built
from the team's own carries, so a feature workhorse back is most of his own
team's number; the leave-one-out column removes the focal player's carries from
his prior team's aggregate, which is the only version that is not partly a
restatement of his own prior. Quarterback carries are dropped throughout --
a scramble is nearly all yards before contact and would score a mobile
quarterback's team as a great run-blocking line.

Coverage is 2018 onward; PFR's charting does not go back further, which is
itself a constraint on any promotion built from it.

**Negative, and this time the chain is diagnosed rather than merely null.**
Yards before contact is the better metric it was expected to be -- it carries
8.8% of itself into the next season against team RYOE's 2.5% -- and it is not
the same number as the runner's half, which moves *against* it within a season
(ybc/att vs yac/att, r = -0.213: teams that block well have backs who create
less, or the split is chart noise). But the two links that would make it usable
both fail.

    mechanism      same-season team ybc/att vs a back's yds/carry   r2 = 5.6%
                   with his own carries removed (LOO)               r2 = 1.0%
    forecast       prior-season LOO ybc/att vs his residual         r = -0.014

Most of what looks like a team's blocking is the back running behind it: taking
the focal player's own carries out of his team's aggregate drops the
contemporaneous fit from 5.6% to 1.0%. That is the same confound that
disqualified team RYOE, and yards before contact does not escape it -- it only
makes it measurable. Whatever is left then has to survive a year, and 1.0% of
variance carried at 8.8% persistence is about 0.09%, which is below the noise
floor of anything this pipeline scores. The forward correlation is duly zero.

The three-year average, the standard repair for a noisy team metric, makes it
worse rather than better: 3.6% persistence against the single year's 8.8%. A
line's blocking is apparently a property of that season's five starters and that
season's scheme, not a franchise trait, so averaging over the churn adds stale
personnel instead of removing noise.

Pass protection ends the same way with a cleaner middle. Pressure rate allowed
persists at 9.9% and is the one metric here whose same-season sign is right and
significant -- more pressure, fewer yards per attempt, r = -0.164 -- but the
prior season's rate says nothing about this season's residual (r = +0.026,
p = 0.66), and the receiving response does not see it at all (r = -0.001).

On the metric the question named: ESPN's pass block and run block win rates are
proprietary and no nflverse feed carries them, so they cannot be tested here.
Yards before contact is the public substitute and it has now been tested. FTN's
``is_qb_fault_sack`` is the one unexplored crumb -- it would separate the sack
the line allowed from the one the quarterback caused -- but FTN charting starts
in 2022, which leaves too few training seasons to build a prior-season feature
for a 2023 holdout.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

from ffmodel.data import ingest

pd.set_option("display.width", 220)
CACHE = "/home/user/fantasy-football-research/.cache/ffmodel-2026"
SEASONS = list(range(2018, 2026))  # PFR advanced charting starts at 2018


def _positions() -> pd.DataFrame:
    ids = ingest.load_ids()
    out = ids[["pfr_id", "position"]].dropna(subset=["pfr_id"]).copy()
    return out.drop_duplicates("pfr_id").rename(columns={"pfr_id": "pfr_player_id"})


def run_block_frames():
    """Team and player yards-before-contact per carry, quarterbacks excluded."""
    wk = ingest.load_pfr_advstats(SEASONS, stat_type="rush", summary_level="week")
    wk = wk[wk.game_type.astype(str).eq("REG")].copy()
    wk = wk.merge(_positions(), on="pfr_player_id", how="left")
    before = len(wk)
    wk = wk[wk.position.ne("QB")]
    print(f"  {before - len(wk)} of {before} weekly rush rows dropped as QB carries")
    wk["carries"] = pd.to_numeric(wk.carries, errors="coerce").fillna(0.0)
    wk["ybc"] = pd.to_numeric(wk.rushing_yards_before_contact, errors="coerce").fillna(0.0)
    wk["yac"] = pd.to_numeric(wk.rushing_yards_after_contact, errors="coerce").fillna(0.0)

    team = wk.groupby(["season", "team"], as_index=False)[["carries", "ybc", "yac"]].sum()
    team = team[team.carries.ge(100)]
    team["team_ybc_att"] = team.ybc / team.carries
    team["team_yac_att"] = team.yac / team.carries
    team["team_ypc"] = (team.ybc + team.yac) / team.carries

    player = wk.groupby(
        ["season", "team", "pfr_player_id", "pfr_player_name"], as_index=False
    )[["carries", "ybc", "yac"]].sum()
    return team, player


def leave_one_out(team: pd.DataFrame, player: pd.DataFrame) -> pd.DataFrame:
    """Team blocking with the focal player's own carries removed."""
    m = player.merge(
        team[["season", "team", "carries", "ybc"]].rename(
            columns={"carries": "team_carries", "ybc": "team_ybc"}
        ),
        on=["season", "team"], how="inner",
    )
    rest_carries = m.team_carries - m.carries
    m["loo_ybc_att"] = np.where(
        rest_carries.ge(50), (m.team_ybc - m.ybc) / rest_carries.replace(0, np.nan), np.nan
    )
    return m


def pass_protection() -> pd.DataFrame:
    wk = ingest.load_pfr_advstats(SEASONS, stat_type="pass", summary_level="week")
    wk = wk[wk.game_type.astype(str).eq("REG")].copy()
    for col in ("times_pressured", "times_sacked", "times_hurried", "times_blitzed"):
        wk[col] = pd.to_numeric(wk[col], errors="coerce").fillna(0.0)
    # times_pressured_pct is per-quarterback; recover the denominator so the team
    # number is dropback-weighted rather than an average of quarterbacks.
    # times_pressured_pct is charted as a fraction, not a percentage: Mahomes
    # at 8 pressures reads 0.178, so the denominator is times_pressured / pct.
    pct = pd.to_numeric(wk.times_pressured_pct, errors="coerce")
    wk["dropbacks"] = np.where(pct.gt(0), wk.times_pressured / pct, np.nan)
    team = wk.groupby(["season", "team"], as_index=False)[
        ["times_pressured", "times_sacked", "dropbacks"]
    ].sum()
    team = team[team.dropbacks.ge(200)]
    team["team_pressure_rate"] = team.times_pressured / team.dropbacks
    team["team_sack_rate_pfr"] = team.times_sacked / team.dropbacks
    return team


def persistence(frame: pd.DataFrame, col: str, label: str, key=("season", "team")):
    cur = frame[[*key, col]].copy()
    nxt = cur.copy()
    nxt["season"] = nxt.season + 1
    j = cur.merge(nxt, on=list(key), how="inner", suffixes=("", "_prior")).dropna()
    if len(j) < 30:
        print(f"  {label:34} too few pairs ({len(j)})")
        return np.nan
    r, pv = stats.pearsonr(j[f"{col}_prior"], j[col])
    v = j[col]
    print(
        f"  {label:34} r={r:+.3f}  r2={r*r:6.1%}  p={pv:.3g}  n={len(j):>4}   "
        f"mean {v.mean():.3f}  sd {v.std():.3f}  cv {v.std()/abs(v.mean()):.1%}"
    )
    return r


def main() -> int:
    pr = pd.read_pickle(f"{CACHE}/player_rows.pkl")
    p = pr[pr.season.isin(SEASONS)].copy()
    p = p[pd.to_numeric(p.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)]

    print("=== building the blocking metrics ===")
    team, player = run_block_frames()
    loo = leave_one_out(team, player)
    protect = pass_protection()
    print(f"  team-season rushing rows {len(team)}, passing rows {len(protect)}")

    print("\n=== how much of a carry is the line's half, and does it persist? ===")
    print("   (the gate: team RYOE carried 2.5% of next year, sack rate 13.3%)")
    for col, lab in (
        ("team_ybc_att", "yards before contact / carry"),
        ("team_yac_att", "yards after contact / carry"),
        ("team_ypc", "team yards / carry"),
    ):
        persistence(team, col, lab)
    for col, lab in (
        ("team_pressure_rate", "pressure rate allowed"),
        ("team_sack_rate_pfr", "sack rate allowed (PFR)"),
    ):
        persistence(protect, col, lab)

    print("\n=== is the split real, or does one half carry the other? ===")
    j = team.dropna(subset=["team_ybc_att", "team_yac_att"])
    r, pv = stats.pearsonr(j.team_ybc_att, j.team_yac_att)
    print(f"  ybc/att vs yac/att within a team-season   r={r:+.3f}  p={pv:.3g}  n={len(j)}")
    for half in ("team_ybc_att", "team_yac_att"):
        r, pv = stats.pearsonr(j[half], j.team_ypc)
        print(f"  {half:16} vs team yds/carry        r={r:+.3f}  p={pv:.3g}")

    print("\n=== does prior-season blocking predict the player's efficiency residual? ===")
    prior_team = team.copy()
    prior_team["season"] = prior_team.season + 1
    prior_team = prior_team[["season", "team", "team_ybc_att", "team_yac_att"]]
    prior_protect = protect.copy()
    prior_protect["season"] = prior_protect.season + 1
    prior_protect = prior_protect[["season", "team", "team_pressure_rate"]]

    # leave-one-out: the focal player's own prior-season carries removed from the
    # prior-season aggregate of whichever team he plays for now.
    gsis = ingest.load_ids()[["gsis_id", "pfr_id"]].dropna()
    loo_key = loo.merge(
        gsis.rename(columns={"pfr_id": "pfr_player_id"}), on="pfr_player_id", how="left"
    )
    loo_key = loo_key.dropna(subset=["gsis_id"])
    loo_key["season"] = loo_key.season + 1
    loo_key = loo_key[["season", "team", "gsis_id", "loo_ybc_att"]].rename(
        columns={"gsis_id": "player_id"}
    )

    m = p.merge(prior_team, on=["season", "team"], how="left")
    m = m.merge(prior_protect, on=["season", "team"], how="left")
    m = m.merge(loo_key, on=["season", "team", "player_id"], how="left")
    # A player who was not on this team last year has no carries to remove.
    m["loo_ybc_att"] = m.loo_ybc_att.fillna(m.team_ybc_att)

    for ycol, pcol, xcol, expo, floor, lab in (
        ("rush_yards_per_carry", "prior_rush_yards_per_carry", "team_ybc_att",
         "rush_att", 25, "yds/carry ~ team ybc/att"),
        ("rush_yards_per_carry", "prior_rush_yards_per_carry", "loo_ybc_att",
         "rush_att", 25, "yds/carry ~ ybc/att (LOO)"),
        ("rush_yards_per_carry", "prior_rush_yards_per_carry", "team_yac_att",
         "rush_att", 25, "yds/carry ~ team yac/att"),
        ("pass_yards_per_attempt", "prior_pass_yards_per_attempt", "team_pressure_rate",
         "pass_att", 50, "yds/att ~ pressure rate"),
        ("rec_yards_per_target", "prior_rec_yards_per_target", "team_pressure_rate",
         "targets", 25, "yds/target ~ pressure rate"),
    ):
        o = pd.to_numeric(m[ycol], errors="coerce")
        q = pd.to_numeric(m[pcol], errors="coerce")
        e = pd.to_numeric(m[expo], errors="coerce").fillna(0)
        x = pd.to_numeric(m[xcol], errors="coerce")
        ok = o.gt(0.1) & q.gt(0.1) & e.ge(floor) & x.notna()
        if ok.sum() < 50:
            print(f"  {lab:30} too few rows ({int(ok.sum())})")
            continue
        resid = np.log(o[ok] / q[ok])
        r, pv = stats.pearsonr(x[ok], resid)
        print(
            f"  {lab:30} r={r:+.3f}  p={pv:.3g}  n={int(ok.sum()):>5}   "
            f"1sd moves residual {r*resid.std()*100:+.1f}%   "
            f"(residual sd {resid.std()*100:.0f}%)"
        )

    print("\n=== the level, not the residual: does blocking predict the rate itself? ===")
    for ycol, xcol, expo, floor, lab in (
        ("rush_yards_per_carry", "team_ybc_att", "rush_att", 25, "yds/carry ~ team ybc/att"),
        ("rush_yards_per_carry", "loo_ybc_att", "rush_att", 25, "yds/carry ~ ybc/att (LOO)"),
        ("pass_yards_per_attempt", "team_pressure_rate", "pass_att", 50, "yds/att ~ pressure rate"),
    ):
        o = pd.to_numeric(m[ycol], errors="coerce")
        e = pd.to_numeric(m[expo], errors="coerce").fillna(0)
        x = pd.to_numeric(m[xcol], errors="coerce")
        ok = o.gt(0.1) & e.ge(floor) & x.notna()
        if ok.sum() < 50:
            print(f"  {lab:30} too few rows ({int(ok.sum())})")
            continue
        r, pv = stats.pearsonr(x[ok], o[ok])
        print(f"  {lab:30} r={r:+.3f}  p={pv:.3g}  n={int(ok.sum()):>5}")

    print("\n=== is there a mechanism at all? same-season blocking vs the rate ===")
    print("   (a null here kills the idea; a hit here with a null above means the")
    print("    line is real but not forecastable from last year)")
    same_team = team[["season", "team", "team_ybc_att", "team_yac_att"]]
    same_loo = loo.merge(
        gsis.rename(columns={"pfr_id": "pfr_player_id"}), on="pfr_player_id", how="left"
    ).dropna(subset=["gsis_id"])
    same_loo = same_loo[["season", "team", "gsis_id", "loo_ybc_att"]].rename(
        columns={"gsis_id": "player_id", "loo_ybc_att": "same_loo_ybc_att"}
    )
    same_protect = protect[["season", "team", "team_pressure_rate"]].rename(
        columns={"team_pressure_rate": "same_pressure_rate"}
    )
    c = p.merge(same_team.rename(columns={
        "team_ybc_att": "same_ybc_att", "team_yac_att": "same_yac_att"
    }), on=["season", "team"], how="left")
    c = c.merge(same_loo, on=["season", "team", "player_id"], how="left")
    c = c.merge(same_protect, on=["season", "team"], how="left")
    for ycol, xcol, expo, floor, lab in (
        ("rush_yards_per_carry", "same_ybc_att", "rush_att", 25, "yds/carry ~ same-yr ybc/att"),
        ("rush_yards_per_carry", "same_loo_ybc_att", "rush_att", 25, "yds/carry ~ same-yr LOO ybc"),
        ("rush_yards_per_carry", "same_yac_att", "rush_att", 25, "yds/carry ~ same-yr yac/att"),
        ("pass_yards_per_attempt", "same_pressure_rate", "pass_att", 50, "yds/att ~ same-yr pressure"),
    ):
        o = pd.to_numeric(c[ycol], errors="coerce")
        e = pd.to_numeric(c[expo], errors="coerce").fillna(0)
        x = pd.to_numeric(c[xcol], errors="coerce")
        ok = o.gt(0.1) & e.ge(floor) & x.notna()
        if ok.sum() < 50:
            print(f"  {lab:32} too few rows ({int(ok.sum())})")
            continue
        r, pv = stats.pearsonr(x[ok], o[ok])
        print(f"  {lab:32} r={r:+.3f}  r2={r*r:5.1%}  p={pv:.3g}  n={int(ok.sum()):>5}")

    print("\n=== a three-year average, the usual fix for a noisy team metric ===")
    tr3 = []
    for season in sorted(team.season.unique()):
        window = team[team.season.between(season - 2, season)]
        agg = window.groupby("team", as_index=False)[["carries", "ybc"]].sum()
        agg["ybc_att_3yr"] = agg.ybc / agg.carries
        agg["season"] = season
        agg["seasons_in_window"] = window.groupby("team").season.nunique().reindex(
            agg.team
        ).to_numpy()
        tr3.append(agg[agg.seasons_in_window.ge(3)][["season", "team", "ybc_att_3yr"]])
    tr3 = pd.concat(tr3, ignore_index=True)
    nxt = team[["season", "team", "team_ybc_att"]].copy()
    nxt["season"] = nxt.season - 1  # next season's single-year value
    j = tr3.merge(nxt, on=["season", "team"], how="inner").dropna()
    if len(j) >= 30:
        r, pv = stats.pearsonr(j.ybc_att_3yr, j.team_ybc_att)
        print(f"  3yr ybc/att -> next-year ybc/att        r={r:+.3f}  r2={r*r:5.1%}  "
              f"p={pv:.3g}  n={len(j)}")
    fwd = tr3.copy()
    fwd["season"] = fwd.season + 1
    m3 = p.merge(fwd, on=["season", "team"], how="left")
    o = pd.to_numeric(m3.rush_yards_per_carry, errors="coerce")
    q = pd.to_numeric(m3.prior_rush_yards_per_carry, errors="coerce")
    e = pd.to_numeric(m3.rush_att, errors="coerce").fillna(0)
    x = pd.to_numeric(m3.ybc_att_3yr, errors="coerce")
    ok = o.gt(0.1) & q.gt(0.1) & e.ge(25) & x.notna()
    if ok.sum() >= 50:
        resid = np.log(o[ok] / q[ok])
        r, pv = stats.pearsonr(x[ok], resid)
        print(f"  3yr ybc/att -> yds/carry residual       r={r:+.3f}  p={pv:.3g}  "
              f"n={int(ok.sum())}   1sd moves residual {r*resid.std()*100:+.1f}%")
    ok2 = o.gt(0.1) & e.ge(25) & x.notna()
    if ok2.sum() >= 50:
        r, pv = stats.pearsonr(x[ok2], o[ok2])
        print(f"  3yr ybc/att -> yds/carry level          r={r:+.3f}  p={pv:.3g}  "
              f"n={int(ok2.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
