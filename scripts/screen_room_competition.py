"""Role *change*, which is where the volume error actually is.

Split-half reliability on the volume layer is 96.7% for carry share and 93.0%
for target share. A season's observed role is very nearly noise-free, so the
model is not failing to measure last year's role -- it is failing to predict who
gains or loses one. That is a different target from anything the layer currently
reads, and twenty room and competition columns exist in the cache that reach no
volume or opportunity model at all.

Screened against the residual of this season's share after the model's own role
prior, which is the change the layer cannot currently see.

**Most of it is the position label wearing a costume.** Room competition is
computed within a position group, and the carry population is 237 quarterbacks
and 684 running backs, so ``prior_target_room_competition`` is 1.000 for every
quarterback and 0.537 for the average back -- an RB/QB indicator with a
continuous face. Against the role prior alone it looks like the largest effect
found anywhere in this line of work; against the role prior plus position
dummies it is nothing:

    feature                            vs prior   vs prior + position
    prior_target_room_competition        -0.222        -0.025  p=0.44
    prior_pass_room_competition          +0.219        +0.002  p=0.96
    prior_carry_room_competition         +0.123        -0.043  p=0.18
    prior_rush_room_quality_advantage    +0.093        +0.133  p=8.4e-05

One survives, and it strengthens rather than shrinks under the control, which is
the shape a real effect has when a confound was working against it.
``prior_rush_room_quality_advantage`` is the quality-weighted gap between a
player and the rest of his position room -- not how crowded the room is, but how
much better than it he was. That is a plausible mechanism for who keeps a role
and who loses one, and it is the one thing here worth a walk-forward.

The three that died are worth recording rather than deleting: they are the same
confound in a new costume, and the descriptive number was the largest in this
whole line of work right up until it was controlled properly.

    python scripts/screen_room_competition.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

CACHE = "/home/user/fantasy-football-research/.cache/ffmodel-2026"
SEASONS = range(2015, 2026)

STREAMS = {
    "carry_share_obs": ("prior_carry_role", "rush_att", "team_carries", 30),
    "target_share_obs": ("prior_target_role", "targets", "team_targets", 30),
}


def _residual(y: pd.Series, columns: pd.DataFrame) -> np.ndarray:
    x = columns.apply(pd.to_numeric, errors="coerce")
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
    p["team_targets"] = p.groupby(["season", "team"]).targets.transform("sum")
    p["team_carries"] = p.groupby(["season", "team"]).rush_att.transform("sum")
    p["target_share_obs"] = p.targets / p.team_targets.replace(0, np.nan)
    p["carry_share_obs"] = p.rush_att / p.team_carries.replace(0, np.nan)

    features = [
        c for c in p.columns
        if "room_competition" in c or "team_competition" in c
        or "room_quality_advantage" in c
    ]
    for response, (prior, exposure, _team, floor) in STREAMS.items():
        y = pd.to_numeric(p[response], errors="coerce")
        q = pd.to_numeric(p[prior], errors="coerce")
        e = pd.to_numeric(p[exposure], errors="coerce").fillna(0)
        keep = y.notna() & q.notna() & e.ge(floor)
        sub, ys = p[keep], y[keep]
        position = pd.get_dummies(sub["position"].astype(str), drop_first=True)
        bare = _residual(ys, sub[[prior]])
        controlled = _residual(ys, pd.concat([sub[[prior]], position.astype(float)], axis=1))
        print(f"\n{'=' * 84}\n{response}   n={len(sub)}   "
              f"prior explains {stats.pearsonr(q[keep], ys)[0] ** 2:.1%}\n{'=' * 84}")
        print(f"  {'position':10} {'mean room competition':>22}  n")
        for name, block in sub.groupby(sub["position"].astype(str)):
            value = pd.to_numeric(
                block.get("prior_target_room_competition"), errors="coerce"
            ).mean()
            print(f"  {name:10} {value:22.3f}  {len(block)}")
        rows = []
        for feature in features:
            x = pd.to_numeric(sub[feature], errors="coerce")
            ok = x.notna().to_numpy()
            if ok.sum() < 150:
                continue
            bare_r = stats.pearsonr(x[ok], bare[ok])[0]
            held_r, held_p = stats.pearsonr(x[ok], controlled[ok])
            rows.append({
                "feature": feature, "n": int(ok.sum()),
                "vs_prior": bare_r, "vs_prior_position": held_r, "p": held_p,
            })
        table = pd.DataFrame(rows).reindex(
            pd.DataFrame(rows).vs_prior_position.abs().sort_values(ascending=False).index
        )
        table["verdict"] = np.where(
            (table.p < 0.01) & (table.vs_prior_position.abs() > 0.05), "survives", ""
        )
        print()
        print(table.head(8).to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
