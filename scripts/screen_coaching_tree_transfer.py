"""Does an incoming play-caller carry his previous offense's shape with him?

This is the coaching-*tree* question that scripts/screen_zone_teammate_coach.py
had to leave open ("it needs the scheme-lineage tables, whose scraper cannot
run here"). The tables exist now.

The churn framing -- does a coaching change widen role dispersion -- is tested
in scripts/screen_coaching_role_churn.py and comes back null. This is the
sharper framing, and the one the lineage table was built for: a scheme carrier
arriving from somewhere else brings tendencies, and those tendencies are a
*direction*, not just added variance.

Three team-season shapes, each about how volume gets distributed rather than
how much of it there is:

  rb_target_share    share of team targets going to running backs -- the
                     checkdown-vs-downfield signature
  target_hhi         Herfindahl concentration of target share over the roster
                     -- does this scheme feed an alpha or spread it around
  rush_rate          carries / (carries + pass attempts)

For each response team-season, the predictor is the scheme coach's own prior
NFL stops (from scheme_lineage), restricted to roles that plausibly own an
offense and averaged with recency weight. The control that matters is the
*team's own* prior-season value of the same shape: teams persist, and a coach
who inherits a run-heavy roster will look run-heavy whether or not he brought
it with him. The question is only what his history adds beyond that.

Leakage-safe by construction: build_scheme_lineage restricts every stop to
prior_season < season, and the team controls are seasons -1, -2 and -3.

Two checks decide whether the surviving result is real, and both are run here
rather than left to the reader:

``--external-only`` (the default) drops stops at the franchise the coach now
works for. Without it the test is partly mechanical: a coach who *stayed* has
his own prior seasons at this same team inside his lineage, so his "carried"
shape is partly just the team's own history under another name. The effect
gets *stronger* under this restriction (+0.163 -> +0.204), which is the
opposite of what a self-correlation artifact would do.

The recency half-life and the role filter are both free parameters, so both
are swept. The result barely moves across either -- half-lives from 2 years to
flat give +0.189 to +0.210, and the recency weighting being nearly irrelevant
is itself the finding that a play-caller's back-usage is a stable career trait
rather than recent form.

    python scripts/screen_coaching_tree_transfer.py
    python scripts/screen_coaching_tree_transfer.py --sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffmodel.data.coaching import load_scheme_lineage  # noqa: E402

# Stops where the coach plausibly shaped the passing/running distribution.
# A quality-control or position coach does not choose the offense.
SCHEME_ROLES = ("offensive coordinator", "head coach")
RECENCY_HALF_LIFE = 6.0


def team_shapes(rows: pd.DataFrame) -> pd.DataFrame:
    """Per team-season distribution shapes, from observed volume."""
    d = rows[
        pd.to_numeric(rows.get("is_replacement_player"), errors="coerce").fillna(0).ne(1)
    ].copy()
    for column in ("targets", "rush_att", "pass_att"):
        d[column] = pd.to_numeric(d.get(column), errors="coerce").fillna(0.0)
    grouped = d.groupby(["season", "team"])
    out = grouped.agg(
        team_targets=("targets", "sum"),
        team_carries=("rush_att", "sum"),
        team_pass_att=("pass_att", "sum"),
    ).reset_index()
    rb = d[d.position.eq("RB")].groupby(["season", "team"])["targets"].sum().rename("rb_targets")
    out = out.merge(rb.reset_index(), on=["season", "team"], how="left")
    out["rb_targets"] = out["rb_targets"].fillna(0.0)
    out["rb_target_share"] = out["rb_targets"] / out["team_targets"].where(out["team_targets"] > 0)
    share = d["targets"] / grouped["targets"].transform("sum").where(lambda s: s > 0)
    out = out.merge(
        (share**2).groupby([d.season, d.team]).sum().rename("target_hhi").reset_index(),
        on=["season", "team"], how="left",
    )
    denominator = out["team_carries"] + out["team_pass_att"]
    out["rush_rate"] = out["team_carries"] / denominator.where(denominator > 0)
    return out[["season", "team", "rb_target_share", "target_hhi", "rush_rate"]]


def sweep(argv_cache_dir: Path) -> int:
    """Half-life and role-filter sensitivity for the one surviving shape."""
    print("rb_target_share transfer, external stops only, 3-season team control\n")
    print("half-life sweep (roles: offensive coordinator + head coach)")
    for half_life in (2.0, 4.0, 6.0, 10.0, 20.0, 1e6):
        label = "flat" if half_life > 1e5 else f"{half_life:g}y"
        main([
            "--cache-dir", str(argv_cache_dir),
            "--half-life", str(half_life),
            "--only", "rb_target_share",
            "--label", f"  half-life {label}",
        ])
    print("\nrole-filter sweep (half-life 6y)")
    for roles, label in (
        (("offensive coordinator",), "  OC only"),
        (("offensive coordinator", "head coach"), "  OC + HC"),
        (("offensive coordinator", "head coach", "quarterbacks coach"), "  OC + HC + QB coach"),
    ):
        main([
            "--cache-dir", str(argv_cache_dir),
            "--only", "rb_target_share",
            "--roles", *roles,
            "--label", label,
        ])
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ffmodel-walkforward"))
    parser.add_argument(
        "--include-own-team-stops",
        action="store_true",
        help="keep stops at the franchise the coach now works for; makes the "
             "test partly mechanical for a coach who stayed (see docstring)",
    )
    parser.add_argument(
        "--half-life", type=float, default=RECENCY_HALF_LIFE,
        help="exponential recency weight on prior stops, in years",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="sweep the half-life and role filter instead of one fit",
    )
    parser.add_argument(
        "--roles", nargs="*", default=list(SCHEME_ROLES),
        help="prior-stop roles that count as owning the offense",
    )
    parser.add_argument("--only", help="report just this one shape")
    parser.add_argument("--label", help="row label, for swept output")
    args = parser.parse_args(argv)
    if args.sweep:
        return sweep(args.cache_dir)

    rows = pd.read_pickle(args.cache_dir / "player_rows.pkl")
    shapes = team_shapes(rows)
    metrics = [args.only] if args.only else ["rb_target_share", "target_hhi", "rush_rate"]

    lineage = load_scheme_lineage()
    role = lineage["prior_role"].astype(str).str.lower()
    lineage = lineage[
        role.str.contains("|".join(args.roles), na=False)
        & ~role.str.contains("assistant|quality control|intern", na=False)
    ].copy()

    # Attach the shape of each prior stop, then average per response
    # team-season with an exponential recency weight.
    stops = lineage.merge(
        shapes.rename(columns={"season": "prior_season", "team": "prior_team_code"}),
        on=["prior_season", "prior_team_code"], how="inner",
    )
    if not args.include_own_team_stops:
        stops = stops[stops["prior_team_code"] != stops["franchise_code"]]
    stops["_w"] = 0.5 ** (stops["recency_years"].astype(float) / args.half_life)
    carried = []
    for metric in metrics:
        values = pd.to_numeric(stops[metric], errors="coerce")
        ok = values.notna()
        block = stops[ok].assign(_v=values[ok] * stops.loc[ok, "_w"])
        agg = block.groupby(["season", "franchise_code"]).agg(
            _num=("_v", "sum"), _den=("_w", "sum"), _stops=("_v", "size")
        ).reset_index()
        agg[f"coach_{metric}"] = agg["_num"] / agg["_den"].where(agg["_den"] > 0)
        carried.append(agg[["season", "franchise_code", f"coach_{metric}", "_stops"]]
                       .rename(columns={"_stops": f"stops_{metric}"}))
    coach = carried[0]
    for extra in carried[1:]:
        coach = coach.merge(extra, on=["season", "franchise_code"], how="outer")

    panel = shapes.rename(columns={"team": "franchise_code"}).merge(
        coach, on=["season", "franchise_code"], how="inner"
    )
    # Three seasons of team history, not one: a team's distribution shape
    # persists well beyond a single year, and a one-season control leaves
    # enough of that persistence in the residual to flatter the coach term.
    for lag in (1, 2, 3):
        # Only the shapes under test: with --only, carrying the other shape
        # columns collides with the ones already on the panel.
        lagged = shapes.rename(columns={"team": "franchise_code"})[
            ["season", "franchise_code", *metrics]
        ].copy()
        lagged["season"] = lagged["season"] + lag
        panel = panel.merge(
            lagged.rename(columns={m: f"team_prior{lag}_{m}" for m in metrics}),
            on=["season", "franchise_code"], how="left",
        )

    if args.label is None:
        stop_column = f"stops_{metrics[0]}"
        print(f"{len(panel)} team-seasons with a scheme-carrier lineage and prior team seasons")
        print(f"seasons {panel.season.min()}-{panel.season.max()}, "
              f"median prior stops used: {panel[stop_column].median():.0f}\n")
        print(f"{'shape':<18}{'n':>5}{'raw r':>9}{'p':>10}"
              f"{'partial r':>12}{'p':>10}   (partial = beyond team seasons -1, -2, -3)")
    for metric in metrics:
        control_columns = [f"team_prior{lag}_{metric}" for lag in (1, 2, 3)]
        block = panel[[metric, f"coach_{metric}", *control_columns]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(block) < 50:
            print(f"{metric:<18}{len(block):>5}{'too few':>9}")
            continue
        y = block[metric].to_numpy(float)
        x = block[f"coach_{metric}"].to_numpy(float)
        control = block[control_columns].to_numpy(float)
        raw = float(np.corrcoef(x, y)[0, 1])
        n = len(block)
        z = np.arctanh(raw) * np.sqrt(max(n - 3, 1))
        p_raw = 2 * stats.norm.sf(abs(z))
        design = np.column_stack([control, np.ones(n)])  # noqa: E501
        xr = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        yr = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        part = float(np.corrcoef(xr, yr)[0, 1])
        zp = np.arctanh(part) * np.sqrt(max(n - 3 - design.shape[1], 1))
        p_part = 2 * stats.norm.sf(abs(zp))
        name = args.label if args.label is not None else metric
        print(f"{name:<24}{n:>5}{raw:>+9.4f}{p_raw:>10.2e}{part:>+12.4f}{p_part:>10.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
