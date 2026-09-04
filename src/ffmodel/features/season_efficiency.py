"""Leak-free player-season efficiency responses and lagged priors.

Efficiency is aggregated as a ratio of season totals, never as an average of
weekly ratios.  Each observed player rate is partially pooled toward the
same-season position mean using an opportunity-equivalent prior.  The pooled
estimate from season ``Y`` is exposed to preseason models only as a feature for
``Y+1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ffmodel.features import crossseason
from ffmodel.features.volume import MODEL_POSITIONS, normalize_model_positions


@dataclass(frozen=True)
class SeasonEfficiencySpec:
    """Definition of one opportunity-normalized season response."""

    name: str
    numerator: str
    denominator: str
    positions: tuple[str, ...]
    prior_opportunities: float
    advanced: bool = False


NON_QB_RECEIVERS = ("RB", "WR", "TE")

EFFICIENCY_SPECS = (
    SeasonEfficiencySpec("pass_completion_rate", "pass_cmp", "pass_att", ("QB",), 100),
    SeasonEfficiencySpec("pass_yards_per_attempt", "pass_yds", "pass_att", ("QB",), 100),
    SeasonEfficiencySpec("pass_td_rate", "pass_td", "pass_att", ("QB",), 200),
    SeasonEfficiencySpec("pass_int_rate", "pass_int", "pass_att", ("QB",), 200),
    SeasonEfficiencySpec("rec_catch_rate", "receptions", "targets", NON_QB_RECEIVERS, 40),
    SeasonEfficiencySpec("rec_yards_per_target", "rec_yds", "targets", NON_QB_RECEIVERS, 40),
    SeasonEfficiencySpec("rec_td_rate", "rec_td", "targets", NON_QB_RECEIVERS, 120),
    SeasonEfficiencySpec("rush_yards_per_carry", "rush_yds", "rush_att", MODEL_POSITIONS, 60),
    SeasonEfficiencySpec("rush_td_rate", "rush_td", "rush_att", MODEL_POSITIONS, 120),
    # A play-involvement denominator keeps the final fantasy-points layer
    # complete without pretending that lost fumbles occur only on carries.
    # Passing attempts proxy QB dropbacks, while targets and carries cover the
    # two skill-player opportunity streams supplied by the volume model.
    SeasonEfficiencySpec(
        "fumble_lost_rate",
        "fumbles_lost",
        "fumble_opportunities",
        MODEL_POSITIONS,
        250,
    ),
    # The modeled response. Fumbling is the player's; losing the ball is not.
    # See the note in ``ffmodel.data.ingest`` for the measurements. The lost
    # rate above is retained because it is still what scoring ultimately needs
    # and what a season is graded on -- it is simply no longer what is fitted.
    SeasonEfficiencySpec(
        "fumble_rate",
        "fumbles",
        "fumble_opportunities",
        MODEL_POSITIONS,
        250,
    ),
    SeasonEfficiencySpec(
        "pass_air_yards_per_attempt", "pass_air_yds", "pass_att", ("QB",), 100, True
    ),
    SeasonEfficiencySpec(
        "pass_yac_per_completion", "pass_yac", "pass_cmp", ("QB",), 80, True
    ),
    SeasonEfficiencySpec(
        "pass_epa_per_attempt", "pass_epa", "pass_att", ("QB",), 100, True
    ),
    SeasonEfficiencySpec(
        "pass_first_down_rate", "pass_first_downs", "pass_att", ("QB",), 100, True
    ),
    SeasonEfficiencySpec(
        "rec_air_yards_per_target", "rec_air_yds", "targets", NON_QB_RECEIVERS, 40, True
    ),
    SeasonEfficiencySpec(
        "rec_yac_per_reception", "rec_yac", "receptions", NON_QB_RECEIVERS, 30, True
    ),
    SeasonEfficiencySpec(
        "rec_epa_per_target", "rec_epa", "targets", NON_QB_RECEIVERS, 40, True
    ),
    SeasonEfficiencySpec(
        "rec_first_down_rate", "rec_first_downs", "targets", NON_QB_RECEIVERS, 50, True
    ),
    SeasonEfficiencySpec(
        "rush_epa_per_carry", "rush_epa", "rush_att", MODEL_POSITIONS, 60, True
    ),
    SeasonEfficiencySpec(
        "rush_first_down_rate", "rush_first_downs", "rush_att", MODEL_POSITIONS, 60, True
    ),
    # Usage context rather than efficiency, but it rides the same machinery:
    # aggregated as a ratio of season totals, pooled toward the position mean so
    # a twenty-carry sample is not taken at face value, and exposed only as a
    # lagged feature. See ``ffmodel.features.carry_context`` for what it is and
    # what it was measured to be worth.
    SeasonEfficiencySpec(
        "rush_short_yardage_share",
        "rush_short_yardage_att",
        "rush_att",
        MODEL_POSITIONS,
        60,
        True,
    ),
)

EFFICIENCY_BY_NAME = {spec.name: spec for spec in EFFICIENCY_SPECS}
EFFICIENCY_LABEL_COLUMNS = tuple(spec.name for spec in EFFICIENCY_SPECS)
SHRUNK_EFFICIENCY_COLUMNS = tuple(f"shrunk_{name}" for name in EFFICIENCY_LABEL_COLUMNS)
PRIOR_EFFICIENCY_FEATURES = tuple(f"prior_{name}" for name in EFFICIENCY_LABEL_COLUMNS)
ADVANCED_EFFICIENCY_FEATURES = tuple(
    f"prior_{spec.name}" for spec in EFFICIENCY_SPECS if spec.advanced
)

VOLUME_QUALITY_METRICS = {
    "pass": (
        "prior_pass_yards_per_attempt",
        "prior_pass_epa_per_attempt",
        "prior_pass_first_down_rate",
        "prior_pass_completion_rate",
    ),
    "rec": (
        "prior_rec_yards_per_target",
        "prior_rec_epa_per_target",
        "prior_rec_first_down_rate",
    ),
    "rush": (
        "prior_rush_yards_per_carry",
        "prior_rush_epa_per_carry",
        "prior_rush_first_down_rate",
        "prior_rush_td_rate",
    ),
}

VOLUME_QUALITY_STRENGTH = {"pass": 100.0, "rec": 40.0, "rush": 60.0}

VOLUME_ROLE_SPECS = {
    "pass": {
        "role": "prior_pass_role",
        "draft": "draft_pass_prior",
        "positions": ("QB",),
        "fallback": {"QB": 0.800, "RB": 0.0005, "WR": 0.0005, "TE": 0.0002},
    },
    "target": {
        "role": "prior_target_role",
        "draft": "draft_target_prior",
        "positions": NON_QB_RECEIVERS,
        "fallback": {"QB": 0.0001, "RB": 0.120, "WR": 0.180, "TE": 0.150},
    },
    "carry": {
        "role": "prior_carry_role",
        "draft": "draft_carry_prior",
        "positions": MODEL_POSITIONS,
        "fallback": {"QB": 0.080, "RB": 0.450, "WR": 0.010, "TE": 0.003},
    },
}

CONDITIONAL_EFFICIENCY_SIGNALS = {
    "pass": (
        "prior_pass_quality_signal",
        "prior_pass_td_rate_centered",
    ),
    "target": (
        "prior_rec_quality_signal",
        "prior_rec_epa_per_target_centered",
    ),
    "carry": ("prior_rush_epa_per_carry_centered",),
}

_CONDITIONAL_CENTER_SOURCES = {
    "prior_pass_td_rate": "prior_pass_td_rate_centered",
    "prior_rec_epa_per_target": "prior_rec_epa_per_target_centered",
    "prior_rush_epa_per_carry": "prior_rush_epa_per_carry_centered",
}

CONDITIONAL_VOLUME_EFFICIENCY_FEATURES = (
    "prior_role_continuity",
    "prior_role_team_change",
    *(
        feature
        for stream, signals in CONDITIONAL_EFFICIENCY_SIGNALS.items()
        for feature in (
            f"prior_{stream}_team_competition",
            f"prior_{stream}_room_competition",
            f"prior_{stream}_role_uncertainty",
            *(signal for signal in signals if signal.endswith("_centered")),
            *(
                f"{signal}_x_{modifier}"
                for signal in signals
                for modifier in (
                    "room",
                    "team",
                    "uncertainty",
                    "returning",
                    "changer",
                    "room_returning",
                )
            ),
        )
    ),
)

VOLUME_EFFICIENCY_DERIVED_FEATURES = tuple(
    feature
    for stream, metrics in VOLUME_QUALITY_METRICS.items()
    for feature in (
        f"prior_{stream}_efficiency_reliability",
        *(f"{metric}_position_rank" for metric in metrics),
        f"prior_{stream}_quality_rank",
        f"prior_{stream}_quality_signal",
        f"prior_{stream}_team_quality_rank",
        f"prior_{stream}_team_quality_signal",
    )
)

_AGGREGATE_COLUMNS = tuple(
    dict.fromkeys(
        column
        for spec in EFFICIENCY_SPECS
        for column in (spec.numerator, spec.denominator)
    )
)

# Raw numerators are retained under an ``eff_`` prefix on preseason modeling
# rows. The prefix keeps current-season scoring outcomes visibly separate from
# preseason features and gives count likelihoods exact integer successes.
EFFICIENCY_NUMERATOR_COLUMNS = tuple(
    dict.fromkeys(spec.numerator for spec in EFFICIENCY_SPECS)
)


def player_season_efficiency(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate and partially pool efficiency for every player-season.

    Optional advanced numerators remain missing when a provider did not measure
    them.  Basic counting statistics retain their canonical zero semantics.
    """
    required = {"season", "player_name", "position"}
    missing = required - set(player_weeks.columns)
    if missing:
        raise ValueError(f"player weeks are missing columns: {sorted(missing)}")

    weeks = normalize_model_positions(player_weeks)
    weeks = weeks[weeks["position"].isin(MODEL_POSITIONS)].copy()
    if "player_id" not in weeks:
        weeks["player_id"] = pd.NA
    weeks["player_key"] = crossseason.player_key(weeks)
    for column in ("pass_att", "targets", "rush_att"):
        if column not in weeks:
            weeks[column] = 0.0
        weeks[column] = pd.to_numeric(weeks[column], errors="coerce")
    weeks["fumble_opportunities"] = (
        weeks["pass_att"].fillna(0)
        + weeks["targets"].fillna(0)
        + weeks["rush_att"].fillna(0)
    )
    for column in _AGGREGATE_COLUMNS:
        if column not in weeks:
            weeks[column] = np.nan
        weeks[column] = pd.to_numeric(weeks[column], errors="coerce")

    weeks["_opportunities"] = (
        weeks["pass_att"].fillna(0)
        + weeks["targets"].fillna(0)
        + weeks["rush_att"].fillna(0)
    )
    identity = (
        weeks.groupby(
            ["season", "player_key", "player_name", "position"], dropna=False
        )["_opportunities"]
        .sum()
        .reset_index()
        .sort_values(
            ["season", "player_key", "_opportunities", "position", "player_name"]
        )
        .drop_duplicates(["season", "player_key"], keep="last")
        .drop(columns="_opportunities")
    )
    totals = (
        weeks.groupby(["season", "player_key"], dropna=False)[list(_AGGREGATE_COLUMNS)]
        .sum(min_count=1)
        .reset_index()
    )
    out = identity.merge(totals, on=["season", "player_key"], how="inner")

    advanced_observed = np.zeros(len(out), dtype=bool)
    for spec in EFFICIENCY_SPECS:
        numerator = pd.to_numeric(out[spec.numerator], errors="coerce")
        denominator = pd.to_numeric(out[spec.denominator], errors="coerce")
        valid = (
            out["position"].isin(spec.positions)
            & numerator.notna()
            & denominator.gt(0)
        )
        raw = np.divide(
            numerator,
            denominator,
            out=np.full(len(out), np.nan, dtype=float),
            where=valid,
        )
        out[spec.name] = raw

        valid_num = numerator.where(valid)
        valid_den = denominator.where(valid)
        groupers = [out["season"], out["position"]]
        pooled_num = valid_num.groupby(groupers, dropna=False).transform(
            "sum", min_count=1
        )
        pooled_den = valid_den.groupby(groupers, dropna=False).transform(
            "sum", min_count=1
        )
        pooled_mean = pooled_num / pooled_den
        season_num = valid_num.groupby(out["season"], dropna=False).transform(
            "sum", min_count=1
        )
        season_den = valid_den.groupby(out["season"], dropna=False).transform(
            "sum", min_count=1
        )
        pooled_mean = pooled_mean.fillna(season_num / season_den)
        shrunk = (
            numerator + spec.prior_opportunities * pooled_mean
        ) / (denominator + spec.prior_opportunities)
        out[f"shrunk_{spec.name}"] = shrunk.where(valid)
        if spec.advanced:
            advanced_observed |= valid.to_numpy()

    out["advanced_efficiency_available"] = advanced_observed.astype(int)
    return out.sort_values(["season", "player_key"]).reset_index(drop=True)


def efficiency_label_columns() -> list[str]:
    """Columns to merge as current-season labels/exposures."""
    return list(
        dict.fromkeys(
            [
                "season",
                "player_key",
                "position",
                "pass_att",
                "targets",
                "rush_att",
                "fumble_opportunities",
                *EFFICIENCY_NUMERATOR_COLUMNS,
                *EFFICIENCY_LABEL_COLUMNS,
                *SHRUNK_EFFICIENCY_COLUMNS,
                "advanced_efficiency_available",
            ]
        )
    )


# Decay on a player's own history, per season, for the career priors below.
#
# The layer predicts every response from a single prior season. That season is a
# noisy measurement of the thing being predicted -- split-half reliability puts
# yards per target at 42.7% signal and receiving touchdown rate at 26.7% -- while
# the underlying talent barely moves, correlating about 0.93 from one year to the
# next once the noise is divided out. Nearly all of the prior's error is
# therefore measurement error in the prior itself, which no covariate placed
# beside it can repair.
#
# Two constructions were compared against the one-season prior, on the population
# where the response is observed at 30+ opportunities:
#
#   response              1 season   rate EWMA   exposure-weighted, decayed
#   rush_yards_per_carry      8.8%       12.7%       15.5%
#   rec_catch_rate           20.9%       24.1%       26.8%
#   rec_yards_per_target      9.3%       12.6%       14.7%
#   rush_td_rate              4.8%        5.1%        5.7%
#   rec_td_rate               2.6%        3.4%        3.7%
#
# The middle column is what ``season_pathways`` already builds: an EWMA of each
# season's *rate*, in which a 20-target season and a 150-target season carry the
# same weight. Accumulating numerators and denominators instead lets a season
# count for the information it holds, and beats it on every response.
#
# 0.7 per season -- a half-life near two seasons, effective memory near three --
# is one constant for every response rather than the per-response optimum. The
# optima differ (0.5 on rushing touchdown rate, 1.0 on receiving) but they were
# read off seasons that include the holdouts, and the loss from using 0.7
# everywhere is at most 0.2 points of r-squared. A tuned constant that cannot be
# tuned honestly is worth less than that.
CAREER_DECAY = 0.7

CAREER_EFFICIENCY_FEATURES = tuple(
    f"prior_{spec.name}_career" for spec in EFFICIENCY_SPECS
)


def _decayed_history(
    values: pd.Series, seasons: pd.Series, decay: float
) -> np.ndarray:
    """Decayed sum of a player's strictly earlier seasons.

    Season gaps decay by the years elapsed, not by one step, so a player who
    misses a year does not carry a two-year-old season as though it were last
    season's.
    """
    out = np.full(len(values), np.nan)
    total = 0.0
    previous: float | None = None
    for i, (value, season) in enumerate(zip(values.to_numpy(), seasons.to_numpy())):
        if previous is not None:
            total *= decay ** max(1, int(season) - int(previous))
        out[i] = total if previous is not None else np.nan
        if np.isfinite(value):
            total += float(value)
            previous = float(season)
        elif previous is not None:
            previous = float(season)
    return out


def add_career_efficiency_priors(
    efficiency: pd.DataFrame, *, decay: float = CAREER_DECAY
) -> pd.DataFrame:
    """Attach an exposure-weighted, decayed prior over a player's own history.

    Emitted on the season whose history it summarises, so the existing lag in
    ``lagged_efficiency_rows`` carries it forward the same way the one-season
    prior is carried. The value on season Y uses seasons strictly before Y.
    """
    out = efficiency.sort_values(["player_key", "season"]).reset_index(drop=True)
    grouped = out.groupby("player_key", dropna=False)
    for spec in EFFICIENCY_SPECS:
        # Membership is checked on the columns, not on the converted result:
        # ``out.get`` returns None for an absent column and ``pd.to_numeric``
        # turns that into a scalar nan, so a None check downstream never fires.
        if spec.numerator not in out or spec.denominator not in out:
            continue
        numerator = pd.to_numeric(out[spec.numerator], errors="coerce")
        denominator = pd.to_numeric(out[spec.denominator], errors="coerce")
        observed = denominator.gt(0)
        history_num = np.concatenate([
            _decayed_history(numerator.where(observed).iloc[idx],
                             out["season"].iloc[idx], decay)
            for idx in grouped.indices.values()
        ])
        history_den = np.concatenate([
            _decayed_history(denominator.where(observed).iloc[idx],
                             out["season"].iloc[idx], decay)
            for idx in grouped.indices.values()
        ])
        order = np.concatenate(list(grouped.indices.values()))
        num = pd.Series(np.nan, index=out.index)
        den = pd.Series(np.nan, index=out.index)
        num.iloc[order] = history_num
        den.iloc[order] = history_den

        # Shrink toward the same season-and-position mean the one-year prior
        # uses, so the two are on one scale and a model can weigh them.
        valid = den.gt(0)
        pooled = pd.to_numeric(out.get(f"shrunk_{spec.name}"), errors="coerce")
        groupers = [out["season"], out["position"]]
        pooled_mean = pooled.groupby(groupers, dropna=False).transform("mean")
        pooled_mean = pooled_mean.fillna(pooled.groupby(out["season"]).transform("mean"))
        out[f"career_{spec.name}"] = (
            (num + spec.prior_opportunities * pooled_mean)
            / (den + spec.prior_opportunities)
        ).where(valid)
        out[f"career_{spec.name}_exposure"] = den.where(valid)
    return out


def lagged_efficiency_rows(efficiency: pd.DataFrame) -> pd.DataFrame:
    """Shift pooled efficiency from season ``Y`` onto preseason ``Y+1`` rows."""
    columns = ["season", "player_key", "advanced_efficiency_available"] + list(
        SHRUNK_EFFICIENCY_COLUMNS
    )
    career = [
        f"career_{name}" for name in EFFICIENCY_LABEL_COLUMNS
        if f"career_{name}" in efficiency
    ]
    out = efficiency[columns + career].copy()
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int) + 1
    out = out.rename(
        columns={
            **{
                f"shrunk_{name}": f"prior_{name}"
                for name in EFFICIENCY_LABEL_COLUMNS
            },
            **{
                f"career_{name}": f"prior_{name}_career"
                for name in EFFICIENCY_LABEL_COLUMNS
            },
            "advanced_efficiency_available": "prior_advanced_efficiency_available",
        }
    )
    return out


def add_volume_efficiency_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Add reliability and nonlinear relative-quality features for volume.

    Percentile ranks are calculated only from lagged observations present on
    the known preseason roster. They are therefore available at prediction
    time and cannot contain current-season outcomes.
    """
    out = rows.copy()
    exposure_columns = {
        "pass": "prior_pass_att",
        "rec": "prior_targets",
        "rush": "prior_rush_att",
    }
    for stream, metrics in VOLUME_QUALITY_METRICS.items():
        exposure = pd.to_numeric(out[exposure_columns[stream]], errors="coerce")
        strength = VOLUME_QUALITY_STRENGTH[stream]
        reliability = exposure / (exposure + strength)
        reliability = reliability.where(exposure.ge(0))
        out[f"prior_{stream}_efficiency_reliability"] = reliability

        rank_columns = []
        for metric in metrics:
            values = pd.to_numeric(
                out.get(metric, pd.Series(np.nan, index=out.index)), errors="coerce"
            )
            rank_column = f"{metric}_position_rank"
            out[rank_column] = values.groupby(
                [out["season"], out["position"]], dropna=False
            ).rank(method="average", pct=True)
            rank_columns.append(rank_column)
        composite = out[rank_columns].mean(axis=1, skipna=True)
        composite = composite.where(out[rank_columns].notna().any(axis=1))
        out[f"prior_{stream}_quality_rank"] = composite
        out[f"prior_{stream}_quality_signal"] = (composite - 0.5) * reliability
        team_rank = composite.groupby(
            [out["season"], out["team"]], dropna=False
        ).rank(method="average", pct=True)
        out[f"prior_{stream}_team_quality_rank"] = team_rank
        out[f"prior_{stream}_team_quality_signal"] = (
            team_rank - 0.5
        ) * reliability
    return out


def add_teammate_quality_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Cross-positional teammate quality: who is throwing the ball.

    Every efficiency spec in this package has an empty feature list, so a
    receiver's yards per target is modelled from his own history, a position
    effect and his projected target volume. Nothing tells the model who throws
    to him, and a receiver moving between a replacement-level and an elite
    quarterback has no mechanism to move.

    ``prior_rec_team_quality_signal`` does not fill that gap despite the name.
    It ranks a player's *own* quality within his team, so it measures relative
    standing among teammates rather than the quality of those teammates.

    This attaches the projected starter's own prior-season passing quality to
    every skill-position row on his team. Everything it reads is available
    before the season starts: last season's quality, and a preseason depth
    chart.

    The confound is worth naming rather than burying. A receiver who kept his
    quarterback contributed to that quarterback's prior composite, so part of
    any measured effect is a player predicting himself. ``team_change`` splits
    that: for a player who moved, the new quarterback's prior quality is
    genuinely exogenous, and an effect that exists only for players who stayed
    is circularity rather than signal.
    """
    required = {"season", "team", "position"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"teammate-quality rows are missing columns: {sorted(missing)}"
        )
    out = rows.copy()
    if "prior_pass_quality_signal" not in out:
        raise ValueError(
            "teammate quality needs prior_pass_quality_signal; run "
            "add_volume_efficiency_features first"
        )

    quarterback = out["position"].astype(str).str.upper().eq("QB")
    listed = pd.to_numeric(
        out.get("qb_listed_starter", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).fillna(0)
    depth = pd.to_numeric(
        out.get("qb_depth_rank", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    # The listed starter, falling back to the shallowest depth chart entry when
    # a team lists nobody. Both are preseason artifacts; ``primary_qb`` is not,
    # because it is derived from what actually happened.
    rank = np.where(listed.to_numpy() == 1, 0.0, depth.fillna(99).to_numpy() + 1.0)
    order = pd.Series(np.where(quarterback.to_numpy(), rank, np.inf), index=out.index)
    signal = pd.to_numeric(out["prior_pass_quality_signal"], errors="coerce")

    chosen = (
        pd.DataFrame({"order": order, "signal": signal, "qb": quarterback})
        .assign(season=out["season"], team=out["team"])
        .sort_values("order")
        .groupby(["season", "team"], dropna=False)
        .first()
    )
    starter_signal = chosen["signal"].where(chosen["qb"])
    keys = pd.MultiIndex.from_frame(out[["season", "team"]])
    out["teammate_qb_quality_signal"] = starter_signal.reindex(keys).to_numpy()
    # A quarterback's own row keeps this empty: his passing quality is already
    # his own feature, and feeding it back as "teammate quality" would let the
    # model read one signal twice under two names.
    out.loc[quarterback, "teammate_qb_quality_signal"] = np.nan
    return out


def add_conditional_volume_efficiency_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Interact lagged efficiency with preseason role context.

    Competition is calculated from lagged role, rookie draft priors, and
    position fallbacks on the known current roster. ``room`` refers to the
    player's current team-position group, while ``team`` uses every position
    eligible for that opportunity stream. All inputs are available before the
    response season begins.
    """
    required = {"season", "team", "position", "team_change"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"conditional efficiency rows are missing columns: {sorted(missing)}"
        )
    out = rows.copy()
    changed = pd.to_numeric(out["team_change"], errors="coerce").fillna(0.0)
    changed = changed.clip(0.0, 1.0)
    cold_start = pd.to_numeric(
        out.get("cold_start", pd.Series(0, index=out.index)), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    out["prior_role_continuity"] = (1.0 - changed) * (1.0 - cold_start)
    out["prior_role_team_change"] = changed

    for source, centered in _CONDITIONAL_CENTER_SOURCES.items():
        values = pd.to_numeric(
            out.get(source, pd.Series(np.nan, index=out.index)), errors="coerce"
        )
        position_median = values.groupby(
            [out["season"], out["position"]], dropna=False
        ).transform("median")
        out[centered] = values - position_median

    for stream, spec in VOLUME_ROLE_SPECS.items():
        role = pd.to_numeric(
            out.get(spec["role"], pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        draft = pd.to_numeric(
            out.get(spec["draft"], pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        fallback = out["position"].map(spec["fallback"]).astype(float)
        prior = role.where(role > 0).combine_first(draft.where(draft > 0))
        prior = prior.combine_first(fallback)
        prior = prior.where(out["position"].isin(spec["positions"]), 0.0)
        prior = prior.fillna(0.0).clip(lower=0.0)

        team_group = [out["season"], out["team"]]
        team_total = prior.groupby(team_group, dropna=False).transform("sum")
        team_share = prior.div(team_total.where(team_total > 0)).fillna(0.0)
        team_leader = team_share.groupby(team_group, dropna=False).transform("max")
        out[f"prior_{stream}_team_competition"] = (1.0 - team_leader).clip(
            0.0, 1.0
        )

        room_group = [out["season"], out["team"], out["position"]]
        room_total = prior.groupby(room_group, dropna=False).transform("sum")
        room_share = prior.div(room_total.where(room_total > 0)).fillna(0.0)
        room_leader = room_share.groupby(room_group, dropna=False).transform("max")
        out[f"prior_{stream}_room_competition"] = (1.0 - room_leader).clip(
            0.0, 1.0
        )
        out[f"prior_{stream}_role_uncertainty"] = (1.0 - room_share).clip(
            0.0, 1.0
        )

        for signal in CONDITIONAL_EFFICIENCY_SIGNALS[stream]:
            efficiency = pd.to_numeric(
                out.get(signal, pd.Series(np.nan, index=out.index)), errors="coerce"
            )
            modifiers = {
                "room": out[f"prior_{stream}_room_competition"],
                "team": out[f"prior_{stream}_team_competition"],
                "uncertainty": out[f"prior_{stream}_role_uncertainty"],
                "returning": out["prior_role_continuity"],
                "changer": out["prior_role_team_change"],
            }
            for modifier, context in modifiers.items():
                out[f"{signal}_x_{modifier}"] = efficiency * context
            out[f"{signal}_x_room_returning"] = (
                out[f"{signal}_x_room"] * out["prior_role_continuity"]
            )
    return out
