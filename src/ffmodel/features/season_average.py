"""Preseason features for season-wide per-game volume projections.

The prediction unit is a player-season role, not an individual matchup.  A
row for season ``Y`` contains roster membership and other information knowable
before that season, predictors derived from ``Y-1``, and realized ``Y`` counts
used only as labels during fitting.  This separation lets walk-forward tests
drop an entire season without changing the feature contract.

The roster in historical backtests is inferred from the player's primary team
that season.  A live projection should supply the actual preseason roster; the
model interface deliberately uses the same columns for either source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ffmodel.config import NFLVERSE_INJURY_FIRST_SEASON, NFLVERSE_INJURY_LAST_SEASON
from ffmodel.data import load_player_weeks
from ffmodel.data import ingest, legacy
from ffmodel.data.wikipedia_coaching import team_identity
from ffmodel.features import crossseason
from ffmodel.features.athleticism import ATHLETIC_FEATURES
from ffmodel.features.combine import COMBINE_FEATURES
from ffmodel.features.draft import (
    expected_rookie_claim,
    expected_rookie_pass_claim,
    load_draft_capital,
)
from ffmodel.features.season_efficiency import (
    CONDITIONAL_VOLUME_EFFICIENCY_FEATURES,
    EFFICIENCY_LABEL_COLUMNS,
    EFFICIENCY_NUMERATOR_COLUMNS,
    PRIOR_EFFICIENCY_FEATURES,
    SHRUNK_EFFICIENCY_COLUMNS,
    VOLUME_EFFICIENCY_DERIVED_FEATURES,
    add_conditional_volume_efficiency_features,
    add_volume_efficiency_features,
    lagged_efficiency_rows,
    player_season_efficiency,
)
from ffmodel.features.season_injury import (
    INJURY_AVAILABILITY_FEATURES,
    add_season_injury_features,
)
from ffmodel.features.season_pathways import (
    PLAYER_PATHWAY_FEATURES,
    add_player_pathway_features,
)
from ffmodel.features.volume import (
    MODEL_POSITIONS,
    normalize_model_positions,
    opportunity_position,
)

TEAM_KEYS = ["season", "team"]
PLAYER_KEYS = ["season", "team", "player_key"]
ROSTER_STATUSES = frozenset({"ACT", "RES", "INA", "EXE"})

# Realized current-season quantities. On a projection season these do not exist,
# and the label merge zero-fills them, so they are restored to missing. ``games``
# is deliberately absent: it is read as an integer exposure at predict time and
# is instead set to the scheduled team-game count.
PROJECTION_BLANK_LABELS = (
    "pass_att",
    "targets",
    "rush_att",
    "pass_attempt_share",
    "target_share",
    "carry_share",
    "late_pass_attempt_share",
    "late_target_share",
    "late_carry_share",
    "observed_availability",
    "observed_qb_workload_share",
    "offense_snaps",
    "team_offense_snaps",
    "snap_share",
    "qb_snap_share",
    "stat_activity_games",
    "fumble_opportunities",
)


@dataclass(frozen=True)
class SeasonAverageData:
    """Model-ready team and player season rows."""

    team_rows: pd.DataFrame
    player_rows: pd.DataFrame


def preseason_roster_snapshot(
    rosters: pd.DataFrame,
    depth_charts: pd.DataFrame | None = None,
    *,
    cutoff_week: int = 1,
) -> pd.DataFrame:
    """Build a leakage-safe offensive roster from data available by a week.

    The default uses regular-season week 1, which is the earliest consistently
    available historical nflverse roster/depth snapshot. Cut, retired, and
    practice-squad rows are excluded; active, inactive, reserve/PUP, and exempt
    players remain because they are valid season-long availability outcomes.
    ``observed_roster_games`` is derived from later weekly rows solely as a
    training label and is not part of ``PRESEASON_FEATURES``.
    """
    required = {"season", "team", "position", "week"}
    missing = required - set(rosters.columns)
    if missing:
        raise ValueError(f"roster rows are missing columns: {sorted(missing)}")
    all_rosters = rosters.copy()
    all_rosters = all_rosters.rename(
        columns={
            "gsis_id": "player_id",
            "full_name": "player_name",
            "status": "roster_status",
            "years_exp": "experience",
        }
    )
    if "game_type" in all_rosters:
        all_rosters = all_rosters[
            all_rosters["game_type"].isna() | all_rosters["game_type"].eq("REG")
        ]
    all_rosters["week"] = pd.to_numeric(all_rosters["week"], errors="coerce")
    status = all_rosters.get(
        "roster_status", pd.Series("ACT", index=all_rosters.index)
    )
    all_rosters["roster_status"] = status.astype(str).str.upper()
    all_rosters["position"] = opportunity_position(all_rosters["position"])
    all_rosters = all_rosters[
        all_rosters["position"].isin(MODEL_POSITIONS)
    ].copy()
    all_rosters = _normalize_teams(all_rosters)
    if "player_id" not in all_rosters:
        all_rosters["player_id"] = pd.NA
    if "player_name" not in all_rosters:
        all_rosters["player_name"] = pd.NA
    all_rosters["player_key"] = crossseason.player_key(all_rosters)
    active_games = (
        all_rosters[all_rosters["roster_status"].eq("ACT")]
        .groupby(PLAYER_KEYS, dropna=False)["week"]
        .nunique()
        .rename("observed_roster_games")
        .reset_index()
    )

    out = all_rosters[all_rosters["week"].le(cutoff_week)].copy()
    out = out[out["roster_status"].isin(ROSTER_STATUSES)].copy()
    out = (
        out.sort_values(["season", "team", "player_key", "week"])
        .drop_duplicates(["season", "team", "player_key"], keep="last")
        .reset_index(drop=True)
    )
    out = out.merge(active_games, on=PLAYER_KEYS, how="left")
    out["observed_roster_games"] = out["observed_roster_games"].fillna(0.0)

    depth = _preseason_depth_snapshot(depth_charts, cutoff_week=cutoff_week)
    if not depth.empty:
        out = out.merge(
            depth,
            on=["season", "team", "player_key"],
            how="left",
            suffixes=("", "_depth"),
        )
    else:
        out["depth_rank"] = np.nan
        out["depth_snapshot_week"] = np.nan
    out["depth_rank"] = pd.to_numeric(out["depth_rank"], errors="coerce")
    out["qb_depth_rank"] = out["depth_rank"].where(out["position"].eq("QB"))
    out["qb_listed_starter"] = (
        out["position"].eq("QB") & out["depth_rank"].eq(1)
    ).astype(int)
    out["roster_active"] = out["roster_status"].eq("ACT").astype(int)
    out["roster_reserve"] = out["roster_status"].isin({"RES", "INA", "EXE"}).astype(int)
    birth = pd.to_datetime(
        out.get("birth_date", pd.Series(pd.NaT, index=out.index)), errors="coerce"
    )
    reference = pd.to_datetime(out["season"].astype(int).astype(str) + "-09-01")
    roster_age = (reference - birth).dt.days / 365.25
    existing_age = pd.to_numeric(
        out.get("age", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    out["age"] = existing_age.combine_first(roster_age)
    out["experience"] = pd.to_numeric(
        out.get("experience", pd.Series(np.nan, index=out.index)), errors="coerce"
    )
    out["roster_snapshot_week"] = out["week"].astype(int)
    out["roster_snapshot_source"] = "nflverse_week1"
    keep = [
        "season",
        "team",
        "player_key",
        "player_id",
        "player_name",
        "position",
        "age",
        "experience",
        "roster_status",
        "roster_active",
        "roster_reserve",
        "depth_rank",
        "qb_depth_rank",
        "qb_listed_starter",
        "roster_snapshot_week",
        "depth_snapshot_week",
        "roster_snapshot_source",
        "observed_roster_games",
    ]
    return out[keep].sort_values(PLAYER_KEYS).reset_index(drop=True)


def load_preseason_roster_snapshot(
    seasons: Iterable[int], *, cutoff_week: int = 1, cache_dir=None
) -> pd.DataFrame:
    """Load nflverse weekly rosters and depth charts at a preseason cutoff."""
    seasons = sorted(set(map(int, seasons)))
    rosters = ingest.load_weekly_rosters(seasons, cache_dir=cache_dir)
    depth = ingest.load_depth_charts(seasons, cache_dir=cache_dir)
    return preseason_roster_snapshot(rosters, depth, cutoff_week=cutoff_week)


def _preseason_depth_snapshot(
    depth_charts: pd.DataFrame | None, *, cutoff_week: int
) -> pd.DataFrame:
    if depth_charts is None or depth_charts.empty:
        return pd.DataFrame(
            columns=TEAM_KEYS
            + ["player_key", "depth_rank", "depth_snapshot_week"]
        )
    depth = depth_charts.copy().rename(
        columns={
            "club_code": "team",
            "gsis_id": "player_id",
            "full_name": "player_name",
            "depth_team": "depth_rank",
        }
    )
    if "game_type" in depth:
        depth = depth[depth["game_type"].isna() | depth["game_type"].eq("REG")]
    if "formation" in depth:
        depth = depth[depth["formation"].isna() | depth["formation"].eq("Offense")]
    depth["week"] = pd.to_numeric(depth["week"], errors="coerce")
    depth = depth[depth["week"].le(cutoff_week)].copy()
    depth["position"] = opportunity_position(depth["position"])
    depth = depth[depth["position"].isin(MODEL_POSITIONS)].copy()
    depth = _normalize_teams(depth)
    depth["player_key"] = crossseason.player_key(depth)
    depth["depth_rank"] = pd.to_numeric(depth["depth_rank"], errors="coerce")
    depth["depth_snapshot_week"] = depth["week"]
    order = ["season", "team", "player_key", "week"]
    if "dt" in depth.columns:
        # The 2025+ feed publishes many snapshots per week, so every row ties on
        # week and the retained one would be arbitrary — a chart written the day
        # after the draft can outrank one from the eve of the season. Order by
        # recency so the cutoff keeps the freshest chart it is allowed to see.
        depth["dt"] = pd.to_datetime(depth["dt"], errors="coerce", utc=True)
        order.append("dt")
    return (
        depth.sort_values(order)
        .drop_duplicates(["season", "team", "player_key"], keep="last")
        [TEAM_KEYS + ["player_key", "depth_rank", "depth_snapshot_week"]]
        .reset_index(drop=True)
    )


def team_season_volume(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate coherent offensive counts and per-game rates by team-season."""
    pw = normalize_model_positions(player_weeks)
    pw = _normalize_teams(pw)
    weekly = (
        pw.groupby(["season", "week", "team"], dropna=False)
        .agg(
            pass_attempts=("pass_att", "sum"),
            sacks=("pass_sacks", "sum"),
            sacks_available=("pass_sacks_available", "max"),
            rush_attempts=("rush_att", "sum"),
            targets=("targets", "sum"),
        )
        .reset_index()
    )
    weekly["opportunity_plays"] = (
        weekly["pass_attempts"] + weekly["rush_attempts"]
    )
    weekly["dropbacks"] = weekly["pass_attempts"] + weekly["sacks"]
    weekly["plays"] = weekly["opportunity_plays"] + weekly["sacks"]
    # A few legacy games omit a quarterback line. Exclude those games from the
    # target-rate response while preserving their play/pass information.
    weekly["target_valid"] = weekly["targets"] <= weekly["pass_attempts"]
    weekly["valid_target_pass_attempts"] = weekly["pass_attempts"].where(
        weekly["target_valid"], 0
    )
    weekly["valid_targets"] = weekly["targets"].where(
        weekly["target_valid"], 0
    )

    rows = (
        weekly.groupby(TEAM_KEYS, dropna=False)
        .agg(
            games=("week", "nunique"),
            opportunity_plays=("opportunity_plays", "sum"),
            plays=("plays", "sum"),
            pass_attempts=("pass_attempts", "sum"),
            sacks=("sacks", "sum"),
            sacks_observed=("sacks_available", "min"),
            dropbacks=("dropbacks", "sum"),
            rush_attempts=("rush_attempts", "sum"),
            targets=("targets", "sum"),
            valid_target_pass_attempts=("valid_target_pass_attempts", "sum"),
            valid_targets=("valid_targets", "sum"),
            valid_target_games=("target_valid", "sum"),
        )
        .reset_index()
    )
    rows["sacks_observed"] = rows["sacks_observed"].astype(bool)
    # Official plays and dropbacks require sacks. Keep them missing for legacy
    # sources instead of presenting schema-filled zeros as measured truth.
    rows.loc[~rows["sacks_observed"], ["plays", "dropbacks"]] = np.nan
    rows["opportunity_plays_per_game"] = rows["opportunity_plays"] / rows["games"]
    rows["plays_per_game"] = rows["plays"] / rows["games"]
    rows["pass_attempts_per_game"] = rows["pass_attempts"] / rows["games"]
    rows["sacks_per_game"] = rows["sacks"].where(rows["sacks_observed"]) / rows["games"]
    rows["dropbacks_per_game"] = rows["dropbacks"] / rows["games"]
    rows["rush_attempts_per_game"] = rows["rush_attempts"] / rows["games"]
    rows["targets_per_game"] = rows["targets"] / rows["games"]
    rows["pass_rate"] = _divide(rows["pass_attempts"], rows["opportunity_plays"])
    rows["sack_rate"] = _divide(rows["sacks"], rows["dropbacks"])
    rows.loc[~rows["sacks_observed"], "sack_rate"] = np.nan
    rows["target_rate"] = _divide(
        rows["valid_targets"], rows["valid_target_pass_attempts"]
    )
    rows["no_target_attempts"] = (
        rows["valid_target_pass_attempts"] - rows["valid_targets"]
    )
    return rows.sort_values(TEAM_KEYS).reset_index(drop=True)


def team_transition_rows(
    team_volume: pd.DataFrame, projection_seasons: Iterable[int] = ()
) -> pd.DataFrame:
    """Attach strictly prior-season team rates to each realized team-season.

    ``projection_seasons`` name seasons that have not been played. They carry no
    realized team volume, so their rows are built from the prior season's rates
    alone and every realized column is left missing rather than zero-filled — a
    zero would be indistinguishable from a genuinely play-less team.
    """
    prior = team_volume[
        TEAM_KEYS
        + [
            "opportunity_plays_per_game",
            "plays_per_game",
            "pass_rate",
            "sack_rate",
            "target_rate",
            "pass_attempts_per_game",
            "sacks_per_game",
            "dropbacks_per_game",
            "rush_attempts_per_game",
            "targets_per_game",
        ]
    ].copy()
    prior["season"] += 1
    prior = prior.rename(
        columns={column: f"prior_{column}" for column in prior.columns if column not in TEAM_KEYS}
    )
    out = team_volume.merge(prior, on=TEAM_KEYS, how="inner")
    projection = sorted({int(season) for season in projection_seasons})
    if projection:
        future = prior[prior["season"].isin(projection)].copy()
        for column in team_volume.columns:
            if column not in TEAM_KEYS:
                future[column] = np.nan
        # ``games`` is exposure, not an outcome: the team model reads it as a
        # positive integer count of games to project over and rejects a missing
        # one. Carry the scheduled slate forward, mirroring the player rows.
        scheduled = pd.to_numeric(
            team_volume.loc[
                team_volume["season"].eq(team_volume["season"].max()), "games"
            ],
            errors="coerce",
        ).mode()
        if not scheduled.empty:
            future["games"] = float(scheduled.iloc[0])
        out = pd.concat([out, future], ignore_index=True, sort=False)
    out["transition"] = (out["season"] - 1).astype(str) + "->" + out["season"].astype(str)
    return out.sort_values(TEAM_KEYS).reset_index(drop=True)


def _projected_team_games(
    team_games: pd.DataFrame, projection_seasons: set[int]
) -> pd.DataFrame:
    """Carry a scheduled team-game count into seasons that have not been played.

    Exposure is taken from the most recent observed season rather than a literal
    so the count follows whatever schedule length the data actually shows.
    """
    if not projection_seasons or team_games.empty:
        return team_games
    latest = team_games[team_games["season"].eq(team_games["season"].max())]
    scheduled = pd.to_numeric(latest["team_games"], errors="coerce").mode()
    if scheduled.empty:
        return team_games
    teams = team_games["team"].dropna().unique()
    future = pd.DataFrame(
        [
            {"season": season, "team": team, "team_games": float(scheduled.iloc[0])}
            for season in sorted(projection_seasons)
            for team in teams
        ]
    )
    return pd.concat([team_games, future], ignore_index=True, sort=False)


def player_team_season_usage(player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Realized labels scoped to the player's preseason team.

    Unlike the cross-season prior builder, this retains one row per player-team
    so a later trade cannot move full-season labels onto the wrong preseason
    roster support.
    """
    pw = normalize_model_positions(player_weeks)
    pw = _normalize_teams(pw)
    pw["player_key"] = crossseason.player_key(pw)
    pw["is_active"] = (
        pw["pass_att"] + pw["targets"] + pw["rush_att"] + pw["receptions"] > 0
    ).astype(int)
    keys = ["season", "team", "player_key", "player_name", "position"]
    out = (
        pw.groupby(keys, dropna=False)
        .agg(
            pass_att=("pass_att", "sum"),
            targets=("targets", "sum"),
            rush_att=("rush_att", "sum"),
            games=("is_active", "sum"),
        )
        .reset_index()
    )
    for count, share in (
        ("pass_att", "pass_attempt_share"),
        ("targets", "target_share"),
        ("rush_att", "carry_share"),
    ):
        total = out.groupby(TEAM_KEYS)[count].transform("sum")
        out[share] = np.divide(
            out[count], total, out=np.zeros(len(out), dtype=float), where=total > 0
        )

    late = pw[pw["week"] >= crossseason.LATE_SEASON_START_WEEK]
    late_counts = (
        late.groupby(keys, dropna=False)
        .agg(
            late_pass_att=("pass_att", "sum"),
            late_targets=("targets", "sum"),
            late_rush_att=("rush_att", "sum"),
        )
        .reset_index()
    )
    for count, share in (
        ("late_pass_att", "late_pass_attempt_share"),
        ("late_targets", "late_target_share"),
        ("late_rush_att", "late_carry_share"),
    ):
        total = late_counts.groupby(TEAM_KEYS)[count].transform("sum")
        late_counts[share] = np.divide(
            late_counts[count],
            total,
            out=np.zeros(len(late_counts), dtype=float),
            where=total > 0,
        )
    keep = keys + [
        "late_pass_attempt_share",
        "late_target_share",
        "late_carry_share",
    ]
    out = out.merge(late_counts[keep], on=keys, how="left")
    out[["late_pass_attempt_share", "late_target_share", "late_carry_share"]] = out[
        ["late_pass_attempt_share", "late_target_share", "late_carry_share"]
    ].fillna(0.0)
    return out.sort_values(PLAYER_KEYS).reset_index(drop=True)


def player_preseason_rows(
    seasons: Iterable[int],
    *,
    source: str = "auto",
    team_volume: pd.DataFrame | None = None,
    player_weeks: pd.DataFrame | None = None,
    roster_snapshot: pd.DataFrame | None = None,
    injury_reports: pd.DataFrame | None = None,
    weekly_rosters: pd.DataFrame | None = None,
    injury_snapshot: pd.DataFrame | None = None,
    injury_cutoff_week: int = 1,
    projection_seasons: Iterable[int] = (),
) -> pd.DataFrame:
    """Build roster rows with prior-year predictors and current-year labels.

    Current-season target/carry counts, active games, and shares are labels or
    likelihood exposures. They are never included in ``PRESEASON_FEATURES``.

    ``projection_seasons`` name seasons that have not been played. No observed
    play-by-play exists for them, so every raw feed is read for the observed
    seasons only; their rows come from ``roster_snapshot`` and carry prior-season
    predictors plus draft capital. Their label columns are structurally absent
    and must never be read as outcomes.
    """
    seasons = sorted(set(int(season) for season in seasons))
    projection = {int(season) for season in projection_seasons}
    unknown = projection - set(seasons)
    if unknown:
        raise ValueError(
            f"projection_seasons must be included in seasons; missing {sorted(unknown)}"
        )
    observed = [season for season in seasons if season not in projection]
    if not observed:
        raise ValueError("at least one observed season is required for prior features")
    if projection and roster_snapshot is None:
        raise ValueError("projection seasons require an explicit roster_snapshot")
    history = crossseason.season_usage(observed, source=source).copy()
    history["position"] = history["position"].astype(str).str.upper()
    history = history[history["position"].isin(MODEL_POSITIONS)].copy()
    history = _normalize_teams(history)
    history["player_key"] = crossseason.player_key(history)
    if player_weeks is None:
        player_weeks = load_player_weeks(observed, source=source)
    if team_volume is None:
        team_volume = team_season_volume(player_weeks)
    efficiency = player_season_efficiency(player_weeks)
    team_games = team_volume[TEAM_KEYS + ["games"]].rename(columns={"games": "team_games"})
    team_games = _projected_team_games(team_games, projection)
    history = history.merge(team_games, on=TEAM_KEYS, how="left")
    snap_usage = load_season_snap_usage(observed, source=source)
    history = _merge_snap_usage(history, snap_usage)

    if roster_snapshot is None:
        usage = history.copy()
        usage["roster_status"] = "INFERRED"
        usage["roster_active"] = 1
        usage["roster_reserve"] = 0
        usage["depth_rank"] = np.nan
        usage["qb_depth_rank"] = np.nan
        usage["qb_listed_starter"] = 0
        usage["roster_snapshot_week"] = np.nan
        usage["depth_snapshot_week"] = np.nan
        usage["roster_snapshot_source"] = "inferred_postseason"
        usage["availability_label_source"] = "stat_activity_proxy"
    else:
        roster = roster_snapshot.copy()
        roster = roster[roster["season"].isin(seasons)].copy()
        roster["position"] = roster["position"].astype(str).str.upper()
        roster = roster[roster["position"].isin(MODEL_POSITIONS)].copy()
        labels = player_team_season_usage(player_weeks)
        label_columns = PLAYER_KEYS + [
            "pass_att",
            "targets",
            "rush_att",
            "games",
            "pass_attempt_share",
            "target_share",
            "carry_share",
            "late_pass_attempt_share",
            "late_target_share",
            "late_carry_share",
        ]
        usage = roster.merge(labels[label_columns], on=PLAYER_KEYS, how="left")
        outcome_columns = [column for column in label_columns if column not in PLAYER_KEYS]
        usage[outcome_columns] = usage[outcome_columns].fillna(0.0)
        demographics = history[
            ["season", "player_key", "age", "experience"]
        ].drop_duplicates(["season", "player_key"])
        usage = usage.merge(
            demographics,
            on=["season", "player_key"],
            how="left",
            suffixes=("", "_history"),
        )
        for column in ("age", "experience"):
            history_column = f"{column}_history"
            usage[column] = pd.to_numeric(usage[column], errors="coerce").combine_first(
                pd.to_numeric(usage[history_column], errors="coerce")
            )
            usage = usage.drop(columns=history_column)
        usage = usage.merge(team_games, on=TEAM_KEYS, how="left")
        if "observed_roster_games" in usage:
            usage["stat_activity_games"] = usage["games"]
            usage["games"] = np.minimum(
                pd.to_numeric(usage["observed_roster_games"], errors="coerce").fillna(0),
                pd.to_numeric(usage["team_games"], errors="coerce").fillna(0),
            )
            usage["availability_label_source"] = "weekly_roster_active"
            roster_games = roster[
                PLAYER_KEYS + ["observed_roster_games"]
            ].drop_duplicates(PLAYER_KEYS)
            history = history.merge(
                roster_games,
                on=PLAYER_KEYS,
                how="left",
                suffixes=("", "_roster"),
            )
            observed = pd.to_numeric(
                history["observed_roster_games"], errors="coerce"
            )
            history["games"] = np.where(
                observed.notna(),
                np.minimum(observed.fillna(0), history["team_games"]),
                history["games"],
            )
        else:
            usage["availability_label_source"] = "stat_activity_proxy"

        usage = _merge_snap_usage(usage, snap_usage)

        replacement = _replacement_player_rows(
            roster=roster,
            labels=labels,
            player_weeks=player_weeks,
            snap_usage=snap_usage,
            team_games=team_games,
        )
        usage = pd.concat([usage, replacement], ignore_index=True, sort=False)
        # The team-specific synthetic identity is stable across seasons, so
        # last year's unexpected-QB demand becomes a valid preseason prior.
        history = pd.concat([history, replacement], ignore_index=True, sort=False)

    usage["is_replacement_qb"] = pd.to_numeric(
        usage.get("is_replacement_qb", pd.Series(0, index=usage.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    usage["is_replacement_player"] = pd.to_numeric(
        usage.get("is_replacement_player", pd.Series(0, index=usage.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    history["is_replacement_qb"] = pd.to_numeric(
        history.get("is_replacement_qb", pd.Series(0, index=history.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    history["is_replacement_player"] = pd.to_numeric(
        history.get("is_replacement_player", pd.Series(0, index=history.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    numerator_renames = {
        column: f"eff_{column}" for column in EFFICIENCY_NUMERATOR_COLUMNS
    }
    efficiency_labels = efficiency[
        [
            "season",
            "player_key",
            *EFFICIENCY_NUMERATOR_COLUMNS,
            *EFFICIENCY_LABEL_COLUMNS,
            *SHRUNK_EFFICIENCY_COLUMNS,
            "advanced_efficiency_available",
        ]
    ].rename(columns=numerator_renames)
    usage = usage.merge(
        efficiency_labels, on=["season", "player_key"], how="left"
    )
    usage = _merge_draft_capital(usage, seasons, source)
    usage = _merge_combine(usage, seasons, source)

    prior_columns = [
        "player_key",
        "season",
        "team",
        "pass_attempt_share",
        "target_share",
        "carry_share",
        "late_pass_attempt_share",
        "late_target_share",
        "late_carry_share",
        "games",
        "team_games",
        "pass_att",
        "targets",
        "rush_att",
        "offense_snaps",
        "team_offense_snaps",
        "snap_share",
        "qb_snap_share",
    ]
    prior = history[prior_columns].copy()
    prior["season"] += 1
    prior = prior.rename(
        columns={
            "team": "prior_team",
            "pass_attempt_share": "prior_pass_attempt_share",
            "target_share": "prior_target_share",
            "carry_share": "prior_carry_share",
            "late_pass_attempt_share": "prior_late_pass_attempt_share",
            "late_target_share": "prior_late_target_share",
            "late_carry_share": "prior_late_carry_share",
            "games": "prior_games",
            "team_games": "prior_team_games",
            "pass_att": "prior_pass_att",
            "targets": "prior_targets",
            "rush_att": "prior_rush_att",
            "offense_snaps": "prior_offense_snaps",
            "team_offense_snaps": "prior_team_offense_snaps",
            "snap_share": "prior_snap_share",
            "qb_snap_share": "prior_qb_snap_share",
        }
    )

    out = usage.merge(prior, on=["player_key", "season"], how="left")
    out = out.merge(
        lagged_efficiency_rows(efficiency),
        on=["player_key", "season"],
        how="left",
    )
    out = out[out["season"] > min(seasons)].copy()
    out["cold_start"] = out["prior_team"].isna().astype(int)
    out["team_change"] = (
        out["prior_team"].notna() & (out["prior_team"] != out["team"])
    ).astype(int)
    out["prior_availability"] = _divide(out["prior_games"], out["prior_team_games"])
    out["observed_availability"] = _divide(out["games"], out["team_games"])
    out["prior_target_per_snap"] = _divide(
        out["prior_targets"], out["prior_offense_snaps"]
    )
    out["prior_carry_per_snap"] = _divide(
        out["prior_rush_att"], out["prior_offense_snaps"]
    )
    out["prior_qb_attempts_per_snap"] = _divide(
        out["prior_pass_att"], out["prior_offense_snaps"]
    )
    out["fumble_opportunities"] = (
        pd.to_numeric(out["pass_att"], errors="coerce").fillna(0)
        + pd.to_numeric(out["targets"], errors="coerce").fillna(0)
        + pd.to_numeric(out["rush_att"], errors="coerce").fillna(0)
    )
    out["prior_fumble_opportunities"] = (
        pd.to_numeric(out["prior_pass_att"], errors="coerce").fillna(0)
        + pd.to_numeric(out["prior_targets"], errors="coerce").fillna(0)
        + pd.to_numeric(out["prior_rush_att"], errors="coerce").fillna(0)
    )
    out = add_volume_efficiency_features(out)
    out = add_season_injury_features(
        out,
        injuries=injury_reports,
        weekly_rosters=weekly_rosters,
        injury_snapshot=injury_snapshot,
        cutoff_week=injury_cutoff_week,
    )

    # Blend the stable full-season role with the more responsive late-season
    # role. Missing values remain missing so the model can apply its learned
    # position/draft cold-start prior.
    out["prior_target_role"] = (
        0.65 * out["prior_target_share"] + 0.35 * out["prior_late_target_share"]
    )
    out["prior_carry_role"] = (
        0.65 * out["prior_carry_share"] + 0.35 * out["prior_late_carry_share"]
    )
    out["prior_pass_role"] = (
        0.65 * out["prior_pass_attempt_share"]
        + 0.35 * out["prior_late_pass_attempt_share"]
    )
    qb_snap_total = out["offense_snaps"].where(out["position"].eq("QB"), 0.0).groupby(
        [out[key] for key in TEAM_KEYS]
    ).transform("sum")
    out["observed_qb_workload_share"] = np.where(
        out["position"].eq("QB"),
        np.divide(
            out["offense_snaps"],
            qb_snap_total,
            out=np.zeros(len(out), dtype=float),
            where=qb_snap_total.to_numpy(dtype=float) > 0,
        ),
        0.0,
    )
    out = _mark_primary_qb(out)

    rookie_claims = out.apply(
        lambda row: expected_rookie_claim(row.get("overall_pick"), row["position"]),
        axis=1,
        result_type="expand",
    )
    out["draft_target_prior"] = rookie_claims[0].to_numpy(dtype=float)
    out["draft_carry_prior"] = rookie_claims[1].to_numpy(dtype=float)
    out["draft_pass_prior"] = out.apply(
        lambda row: expected_rookie_pass_claim(
            row.get("overall_pick"), row["position"]
        ),
        axis=1,
    ).to_numpy(dtype=float)
    out = add_conditional_volume_efficiency_features(out)
    out["is_projection"] = out["season"].isin(projection).astype(int)
    if projection:
        # These are outcomes, and no outcome exists yet. The label merge
        # zero-fills absent rows, which on an unplayed season would present a
        # structural absence as a realized zero, so restore them to missing.
        future = out["season"].isin(projection)
        out.loc[future, [c for c in PROJECTION_BLANK_LABELS if c in out.columns]] = np.nan
        # Exposure for an unplayed season is the scheduled slate, not a realized
        # count. How much of it a player is expected to be available for is the
        # availability model's job, not an input.
        out.loc[future, "games"] = pd.to_numeric(
            out.loc[future, "team_games"], errors="coerce"
        )
    out["transition"] = (out["season"] - 1).astype(str) + "->" + out["season"].astype(str)
    return out.sort_values(PLAYER_KEYS).reset_index(drop=True)


def build_season_average_data(
    seasons: Iterable[int],
    source: str = "auto",
    *,
    roster_mode: str = "auto",
    roster_cutoff_week: int = 1,
    roster_snapshot: pd.DataFrame | None = None,
    roster_cache_dir=None,
    injury_reports: pd.DataFrame | None = None,
    weekly_rosters: pd.DataFrame | None = None,
    injury_snapshot: pd.DataFrame | None = None,
    projection_seasons: Iterable[int] = (),
) -> SeasonAverageData:
    """Build the complete season-average modeling dataset.

    ``projection_seasons`` name seasons that have not been played yet. They are
    excluded from every raw feed, require an explicit ``roster_snapshot``, and
    produce feature-only rows for forward projection.
    """
    seasons = sorted(set(int(season) for season in seasons))
    projection = {int(season) for season in projection_seasons}
    unknown = projection - set(seasons)
    if unknown:
        raise ValueError(
            f"projection_seasons must be included in seasons; missing {sorted(unknown)}"
        )
    observed = [season for season in seasons if season not in projection]
    if not observed:
        raise ValueError("at least one observed season is required for prior features")
    if roster_mode not in {"auto", "point_in_time", "inferred"}:
        raise ValueError("roster_mode must be 'auto', 'point_in_time', or 'inferred'")
    if projection and roster_snapshot is None:
        raise ValueError(
            "projection seasons have no published roster; pass roster_snapshot "
            "with rows for each projection season"
        )
    player_weeks = load_player_weeks(observed, source=source)
    teams = team_season_volume(player_weeks)
    # No upper bound: the loader drops seasons the feed will not serve, so
    # coverage follows the data rather than a constant that goes stale.
    injury_seasons = list(
        range(max(NFLVERSE_INJURY_FIRST_SEASON, min(observed) - 3), max(observed) + 1)
    )
    if injury_reports is None and source != "legacy" and injury_seasons:
        try:
            injury_reports = ingest.load_injuries(
                injury_seasons, cache_dir=roster_cache_dir
            )
        except (ingest.DataUnavailableError, OSError):
            # Injury features remain explicitly unavailable rather than
            # allowing a failed optional enrichment to block a volume run.
            injury_reports = pd.DataFrame()
    if (
        weekly_rosters is None
        and injury_reports is not None
        and not injury_reports.empty
        and injury_seasons
    ):
        try:
            weekly_rosters = ingest.load_weekly_rosters(
                injury_seasons, cache_dir=roster_cache_dir
            )
        except (ingest.DataUnavailableError, OSError):
            weekly_rosters = pd.DataFrame()
    if roster_snapshot is None and roster_mode != "inferred":
        should_try = roster_mode == "point_in_time" or source != "legacy"
        if should_try:
            try:
                roster_snapshot = load_preseason_roster_snapshot(
                    observed,
                    cutoff_week=roster_cutoff_week,
                    cache_dir=roster_cache_dir,
                )
            except (ingest.DataUnavailableError, OSError):
                if roster_mode == "point_in_time":
                    raise
    player_rows = player_preseason_rows(
        seasons,
        source=source,
        team_volume=teams,
        player_weeks=player_weeks,
        roster_snapshot=roster_snapshot,
        injury_reports=injury_reports,
        weekly_rosters=weekly_rosters,
        injury_snapshot=injury_snapshot,
        injury_cutoff_week=roster_cutoff_week,
        projection_seasons=projection,
    )
    return SeasonAverageData(
        team_rows=team_transition_rows(teams, projection_seasons=projection),
        player_rows=add_player_pathway_features(player_rows),
    )


def build_projection_data(
    projection_season: int,
    *,
    roster_snapshot: pd.DataFrame,
    history_seasons: Iterable[int] | None = None,
    history_length: int = 7,
    source: str = "auto",
    **kwargs,
) -> SeasonAverageData:
    """Build feature-only rows for a season that has not been played.

    The season-average pipeline is otherwise backtest-shaped: it scores seasons
    whose play-by-play already exists. A forward projection has no such data, so
    the row universe comes from ``roster_snapshot`` and every predictor is drawn
    from strictly prior seasons and draft capital.

    ``roster_snapshot`` must carry rows for ``projection_season``; no published
    week-1 roster exists before the season starts, so it has to be supplied (an
    archived snapshot, or one derived from preseason depth charts).
    """
    projection_season = int(projection_season)
    if history_seasons is None:
        history_seasons = range(projection_season - history_length, projection_season)
    history = sorted({int(season) for season in history_seasons})
    if not history:
        raise ValueError("history_seasons must contain at least one season")
    if projection_season in history:
        raise ValueError("projection_season must not appear in history_seasons")
    if max(history) != projection_season - 1:
        raise ValueError(
            "history must run up to the season before projection_season; "
            f"got {max(history)} for projection {projection_season}"
        )
    snapshot_seasons = set(
        pd.to_numeric(roster_snapshot["season"], errors="coerce").dropna().astype(int)
    )
    if projection_season not in snapshot_seasons:
        raise ValueError(
            f"roster_snapshot has no rows for projection season {projection_season}"
        )
    return build_season_average_data(
        history + [projection_season],
        source=source,
        roster_snapshot=roster_snapshot,
        projection_seasons=[projection_season],
        **kwargs,
    )


def _mark_primary_qb(rows: pd.DataFrame) -> pd.DataFrame:
    """Label one realized primary passer per team-season for starter fitting."""
    out = rows.copy()
    out["primary_qb"] = 0
    quarterbacks = out[out["position"].eq("QB")].copy()
    if quarterbacks.empty:
        return out
    quarterbacks["_pass"] = pd.to_numeric(
        quarterbacks.get("pass_att"), errors="coerce"
    ).fillna(0.0)
    quarterbacks["_listed"] = pd.to_numeric(
        quarterbacks.get("qb_listed_starter"), errors="coerce"
    ).fillna(0.0)
    quarterbacks["_depth"] = -pd.to_numeric(
        quarterbacks.get("qb_depth_rank"), errors="coerce"
    ).fillna(99.0)
    selected = (
        quarterbacks.sort_values(
            TEAM_KEYS + ["_pass", "_listed", "_depth", "player_key"]
        )
        .groupby(TEAM_KEYS, dropna=False)
        .tail(1)
        .index
    )
    out.loc[selected, "primary_qb"] = 1
    return out


def _replacement_player_rows(
    *,
    roster: pd.DataFrame,
    labels: pd.DataFrame,
    player_weeks: pd.DataFrame,
    snap_usage: pd.DataFrame,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate volume absent from the point-in-time roster by position.

    One stable synthetic QB/RB/WR/TE identity per team-season represents later
    signings and emergency call-ups. Zero-volume rows are retained so the
    models can learn how often each reserve bucket is actually needed.
    """
    teams = roster[TEAM_KEYS].drop_duplicates().sort_values(TEAM_KEYS).reset_index(drop=True)
    positions = pd.DataFrame({"position": list(MODEL_POSITIONS)})
    teams["_join"] = 1
    positions["_join"] = 1
    support = teams.merge(positions, on="_join").drop(columns="_join")
    roster_keys = roster[PLAYER_KEYS].drop_duplicates().assign(_on_roster=1)
    residual = labels.merge(roster_keys, on=PLAYER_KEYS, how="left")
    residual = residual[residual["_on_roster"].isna()].copy()

    aggregate_columns = {
        "pass_att": "sum",
        "targets": "sum",
        "rush_att": "sum",
        "late_pass_attempt_share": "sum",
        "late_target_share": "sum",
        "late_carry_share": "sum",
    }
    if residual.empty:
        counts = support.copy()
        for column in aggregate_columns:
            counts[column] = 0.0
    else:
        counts = (
            residual.groupby(TEAM_KEYS + ["position"], dropna=False)
            .agg(aggregate_columns)
            .reset_index()
        )
        counts = support.merge(
            counts, on=TEAM_KEYS + ["position"], how="left"
        )
        counts[list(aggregate_columns)] = counts[list(aggregate_columns)].fillna(0.0)

    totals = labels.groupby(TEAM_KEYS, dropna=False).agg(
        team_pass_att=("pass_att", "sum"),
        team_targets=("targets", "sum"),
        team_rush_att=("rush_att", "sum"),
    ).reset_index()
    counts = counts.merge(totals, on=TEAM_KEYS, how="left")
    for count, total, share in (
        ("pass_att", "team_pass_att", "pass_attempt_share"),
        ("targets", "team_targets", "target_share"),
        ("rush_att", "team_rush_att", "carry_share"),
    ):
        counts[share] = _divide(counts[count], counts[total])
        counts[share] = pd.Series(counts[share], index=counts.index).fillna(0.0)

    weeks = normalize_model_positions(player_weeks)
    weeks = _normalize_teams(weeks)
    weeks["player_key"] = crossseason.player_key(weeks)
    weeks = weeks.merge(roster_keys, on=PLAYER_KEYS, how="left")
    weeks = weeks[weeks["_on_roster"].isna()].copy()
    weeks["_volume"] = (
        pd.to_numeric(weeks["pass_att"], errors="coerce").fillna(0)
        + pd.to_numeric(weeks["rush_att"], errors="coerce").fillna(0)
        + pd.to_numeric(weeks["targets"], errors="coerce").fillna(0)
        + pd.to_numeric(weeks["receptions"], errors="coerce").fillna(0)
    )
    active = (
        weeks[weeks["_volume"].gt(0)]
        .groupby(TEAM_KEYS + ["position"], dropna=False)["week"]
        .nunique()
        .rename("games")
        .reset_index()
    )
    counts = counts.merge(active, on=TEAM_KEYS + ["position"], how="left")
    counts["games"] = counts["games"].fillna(0.0)

    residual_keys = residual[PLAYER_KEYS].drop_duplicates() if not residual.empty else pd.DataFrame(columns=PLAYER_KEYS)
    residual_snaps = snap_usage.merge(residual_keys.assign(_residual=1), on=PLAYER_KEYS, how="inner")
    if residual_snaps.empty:
        snap_totals = support.assign(
            offense_snaps=0.0,
            snap_share=0.0,
            qb_snap_share=0.0,
            snap_counts_observed=0,
        )
    else:
        snap_positions = residual[PLAYER_KEYS + ["position"]].drop_duplicates()
        residual_snaps = residual_snaps.merge(
            snap_positions, on=PLAYER_KEYS, how="left"
        )
        snap_totals = residual_snaps.groupby(
            TEAM_KEYS + ["position"], dropna=False
        ).agg(
            offense_snaps=("offense_snaps", "sum"),
            snap_share=("snap_share", "sum"),
            qb_snap_share=("qb_snap_share", "sum"),
            snap_counts_observed=("snap_counts_observed", "max"),
        ).reset_index()
        snap_totals = support.merge(
            snap_totals, on=TEAM_KEYS + ["position"], how="left"
        )
    counts = counts.merge(snap_totals, on=TEAM_KEYS + ["position"], how="left")
    observed_snap_team = snap_usage[TEAM_KEYS + ["team_offense_snaps"]].drop_duplicates(TEAM_KEYS)
    counts = counts.merge(observed_snap_team, on=TEAM_KEYS, how="left")
    for column in ("offense_snaps", "snap_share", "qb_snap_share", "snap_counts_observed"):
        counts[column] = counts[column].fillna(0.0)

    counts = counts.merge(team_games, on=TEAM_KEYS, how="left")
    counts["player_key"] = (
        "__replacement_"
        + counts["position"].str.lower()
        + "__"
        + counts["team"].astype(str)
    )
    counts["player_id"] = counts["player_key"]
    counts["player_name"] = "Replacement " + counts["position"]
    counts["age"] = np.nan
    counts["experience"] = 0.0
    counts["roster_status"] = "REPLACEMENT"
    counts["roster_active"] = 0
    counts["roster_reserve"] = 1
    counts["depth_rank"] = 99.0
    counts["qb_depth_rank"] = 99.0
    counts["qb_listed_starter"] = 0
    counts["is_replacement_player"] = 1
    counts["is_replacement_qb"] = counts["position"].eq("QB").astype(int)
    counts["roster_snapshot_week"] = roster["roster_snapshot_week"].min()
    counts["depth_snapshot_week"] = np.nan
    counts["roster_snapshot_source"] = "synthetic_replacement_bucket"
    counts["observed_roster_games"] = counts["games"]
    counts["stat_activity_games"] = counts["games"]
    counts["availability_label_source"] = "nonroster_position_activity"
    return counts.drop(
        columns=["team_pass_att", "team_targets", "team_rush_att"], errors="ignore"
    )


def load_season_snap_usage(
    seasons: Iterable[int], source: str = "auto"
) -> pd.DataFrame:
    """Season offensive-snap labels keyed to the canonical player identity.

    nflverse snap counts use PFR ids, while weekly stats and depth charts use
    GSIS ids. The nflverse player table bridges those identifiers. Outcomes in
    this table are labels only; the feature contract exposes only their lagged
    versions to a projected season.
    """
    seasons = sorted(set(map(int, seasons)))
    if source != "legacy":
        try:
            snaps = ingest.load_snap_counts(seasons)
            players = ingest.load_ids()
            return _nflverse_season_snap_usage(snaps, players)
        except (ingest.DataUnavailableError, OSError):
            if source == "nflverse":
                raise
    try:
        snaps = legacy.load_snapcounts(seasons)
    except Exception:
        snaps = pd.DataFrame()
    return _legacy_season_snap_usage(snaps)


def _nflverse_season_snap_usage(
    snaps: pd.DataFrame, players: pd.DataFrame
) -> pd.DataFrame:
    columns = PLAYER_KEYS + [
        "offense_snaps",
        "team_offense_snaps",
        "snap_share",
        "qb_snap_share",
        "snap_counts_observed",
    ]
    if snaps.empty:
        return pd.DataFrame(columns=columns)
    out = snaps.copy().rename(columns={"player": "player_name"})
    if "game_type" in out:
        out = out[out["game_type"].eq("REG")].copy()
    out["offense_snaps"] = pd.to_numeric(
        out.get("offense_snaps"), errors="coerce"
    ).fillna(0.0)
    out["offense_pct"] = pd.to_numeric(
        out.get("offense_pct"), errors="coerce"
    )
    # Estimate each team's offensive plays from the published count/percentage
    # pairs. Percentages are rounded, so the within-game median is more stable
    # than any one player's ratio; max snaps is a safe fallback.
    valid_pct = out["offense_pct"].gt(0)
    out["_team_game_snaps"] = np.where(
        valid_pct,
        out["offense_snaps"] / out["offense_pct"],
        np.nan,
    )
    game_keys = ["season", "week", "team"]
    game_totals = (
        out.groupby(game_keys, dropna=False)
        .agg(
            team_game_snaps=("_team_game_snaps", "median"),
            max_player_snaps=("offense_snaps", "max"),
        )
        .reset_index()
    )
    game_totals["team_game_snaps"] = game_totals["team_game_snaps"].fillna(
        game_totals["max_player_snaps"]
    )
    out = out.merge(game_totals[game_keys + ["team_game_snaps"]], on=game_keys)
    out["position"] = opportunity_position(out["position"])
    out = out[out["position"].isin(MODEL_POSITIONS)].copy()
    out = _normalize_teams(out)

    bridge = players[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    out = out.merge(
        bridge,
        left_on="pfr_player_id",
        right_on="pfr_id",
        how="left",
    )
    out["player_id"] = out["gsis_id"]
    out["player_key"] = crossseason.player_key(out)
    player = (
        out.groupby(PLAYER_KEYS, dropna=False)
        .agg(offense_snaps=("offense_snaps", "sum"))
        .reset_index()
    )
    team_snaps = (
        out[game_keys + ["team_game_snaps"]]
        .drop_duplicates(game_keys)
        .groupby(TEAM_KEYS, dropna=False)["team_game_snaps"]
        .sum()
        .rename("team_offense_snaps")
        .reset_index()
    )
    player = player.merge(team_snaps, on=TEAM_KEYS, how="left")
    player["snap_share"] = _divide(
        player["offense_snaps"], player["team_offense_snaps"]
    )
    qb_total = player["offense_snaps"].where(
        player["player_key"].isin(
            out.loc[out["position"].eq("QB"), "player_key"].unique()
        ),
        0.0,
    ).groupby([player[key] for key in TEAM_KEYS]).transform("sum")
    is_qb = player["player_key"].isin(
        out.loc[out["position"].eq("QB"), "player_key"].unique()
    )
    player["qb_snap_share"] = np.where(
        is_qb,
        np.divide(
            player["offense_snaps"],
            qb_total,
            out=np.zeros(len(player), dtype=float),
            where=qb_total.to_numpy(dtype=float) > 0,
        ),
        0.0,
    )
    player["snap_counts_observed"] = 1
    return player[columns].sort_values(PLAYER_KEYS).reset_index(drop=True)


def _legacy_season_snap_usage(snaps: pd.DataFrame) -> pd.DataFrame:
    columns = PLAYER_KEYS + [
        "offense_snaps",
        "team_offense_snaps",
        "snap_share",
        "qb_snap_share",
        "snap_counts_observed",
    ]
    if snaps.empty:
        return pd.DataFrame(columns=columns)
    out = snaps.copy()
    out["position"] = opportunity_position(out["position"])
    out = out[out["position"].isin(MODEL_POSITIONS)].copy()
    out = _normalize_teams(out)
    out["player_key"] = crossseason.player_key(out)
    out["offense_snaps"] = pd.to_numeric(
        out.get("snaps"), errors="coerce"
    ).fillna(0.0)
    values = out.get("snap_pct", pd.Series(np.nan, index=out.index)).astype(str).str.rstrip("%")
    out["snap_share"] = pd.to_numeric(values, errors="coerce")
    if out["snap_share"].dropna().gt(1.0).any():
        out["snap_share"] /= 100.0
    implied_team_snaps = np.divide(
        out["offense_snaps"],
        out["snap_share"],
        out=np.full(len(out), np.nan, dtype=float),
        where=out["snap_share"].to_numpy(dtype=float) > 0,
    )
    out["team_offense_snaps"] = pd.Series(implied_team_snaps).groupby(
        [out[key] for key in TEAM_KEYS]
    ).transform("median")
    qb_fallback = out["offense_snaps"].where(out["position"].eq("QB"), 0.0).groupby(
        [out[key] for key in TEAM_KEYS]
    ).transform("sum")
    out["team_offense_snaps"] = out["team_offense_snaps"].fillna(qb_fallback)
    qb_total = out["offense_snaps"].where(out["position"].eq("QB"), 0.0).groupby(
        [out[key] for key in TEAM_KEYS]
    ).transform("sum")
    out["qb_snap_share"] = np.where(
        out["position"].eq("QB"),
        np.divide(
            out["offense_snaps"],
            qb_total,
            out=np.zeros(len(out), dtype=float),
            where=qb_total.to_numpy(dtype=float) > 0,
        ),
        0.0,
    )
    out["snap_counts_observed"] = 1
    return out[columns].drop_duplicates(PLAYER_KEYS).reset_index(drop=True)


def _merge_snap_usage(rows: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    labels = [
        "offense_snaps",
        "team_offense_snaps",
        "snap_share",
        "qb_snap_share",
        "snap_counts_observed",
    ]
    out = out.drop(columns=[column for column in labels if column in out], errors="ignore")
    if snaps.empty:
        for column in labels:
            out[column] = np.nan
        return out
    out = out.merge(snaps[PLAYER_KEYS + labels], on=PLAYER_KEYS, how="left")
    observed_teams = snaps[TEAM_KEYS].drop_duplicates().assign(_snaps_available=1)
    out = out.merge(observed_teams, on=TEAM_KEYS, how="left")
    available = out["_snaps_available"].eq(1)
    for column in ("offense_snaps", "snap_share", "qb_snap_share"):
        out.loc[available, column] = out.loc[available, column].fillna(0.0)
    out["snap_counts_observed"] = available.astype(int)
    return out.drop(columns="_snaps_available")


def _merge_draft_capital(
    usage: pd.DataFrame, seasons: list[int], source: str
) -> pd.DataFrame:
    try:
        draft = load_draft_capital(seasons, source=source)
    except Exception:
        draft = pd.DataFrame()
    if draft.empty:
        usage["overall_pick"] = np.nan
        return usage
    draft = _normalize_teams(draft)

    # Identifier join first. A name is not a key, and the players whose names are
    # newest and least standardised are exactly the population a draft prior
    # exists to serve, so matching them on name is weakest where it matters most.
    picks = pd.Series(np.nan, index=usage.index, dtype=float)
    if "player_id" in draft.columns:
        keyed = draft.dropna(subset=["player_id"]).copy()
        if not keyed.empty:
            keyed["player_key"] = crossseason.player_key(keyed)
            by_key = keyed.drop_duplicates(["player_key", "season"]).set_index(
                ["player_key", "season"]
            )["overall_pick"]
            keys = pd.MultiIndex.from_arrays([usage["player_key"], usage["season"]])
            picks = pd.Series(
                pd.to_numeric(by_key.reindex(keys), errors="coerce").to_numpy(
                    dtype=float
                ),
                index=usage.index,
            )

    # Name fallback for rows no identifier could place.
    by_name = draft.drop_duplicates(["player_name", "position", "season"]).set_index(
        ["player_name", "position", "season"]
    )["overall_pick"]
    names = pd.MultiIndex.from_arrays(
        [usage["player_name"], usage["position"], usage["season"]]
    )
    named = pd.Series(
        pd.to_numeric(by_name.reindex(names), errors="coerce").to_numpy(dtype=float),
        index=usage.index,
    )
    usage["overall_pick"] = picks.where(picks.notna(), named)
    return usage


def _merge_combine(
    usage: pd.DataFrame, seasons: list[int], source: str
) -> pd.DataFrame:
    """Attach combine measurables and the shape of their absence.

    Testing happens once, before the draft, so these are permanent attributes of
    a player rather than season features and are joined on identity alone. The
    window reaches back past the earliest modelled season to cover players
    drafted well before it.
    """
    from ffmodel.features.combine import (
        combine_feature_rows,
        load_combine_measurables,
        merge_combine_features,
    )

    from ffmodel.features.athleticism import merge_athletic_score

    features = pd.DataFrame()
    if source != "legacy":
        try:
            features = combine_feature_rows(
                load_combine_measurables(range(min(seasons) - 14, max(seasons) + 1))
            )
        except Exception:
            # Athletic testing enriches the cold start; it never gates a build.
            features = pd.DataFrame()
    usage = merge_combine_features(usage, features)
    try:
        usage = merge_athletic_score(usage, features)
    except Exception:
        for column in ATHLETIC_FEATURES:
            usage[column] = np.nan
    return usage


def _divide(numerator, denominator) -> np.ndarray:
    numerator = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    denominator = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=np.isfinite(denominator) & (denominator > 0),
    )


def _normalize_teams(frame: pd.DataFrame) -> pd.DataFrame:
    """Map era/provider abbreviations onto stable franchise codes."""
    out = frame.copy()
    out["team"] = [
        team_identity(team, int(season)).franchise_code
        for team, season in zip(out["team"], out["season"])
    ]
    return out


PRESEASON_FEATURES = (
    "prior_pass_role",
    "prior_target_role",
    "prior_carry_role",
    "prior_availability",
    "prior_snap_share",
    "prior_qb_snap_share",
    "prior_target_per_snap",
    "prior_carry_per_snap",
    "prior_qb_attempts_per_snap",
    "age",
    "experience",
    "team_change",
    "cold_start",
    "roster_active",
    "roster_reserve",
    "depth_rank",
    "qb_depth_rank",
    "qb_listed_starter",
    "is_replacement_qb",
    "is_replacement_player",
    "draft_target_prior",
    "draft_carry_prior",
    "draft_pass_prior",
    # Measured before the draft, so leakage-safe for every season. Absence is
    # carried as its own signal rather than imputed: never invited and invited
    # but untested are different facts, and neither is a slow time.
    *COMBINE_FEATURES,
    # One position-normalized athletic score: RAS where supplied, a combine
    # composite otherwise. Measured before the draft, so safe for every season.
    *ATHLETIC_FEATURES,
    *PRIOR_EFFICIENCY_FEATURES,
    *VOLUME_EFFICIENCY_DERIVED_FEATURES,
    *CONDITIONAL_VOLUME_EFFICIENCY_FEATURES,
    *PLAYER_PATHWAY_FEATURES,
    *INJURY_AVAILABILITY_FEATURES,
    "prior_advanced_efficiency_available",
)
