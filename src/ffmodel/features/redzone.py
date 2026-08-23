"""Red-zone and goal-line usage, built from play-by-play.

The frame carries EPA, air yards and shrunk touchdown rates, but nothing about
*where on the field* a player's opportunities came from. That gap matters
because touchdowns are the noisiest component of fantasy scoring and the least
predictable from volume alone: two running backs with identical carry shares
score very differently if one of them is the team's goal-line back and the other
leaves the field at the five.

The quantity of interest is not red-zone volume, which is mostly a restatement
of overall volume. It is the **differential** — whether a player's share of his
team's red-zone work exceeds his share of its ordinary work. That is the part
that is not already in the frame, and it is the part that would encode "goal-line
back" as a trait rather than as a season's worth of luck.

Two zones, because they answer different questions. Inside the twenty is the
scoring-opportunity zone and gives reasonable sample sizes. Inside the five is
where touchdowns are actually decided and where personnel packages change most,
but a player-season may have only a handful of snaps there, so its share is
noisy and is reported separately rather than blended in.

Regular season only. The pipeline's postseason features are a separate,
deliberately gated signal, and mixing postseason snaps into a usage rate would
silently give players on good teams more of it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

RED_ZONE_YARDS = 20
GOAL_LINE_YARDS = 5

# Below this many team plays in a zone, a share is not a rate — it is one or two
# plays. The column is left missing rather than reported as 0.0 or 1.0.
MIN_TEAM_PLAYS = 12

REDZONE_FEATURES = (
    "prior_redzone_carry_share_diff",
    "prior_redzone_target_share_diff",
    "prior_goalline_carry_share_diff",
    "prior_goalline_target_share_diff",
)


def _usage(plays: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per player-team-season carry and target counts, plus team totals."""
    runs = plays[plays.rush_attempt.eq(1) & plays.rusher_player_id.notna()]
    passes = plays[plays.pass_attempt.eq(1) & plays.receiver_player_id.notna()]
    carries = (
        runs.groupby(["season", "posteam", "rusher_player_id"], as_index=False)
        .size()
        .rename(columns={"rusher_player_id": "player_id", "size": f"{prefix}_carries"})
    )
    targets = (
        passes.groupby(["season", "posteam", "receiver_player_id"], as_index=False)
        .size()
        .rename(columns={"receiver_player_id": "player_id", "size": f"{prefix}_targets"})
    )
    out = carries.merge(targets, on=["season", "posteam", "player_id"], how="outer")
    out[[f"{prefix}_carries", f"{prefix}_targets"]] = out[
        [f"{prefix}_carries", f"{prefix}_targets"]
    ].fillna(0.0)
    team = out.groupby(["season", "posteam"], as_index=False)[
        [f"{prefix}_carries", f"{prefix}_targets"]
    ].sum()
    team = team.rename(
        columns={
            f"{prefix}_carries": f"{prefix}_team_carries",
            f"{prefix}_targets": f"{prefix}_team_targets",
        }
    )
    return out.merge(team, on=["season", "posteam"], how="left")


def _share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Share, missing where the denominator is too small to be a rate."""
    usable = denominator.ge(MIN_TEAM_PLAYS)
    return pd.Series(
        np.where(usable, numerator / denominator.replace(0, np.nan), np.nan),
        index=numerator.index,
        dtype=float,
    )


def zone_usage(seasons: Iterable[int], load=None) -> pd.DataFrame:
    """Player-team-season shares of team work, overall and by field zone.

    Loaded a season at a time and reduced immediately: play-by-play is 372
    columns wide and holding ten seasons of it costs far more memory than the
    six columns this needs.
    """
    if load is None:
        from ffmodel.data import ingest

        load = lambda years: ingest.load_pbp(years)

    columns = [
        "season",
        "posteam",
        "play_type",
        "season_type",
        "yardline_100",
        "rush_attempt",
        "pass_attempt",
        "rusher_player_id",
        "receiver_player_id",
    ]
    blocks = []
    for season in sorted({int(s) for s in seasons}):
        plays = load([season])
        if "season" not in plays:
            plays = plays.assign(season=season)
        keep = [c for c in columns if c in plays]
        plays = plays[keep].copy()
        if "season_type" in plays:
            plays = plays[plays.season_type.astype(str).eq("REG")]
        plays = plays[plays.play_type.isin(["run", "pass"])]
        plays["yardline_100"] = pd.to_numeric(plays.yardline_100, errors="coerce")

        overall = _usage(plays, "all")
        merged = overall
        for prefix, zone in (
            ("rz", plays[plays.yardline_100.le(RED_ZONE_YARDS)]),
            ("gl", plays[plays.yardline_100.le(GOAL_LINE_YARDS)]),
        ):
            usage = _usage(zone, prefix)
            player = usage[
                ["season", "posteam", "player_id", f"{prefix}_carries", f"{prefix}_targets"]
            ]
            # Team totals join on the team, not on the player. A back who never
            # got a goal-line carry has a share of zero on a team with thirty of
            # them; taking the denominator off his own (absent) row would leave
            # it missing instead, which is a different claim and would drop half
            # the population.
            team = (
                usage[
                    [
                        "season",
                        "posteam",
                        f"{prefix}_team_carries",
                        f"{prefix}_team_targets",
                    ]
                ]
                .drop_duplicates(["season", "posteam"])
            )
            merged = merged.merge(
                player, on=["season", "posteam", "player_id"], how="left"
            ).merge(team, on=["season", "posteam"], how="left")
        blocks.append(merged)

    frame = pd.concat(blocks, ignore_index=True)
    for prefix in ("all", "rz", "gl"):
        for stream in ("carries", "targets"):
            frame[f"{prefix}_{stream}"] = frame[f"{prefix}_{stream}"].fillna(0.0)
            team = frame[f"{prefix}_team_{stream}"]
            frame[f"{prefix}_{stream}_share"] = _share(
                frame[f"{prefix}_{stream}"], team.fillna(0.0)
            )

    # The differential is the whole point: how much more of the team's scoring
    # work a player takes than his ordinary workload would imply. A pure
    # red-zone share is mostly a restatement of overall share and would tell the
    # model something it already knows.
    for zone in ("rz", "gl"):
        for stream in ("carries", "targets"):
            frame[f"{zone}_{stream}_share_diff"] = (
                frame[f"{zone}_{stream}_share"] - frame[f"all_{stream}_share"]
            )
    return frame


def add_redzone_features(
    rows: pd.DataFrame, usage: pd.DataFrame | None = None, seasons=None
) -> pd.DataFrame:
    """Attach *prior-season* zone differentials to player-season rows.

    Prior season, so the feature is available before the season it forecasts.
    Joined on player and season only, not team: a player who changed teams keeps
    the goal-line role he earned elsewhere as evidence about him, which is the
    trait the feature is trying to capture.
    """
    out = rows.copy()
    if usage is None:
        needed = pd.to_numeric(out.get("season"), errors="coerce").dropna().unique()
        usage = zone_usage(sorted({int(s) - 1 for s in needed}))

    prior = (
        usage.groupby(["season", "player_id"], as_index=False)[
            [
                "rz_carries_share_diff",
                "rz_targets_share_diff",
                "gl_carries_share_diff",
                "gl_targets_share_diff",
            ]
        ]
        .mean()
    )
    prior["season"] = prior["season"] + 1
    prior = prior.rename(
        columns={
            "rz_carries_share_diff": "prior_redzone_carry_share_diff",
            "rz_targets_share_diff": "prior_redzone_target_share_diff",
            "gl_carries_share_diff": "prior_goalline_carry_share_diff",
            "gl_targets_share_diff": "prior_goalline_target_share_diff",
        }
    )
    before = len(out)
    out = out.merge(prior, on=["season", "player_id"], how="left")
    if len(out) != before:
        raise AssertionError(
            f"the red-zone join changed the row count {before} -> {len(out)}; "
            "usage has duplicate player-seasons"
        )
    return out
