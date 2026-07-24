"""Open-data pass-play participation proxies for route opportunity.

nflverse participation identifies every offensive player on a play and the
targeted route, but it does not chart a route assignment for every receiver.
We therefore call this pass-play participation rather than routes run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def season_pass_play_participation(
    participation: pd.DataFrame, players: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Count targeted pass plays on which each skill player was on offense."""
    required = {"nflverse_game_id", "offense_players", "route"}
    missing = required - set(participation.columns)
    if missing:
        raise ValueError(f"participation rows are missing columns: {sorted(missing)}")

    position_lookup: dict[str, str] = {}
    if players is not None and not players.empty and "gsis_id" in players:
        source = players.get("position", players.get("position_group"))
        position_lookup = dict(
            zip(players["gsis_id"].astype(str), source.fillna("").astype(str))
        )

    rows: list[tuple[int, str, str]] = []
    frame = participation.copy()
    route = frame["route"].fillna("").astype(str).str.strip()
    frame = frame[route.ne("")]
    if "offense_positions" not in frame:
        frame["offense_positions"] = pd.NA
    for game_id, player_text, position_text in frame[
        ["nflverse_game_id", "offense_players", "offense_positions"]
    ].itertuples(index=False, name=None):
        try:
            season = int(str(game_id)[:4])
        except (TypeError, ValueError):
            continue
        play_players = str(player_text).split(";")
        if pd.notna(position_text):
            positions = str(position_text).split(";")
        else:
            positions = [position_lookup.get(player, "") for player in play_players]
        if len(play_players) != len(positions):
            continue
        for player_key, position in zip(play_players, positions):
            raw_position = position.upper().strip()
            if "QB" in raw_position:
                normalized = "QB"
            elif "WR" in raw_position:
                normalized = "WR"
            elif "TE" in raw_position:
                normalized = "TE"
            elif any(label in raw_position for label in ("RB", "HB", "FB")):
                normalized = "RB"
            else:
                normalized = "OTHER"
            if normalized in {"QB", "RB", "WR", "TE"} and player_key:
                rows.append((season, player_key, normalized))
    if not rows:
        return pd.DataFrame(
            columns=["season", "player_key", "position", "pass_play_opportunities"]
        )
    expanded = pd.DataFrame(rows, columns=["season", "player_key", "position"])
    return (
        expanded.groupby(["season", "player_key", "position"], dropna=False)
        .size()
        .rename("pass_play_opportunities")
        .reset_index()
    )


def attach_lagged_participation_features(
    player_rows: pd.DataFrame,
    participation: pd.DataFrame,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach Y-1 pass-play exposure and target rate to preseason Y rows."""
    prior = season_pass_play_participation(participation, players=players)
    prior["season"] = pd.to_numeric(prior["season"], errors="raise").astype(int) + 1
    prior = prior.rename(
        columns={"pass_play_opportunities": "prior_pass_play_opportunities"}
    ).drop(columns="position")
    out = player_rows.merge(prior, on=["season", "player_key"], how="left")
    denominator = pd.to_numeric(
        out["prior_pass_play_opportunities"], errors="coerce"
    ).to_numpy(dtype=float)
    numerator = pd.to_numeric(out["prior_targets"], errors="coerce").to_numpy(
        dtype=float
    )
    out["prior_targets_per_pass_play"] = np.divide(
        numerator,
        denominator,
        out=np.full(len(out), np.nan, dtype=float),
        where=np.isfinite(denominator) & (denominator > 0),
    )
    out["prior_participation_available"] = (
        pd.to_numeric(out["prior_pass_play_opportunities"], errors="coerce")
        .notna()
        .astype(int)
    )
    return out
