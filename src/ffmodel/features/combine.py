"""Combine measurables, and the fact of their absence.

Athletic testing is committed in this repo but was never read by any feature.
It is a second cold-start signal beyond draft slot, which otherwise carries a
rookie's projection alone.

Absence is not missing at random and is not one thing. A player outside the
combine feed was not invited, which is itself a scouting verdict. A player
inside it with no time recorded was invited and did not test — usually injury,
sometimes a deliberate hold for a pro day. Both differ from a slow time, so
they are exposed as their own features and no measurement is ever imputed.
That follows the same rule the ingest layer applies to sacks: an unmeasured
quantity must not reach a model as a measured one.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Drills, and whether a larger number is better. Height and weight are recorded
# for nearly everyone and are size rather than performance, so they are kept
# separate from the drill-completion count.
COMBINE_DRILLS = {
    "forty": False,
    "vertical": True,
    "broad_jump": True,
    "bench": True,
    "cone": False,
    "shuttle": False,
}
COMBINE_SIZE = ("ht", "wt")

COMBINE_FEATURES = (
    *(f"combine_{drill}" for drill in COMBINE_DRILLS),
    *(f"combine_{column}" for column in COMBINE_SIZE),
    "combine_invited",
    "combine_drills_completed",
    "combine_tested",
)


def load_combine_measurables(
    seasons: Iterable[int], *, refresh: bool = False, cache_dir=None
) -> pd.DataFrame:
    """Per-player combine measurables keyed by canonical player id where possible."""
    from ffmodel.data import ingest
    from ffmodel.data.identity import resolve_player_ids

    seasons = sorted({int(season) for season in seasons})
    frames = []
    for season in seasons:
        try:
            frames.append(ingest.load_combine([season], refresh=refresh, cache_dir=cache_dir))
        except Exception:
            # A season with no published combine is a real state, not an error:
            # the rows simply do not exist yet for an upcoming class.
            continue
    if not frames:
        return pd.DataFrame(columns=["season", "player_name", "position", "player_id"])
    raw = pd.concat(frames, ignore_index=True)

    out = raw.rename(columns={"pos": "position", "pfr_id": "pfr_player_id"}).copy()
    out["season"] = pd.to_numeric(out.get("season"), errors="coerce")
    out["position"] = out["position"].astype(str).str.upper()
    try:
        out["player_id"] = resolve_player_ids(out, refresh=refresh, cache_dir=cache_dir)
    except Exception:
        out["player_id"] = pd.NA
    return out


def parse_height_inches(values: pd.Series) -> pd.Series:
    """Convert ``feet-inches`` strings such as ``6-5`` to total inches.

    The feed records height this way, so reading it as a plain number silently
    discards every row rather than failing.
    """
    text = values.astype("string").str.strip()
    parts = text.str.extract(r"^(\d+)\s*-\s*(\d+(?:\.\d+)?)$")
    feet = pd.to_numeric(parts[0], errors="coerce")
    inches = pd.to_numeric(parts[1], errors="coerce")
    combined = feet * 12.0 + inches
    # Anything already numeric is passed through unchanged.
    return combined.where(combined.notna(), pd.to_numeric(text, errors="coerce"))


def combine_feature_rows(measurables: pd.DataFrame) -> pd.DataFrame:
    """Reshape raw measurables into the modelled feature columns.

    Every player present here was invited. ``combine_drills_completed`` counts
    the timed and jumped events they actually finished, so an invitee who tested
    nothing is distinguishable from one who ran a slow forty.
    """
    if measurables.empty:
        return pd.DataFrame(columns=["season", "player_name", "position", "player_id", *COMBINE_FEATURES])

    out = measurables.copy()
    drills = []
    for drill in COMBINE_DRILLS:
        column = f"combine_{drill}"
        out[column] = pd.to_numeric(out.get(drill), errors="coerce")
        drills.append(column)
    out["combine_ht"] = parse_height_inches(out.get("ht", pd.Series(dtype="string")))
    out["combine_wt"] = pd.to_numeric(out.get("wt"), errors="coerce")

    out["combine_invited"] = 1
    out["combine_drills_completed"] = out[drills].notna().sum(axis=1).astype(float)
    out["combine_tested"] = (out["combine_drills_completed"] > 0).astype(float)
    keep = ["season", "player_name", "position", "player_id", *COMBINE_FEATURES]
    return out[[column for column in keep if column in out.columns]]


def merge_combine_features(
    rows: pd.DataFrame, features: pd.DataFrame, *, key: str = "player_key"
) -> pd.DataFrame:
    """Attach combine features, marking players the feed never listed.

    Rows that do not match are not invitees, so ``combine_invited`` is zero
    rather than missing — that is a known fact about the player. The
    measurements stay missing, because none were taken.
    """
    out = rows.copy()
    if features.empty or key not in out.columns:
        for column in COMBINE_FEATURES:
            out[column] = 0.0 if column == "combine_invited" else np.nan
        return out

    keyed = features.dropna(subset=["player_id"]).copy()
    keyed = keyed.drop_duplicates("player_id").set_index("player_id")
    lookup = out[key].astype("string")
    for column in COMBINE_FEATURES:
        if column in keyed.columns:
            out[column] = lookup.map(keyed[column]).astype(float)
        else:
            out[column] = np.nan

    # Never invited is a scouting signal; a missing flag would be read as unknown.
    out["combine_invited"] = out["combine_invited"].fillna(0.0)
    out["combine_tested"] = out["combine_tested"].where(
        out["combine_invited"].gt(0), 0.0
    )
    out["combine_drills_completed"] = out["combine_drills_completed"].where(
        out["combine_invited"].gt(0), 0.0
    )
    return out
