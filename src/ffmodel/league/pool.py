"""One table of every draftable player-week, across all six positions.

The three panels this package already builds cover different populations and
were built for different questions: the skill panel is one row per rostered
QB/RB/WR/TE per team-week, the kicker panel one row per rostered kicker, the
defense panel one row per club. A league needs them stacked, because a roster
holds all six and a lineup decision compares a tight end against a flex-eligible
back on the same scale.

It also needs a draft board, and that is where the existing ADP loader stops
being usable: :func:`ffmodel.features.market.read_adp_file` filters to
``MODEL_POSITIONS`` -- the four skill positions -- and caps at rank 300, which
is correct for a feature that feeds the weekly model and useless for a draft
that must seat a kicker and a defense on every one of twelve rosters. So this
module reads the same files with the position filter removed.

**Defenses join on team, not name.** The ADP file lists them as
``"San Francisco 49ers DST   (9)"`` while the defense panel keys on the club
code ``SF``, so the join runs through nflverse's own team table rather than
through a name normaliser that would have to guess.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ffmodel.features.market import DEFAULT_ADP_DIR, _name_key
from ffmodel.league.config import POSITIONS

# Columns every panel contributes, so the stacked frame has one schema.
POOL_COLUMNS = (
    "season",
    "week",
    "team",
    "player_key",
    "player_name",
    "position",
    "points",
    "played",
)


def read_adp_all_positions(path: Path) -> pd.DataFrame:
    """One season's board, keeping kickers and defenses.

    Mirrors :func:`ffmodel.features.market.read_adp_file` except for the two
    choices that make that function unsuitable here: it keeps every position,
    and it does not cap the rank, because a twelve-team league drafting fifteen
    rounds seats 180 players and the tail of the board is exactly where the
    last few of those come from.
    """
    raw = pd.read_csv(path)
    player = raw["Player (Bye)"].astype(str).str.strip()
    # "Name TEAM (BYE)" in recent files, a bare name in older ones.
    parsed = player.str.extract(r"^(?P<name>.*?)\s+[A-Z]{2,3}\s*\(\w+\)$")["name"]
    name = parsed.fillna(player).str.strip()
    position = raw["POS"].astype(str).str.extract(r"^([A-Z]+)")[0]

    out = pd.DataFrame(
        {
            "adp_name": name,
            "key": _name_key(name),
            "position": position,
            "adp_rank": pd.to_numeric(raw["Rank"], errors="coerce"),
        }
    )
    out = out[out["position"].isin(POSITIONS) & out["adp_rank"].notna()]
    # Two players who normalise to the same key cannot be told apart; giving
    # one of them the other's draft slot is worse than leaving both unranked.
    duplicated = out["key"].duplicated(keep=False) & out["position"].ne("DST")
    return out[~duplicated].reset_index(drop=True)


def _franchise_ids() -> tuple[dict[str, str], dict[str, str]]:
    """Club name -> franchise id, and club code -> franchise id.

    The defense join cannot go through the team *code*, because the ADP files
    name franchises as they are known now rather than as they were known then:
    the 2016 board lists "Las Vegas Raiders" for a club the 2016 panel codes
    ``OAK``, and does the same for the Chargers and the Rams. Matching the
    modern name against a modern code would silently drop every relocated
    franchise from every older season -- four of thirty-two, and only in the
    seasons where they moved, which is exactly the kind of gap that looks like
    noise rather than a bug.

    nflverse's ``team_id`` is the franchise identity and survives the move:
    ``OAK`` and ``LV`` are both 2520, ``SD`` and ``LAC`` both 4400, ``STL``,
    ``LA`` and ``LAR`` all 2510. So the join runs through that instead.
    """
    import nflreadpy as nfl

    teams = nfl.load_teams().to_pandas()
    needed = {"team_abbr", "team_name", "team_id"}
    missing = needed - set(teams.columns)
    if missing:
        raise ValueError(f"nflverse team table is missing {sorted(missing)}")
    teams = teams.dropna(subset=["team_abbr", "team_name", "team_id"])
    ids = teams["team_id"].astype(str)
    by_name = dict(zip(teams["team_name"].astype(str), ids))
    by_abbr = dict(zip(teams["team_abbr"].astype(str), ids))
    if not by_name or not by_abbr:
        raise ValueError("nflverse team table produced no franchise mapping")
    return by_name, by_abbr


def attach_adp_ranks(pool: pd.DataFrame, directory: Path | str = DEFAULT_ADP_DIR):
    """Attach the draft board to a stacked pool, one rank per player-season."""
    directory = Path(directory)
    seasons = sorted({int(s) for s in pool["season"].unique()})

    boards = []
    for season in seasons:
        path = directory / f"FantasyPros_{season}_Overall_ADP_Rankings.csv"
        if not path.exists():
            raise FileNotFoundError(f"no ADP list for {season} at {path}")
        block = read_adp_all_positions(path)
        block["season"] = season
        boards.append(block)
    board = pd.concat(boards, ignore_index=True)

    # Skill players and kickers join on the normalised name.
    named = board[board["position"] != "DST"].copy()
    frame = pool.copy()
    frame["key"] = _name_key(frame["player_name"].fillna(""))
    merged = frame.merge(
        named[["season", "key", "position", "adp_rank"]],
        on=["season", "key", "position"],
        how="left",
    )

    # Defenses join on franchise identity rather than club code or name.
    defenses = board[board["position"] == "DST"].copy()
    is_dst = merged["position"].eq("DST")
    if not defenses.empty and is_dst.any():
        by_name, by_abbr = _franchise_ids()
        stripped = (
            defenses["adp_name"]
            .str.replace(r"\s*DST\s*$", "", regex=True)
            .str.strip()
        )
        defenses["franchise"] = stripped.map(by_name)
        unmatched = stripped[defenses["franchise"].isna()].unique().tolist()
        if unmatched:
            raise ValueError(
                f"{len(unmatched)} defense name(s) not in the nflverse team "
                f"table: {unmatched}. A silently unmatched defense is a roster "
                "slot no team can fill."
            )
        lookup = defenses.set_index(["season", "franchise"])["adp_rank"]
        franchise = merged.loc[is_dst, "team"].astype(str).map(by_abbr)
        keys = pd.MultiIndex.from_arrays(
            [merged.loc[is_dst, "season"], franchise]
        )
        merged.loc[is_dst, "adp_rank"] = lookup.reindex(keys).to_numpy()

    merged["adp_drafted"] = merged["adp_rank"].notna()
    return merged.drop(columns=["key"], errors="ignore")


def build_player_pool(
    seasons,
    *,
    features_cache: Path | None = None,
    kicker_cache: Path | None = None,
    defense_cache: Path | None = None,
    adp_directory: Path | str = DEFAULT_ADP_DIR,
) -> pd.DataFrame:
    """Every draftable player-week for the requested seasons, with a draft board.

    Reads the cached panels when they are available, because rebuilding three
    panels from nflverse takes minutes and the league only needs the columns
    below. The caches are the same ones ``scripts/validate_weekly.py`` and
    ``scripts/validate_specialists.py`` write.
    """
    from ffmodel.weekly import FEATURES_CACHE

    seasons = sorted({int(s) for s in seasons})
    features_cache = FEATURES_CACHE if features_cache is None else Path(features_cache)
    kicker_cache = (
        Path(".cache/weekly_kickers_2016_2025.pkl")
        if kicker_cache is None
        else Path(kicker_cache)
    )
    defense_cache = (
        Path(".cache/weekly_defenses_2016_2025.pkl")
        if defense_cache is None
        else Path(defense_cache)
    )

    frames = []
    for path, label in (
        (features_cache, "skill"),
        (kicker_cache, "kicker"),
        (defense_cache, "defense"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"no {label} panel at {path}. Build it with "
                "scripts/validate_weekly.py or scripts/validate_specialists.py."
            )
        block = pd.read_pickle(path)
        block = block[block["season"].isin(seasons)]
        missing = [c for c in POOL_COLUMNS if c not in block.columns]
        if missing:
            raise ValueError(f"{label} panel is missing {missing}")
        frames.append(block[list(POOL_COLUMNS)].copy())

    pool = pd.concat(frames, ignore_index=True)
    pool["points"] = pd.to_numeric(pool["points"], errors="coerce").fillna(0.0)
    pool["played"] = pd.to_numeric(pool["played"], errors="coerce").fillna(0).astype(int)
    pool = pool[pool["position"].isin(POSITIONS)]
    # A defense's name is its club; a skill player missing one cannot be drafted
    # by a policy that has to show it to somebody.
    pool["player_name"] = pool["player_name"].fillna(pool["player_key"])
    return attach_adp_ranks(pool, adp_directory).reset_index(drop=True)
