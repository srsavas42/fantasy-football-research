"""Shared cached nflverse frames for the walk-forward validation scripts.

Building the season-average dataset takes a couple of minutes and every
comparison needs the *same* frames — a candidate scored against a baseline built
from a separate pull is not a controlled comparison. These helpers cache one
build and hand the identical frames to every configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CACHE = Path(".cache/ffmodel-walkforward")
# 2025 is complete as of this writing and is deliberately *not* in the default
# holdouts. Every promotion decision in this package was made on 2022-2024, so
# 2025 is the one season no choice here has seen — the closest thing to a real
# out-of-sample test. Keep it that way: score it, do not select on it.
DEFAULT_SEASONS = range(2014, 2026)
FINGERPRINT_VERSION = 2
HOLDOUTS = (2022, 2023, 2024)


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("label", help="name for the output JSON, e.g. 'baseline'")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--holdouts", nargs="+", type=int, default=list(HOLDOUTS)
    )
    parser.add_argument(
        "--couple-gate",
        action="store_true",
        help="force the availability-coupled QB gate on (it is on by default)",
    )
    parser.add_argument(
        "--no-couple-gate",
        action="store_true",
        help="force the availability-coupled QB gate off, for ablations",
    )
    parser.add_argument(
        "--play-transition",
        action="store_true",
        help="restore the per-row play-rate random effect (redundant with the "
             "NegativeBinomial's own dispersion; off by default)",
    )
    # Tri-state on purpose. ``store_true`` here silently overrode a *promoted*
    # model default with the flag's own default of False, so every run that
    # forgot the flag validated a configuration nobody ships. Leaving this None
    # keeps whatever the model itself says.
    parser.add_argument(
        "--postseason",
        dest="postseason",
        action="store_const",
        const=True,
        default=None,
        help="force the lagged postseason role features on (promoted, so this "
             "is already the model default)",
    )
    parser.add_argument(
        "--no-postseason",
        dest="postseason",
        action="store_const",
        const=False,
        help="force the lagged postseason role features off, for ablations",
    )
    parser.add_argument(
        "--injury-availability",
        dest="injury_availability",
        action="store_const",
        const=True,
        default=None,
        help="let the availability regression read injury history and the "
             "preseason injury snapshot",
    )
    parser.add_argument(
        "--no-injury-availability",
        dest="injury_availability",
        action="store_const",
        const=False,
        help="force the injury features out, for the paired arm",
    )
    parser.add_argument(
        "--market-adp",
        dest="market_adp",
        action="store_const",
        const=True,
        default=None,
        help="force preseason ADP into the role and playing-time regressions",
    )
    parser.add_argument(
        "--no-market-adp",
        dest="market_adp",
        action="store_const",
        const=False,
        help="force preseason ADP out, for the paired arm of the ablation",
    )
    parser.add_argument(
        "--market-adp-availability",
        dest="market_adp_availability",
        action="store_const",
        const=True,
        default=None,
        help="give the availability regression the preseason draft board. The "
             "--market-adp arm deliberately excludes this layer, so its result "
             "says nothing about this one",
    )
    parser.add_argument(
        "--no-market-adp-availability",
        dest="market_adp_availability",
        action="store_const",
        const=False,
        help="force the board out of the availability regression, for the "
             "paired arm",
    )
    parser.add_argument(
        "--availability-target",
        choices=("roster", "snap"),
        default=None,
        help="which exposure the availability and snap layers are built on: "
             "'roster' is games (roster-active weeks, the historical default), "
             "'snap' is snap_games (weeks with an offensive snap). Unset keeps "
             "the model's own value. Note that this changes what availability "
             "*means*, so layer-level availability metrics are not comparable "
             "across it -- judge it on total season points",
    )
    parser.add_argument(
        "--market-win-totals",
        dest="market_win_totals",
        action="store_const",
        const=True,
        default=None,
        help="give the team layer the preseason Vegas win total, as a "
             "within-season z-score (requires a cache built with "
             "augment_cache_features.py --feature win-totals)",
    )
    parser.add_argument(
        "--no-market-win-totals",
        dest="market_win_totals",
        action="store_const",
        const=False,
        help="force the win totals out of the team layer, for the paired arm",
    )
    parser.add_argument(
        "--market-adp-qb",
        dest="market_adp_qb",
        action="store_const",
        const=True,
        default=None,
        help="give the quarterback room the preseason draft board: the passing "
             "share softmax, its hurdle, and pass attempts per snap",
    )
    parser.add_argument(
        "--no-market-adp-qb",
        dest="market_adp_qb",
        action="store_const",
        const=False,
        help="force the board out of the quarterback room, for the paired arm",
    )
    parser.add_argument(
        "--market-adp-interactions",
        dest="market_adp_interactions",
        action="store_const",
        const=True,
        default=None,
        help="give each position its own ADP rank slope and drafted effect "
             "(requires --market-adp)",
    )
    parser.add_argument(
        "--no-market-adp-interactions",
        dest="market_adp_interactions",
        action="store_const",
        const=False,
        help="force the per-position ADP terms off",
    )
    parser.add_argument(
        "--innovation-cap",
        type=float,
        default=None,
        help="override the target and carry allocators' role-innovation cap",
    )
    parser.add_argument(
        "--calibrated-innovation",
        action="store_true",
        help="solve for the input noise scale that realizes the churn the "
             "estimator measured, instead of using the measurement directly",
    )
    parser.add_argument(
        "--cold-role-innovation",
        dest="cold_role_innovation",
        action="store_const",
        const=True,
        default=None,
        help="give players with no prior role of their own a wider role "
             "innovation, sized from the training data's own cold-vs-warm "
             "log-share dispersion ratio (promoted, so already the default)",
    )
    parser.add_argument(
        "--no-cold-role-innovation",
        dest="cold_role_innovation",
        action="store_const",
        const=False,
        help="force the cold-role widening off, for ablations",
    )
    parser.add_argument(
        "--legacy-rookie-prior",
        action="store_true",
        help="rebuild the draft priors from the pre-2026-09 share-fit curves, "
             "for the paired arm of the rookie-prior refit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip holdouts already present in this label's report, provided "
             "its frames fingerprint matches this run",
    )
    parser.add_argument(
        "--coaching-scheme",
        dest="coaching_scheme",
        action="store_const",
        const=True,
        default=None,
        help="let the target softmax read the scheme carrier's carried "
             "backfield tendency, interacted with the back indicator "
             "(see ffmodel.features.coaching_scheme)",
    )
    parser.add_argument(
        "--no-coaching-scheme",
        dest="coaching_scheme",
        action="store_const",
        const=False,
        help="force the coaching-scheme feature out, for the paired arm",
    )
    parser.add_argument(
        "--room-structure",
        dest="room_structure",
        action="store_const",
        const=True,
        default=None,
        help="let the target softmax read within-room structure -- the "
             "player's share of his own positional room and his receiving "
             "quality against the rest of it (see TARGET_ROOM_FEATURES)",
    )
    parser.add_argument(
        "--no-room-structure",
        dest="room_structure",
        action="store_const",
        const=False,
        help="force the room-structure features out, for the paired arm",
    )
    parser.add_argument(
        "--teammate-quality",
        action="store_true",
        help="let the receiving efficiency responses read the projected "
             "starting quarterback's prior passing quality (scoring runs only)",
    )
    parser.add_argument(
        "--snap-feature-prior",
        type=float,
        default=None,
        help="width of the snap model's projected-feature prior. 0.35 is the "
             "historical value; wider lets depth_rank and is_replacement_player "
             "off the leash, which buys held-out snap MAE on RB and WR and "
             "costs backup quarterbacks",
    )
    parser.add_argument(
        "--cold-role-scale-mode",
        choices=("relative", "measured"),
        default=None,
        help="how the cold rows' scale is derived: 'relative' keeps the cold-"
             "to-warm dispersion ratio and inherits the cap's compression, "
             "'measured' targets the cold population's own dispersion. Unset "
             "keeps the model's own value, which is 'measured' and promoted. A "
             "default here silently overrides it, which is exactly what "
             "--postseason did before it was made tri-state",
    )
    parser.add_argument(
        "--mean-preserving-innovation",
        action="store_true",
        help="correct the softmax renormalization bias in every allocation "
             "layer (workload, target and carry)",
    )
    parser.add_argument(
        "--mean-preserving-layers",
        nargs="+",
        default=None,
        choices=("workload", "target", "carry"),
        help="apply the correction to only these layers. The layers disagree: "
             "the workload one costs 4-7%% CRPS on the passing streams while "
             "the carry one improves carry MAE on every fold",
    )
    parser.add_argument(
        "--efficiency-exposure-floor",
        type=int,
        default=None,
        help="lower every efficiency response's training exposure floor to at "
             "most this many opportunities (scoring runs only)",
    )
    parser.add_argument(
        "--volume-feature-draws",
        type=int,
        default=200,
        help="sampler budget for the per-season cross-fits that build the "
             "training-time oof_* covariates under --volume-feature-estimator "
             "pipeline. Only their posterior *mean* is consumed, so this can be "
             "far cheaper than the fits being validated (scoring runs only)",
    )
    parser.add_argument(
        "--volume-feature-estimator",
        choices=("ridge", "pipeline"),
        default=None,
        help="how training-time oof_* volume covariates are built (scoring runs only)",
    )
    return parser


def frames_fingerprint(
    player_rows: pd.DataFrame, team_rows: pd.DataFrame, cache_dir: Path
) -> dict[str, object]:
    """Identify the build a run read, so cross-cache comparisons are catchable.

    Two caches covering the same seasons are not the same data. nflverse revises
    history, so a rebuild changes values under an unchanged row count -- between
    the two caches in this repo, 69 of 289 columns differ over the pre-2022
    seasons that both builds cover, with identical row counts. Absolute levels
    from runs on different builds are not comparable, and nothing about the
    output says so. Recording a content hash lets the gate say so instead.
    """
    # Salted per position so the combine is not commutative: a plain XOR gives
    # the same digest if the two frames are passed the other way round, which
    # would make an argument-order mistake invisible to the very check meant to
    # catch invisible mistakes. Column order is normalised because it carries no
    # information; row order is not, because these come from a stable build and
    # a reordering is a real difference worth flagging.
    digest = 0
    for position, frame in enumerate((player_rows, team_rows)):
        ordered = frame.sort_index(axis=1)
        frame_hash = int(pd.util.hash_pandas_object(ordered, index=False).sum())
        digest ^= (frame_hash * (2 * position + 1)) & 0xFFFFFFFFFFFFFFFF
    return {
        # Bumped whenever the hashing changes. Without it, a digest computed by
        # an older version reads as "different data" rather than "different
        # method", and the gate would block a comparison that is actually fine.
        "version": FINGERPRINT_VERSION,
        "cache_dir": str(cache_dir),
        "player_rows": int(len(player_rows)),
        "team_rows": int(len(team_rows)),
        "seasons": [int(player_rows.season.min()), int(player_rows.season.max())],
        "digest": f"{digest & 0xFFFFFFFFFFFFFFFF:016x}",
    }


def load_frames(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached player/team season rows, building them on first use."""
    cache_dir = Path(cache_dir)
    player_path = cache_dir / "player_rows.pkl"
    team_path = cache_dir / "team_rows.pkl"
    if player_path.exists() and team_path.exists():
        return pd.read_pickle(player_path), pd.read_pickle(team_path)

    from ffmodel.features.season_average import build_season_average_data

    data = build_season_average_data(
        DEFAULT_SEASONS, source="nflverse", roster_mode="point_in_time"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    data.player_rows.to_pickle(player_path)
    data.team_rows.to_pickle(team_path)
    return data.player_rows, data.team_rows


def gate_override(args: argparse.Namespace) -> bool | None:
    """Explicit coupling choice, or None to keep the model's own default."""
    if args.couple_gate and args.no_couple_gate:
        raise SystemExit("choose at most one of --couple-gate / --no-couple-gate")
    if args.couple_gate:
        return True
    if args.no_couple_gate:
        return False
    return None


def apply_legacy_rookie_prior(*frames) -> None:
    """Rebuild the three draft-prior columns from the pre-refit share curves.

    In place, on frames that have already been built, so the two arms of the
    rookie-prior comparison differ only in the curve. Rebuilding the cache
    instead would give the baseline arm a different frames fingerprint and the
    comparison would stop being paired.
    """
    import numpy as np
    import pandas as pd

    from ffmodel.features.draft import LEGACY_SHARE_FIT_CURVES

    def claim(pick, position: str, stream: str) -> float:
        base, scale = LEGACY_SHARE_FIT_CURVES.get((position, stream), (0.0, 60.0))
        if base <= 0:
            return 0.0
        slot = 220.0 if pick is None or pd.isna(pick) else float(pick)
        return float(base * np.exp(-(slot - 1.0) / scale))

    for frame in frames:
        picks = frame["overall_pick"]
        for stream, column in (
            ("target", "draft_target_prior"),
            ("carry", "draft_carry_prior"),
            ("pass", "draft_pass_prior"),
        ):
            frame[column] = [
                claim(pick, position, stream)
                for pick, position in zip(picks, frame["position"])
            ]
