"""Copy a walk-forward cache and add one derived feature to the player rows.

A new feature has to exist in the frames before a model can be scored on it,
and the obvious way to get it there -- rebuild the cache from the source -- is
the wrong way for an ablation. A fresh pull differs from the old one in more
than the new column: sixty-nine of two hundred eighty-nine columns differed
between two caches on this branch with identical row counts, because upstream
data is revised. An arm scored on a rebuilt cache against a baseline scored on
the old one is not measuring the feature.

Copying the existing frames and adding the column keeps every other input
byte-identical, so the only difference between the two arms is the thing being
tested. Both arms must then run against this new directory: the baseline too,
even though it never reads the column. The frame fingerprint changes when the
schema does, and the acceptance gate refuses a comparison across fingerprints,
which is the check that catches getting this wrong.

The feature builders here are pure functions of columns the cache already has,
so this needs no network and takes seconds.
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.features.market import ADP_FEATURES, add_market_adp_features
from ffmodel.features.season_efficiency import add_teammate_quality_features
from ffmodel.features.snaps import SNAP_EXPOSURE_FEATURES, add_snap_exposure

BUILDERS = {
    "market-adp": (add_market_adp_features, ADP_FEATURES),
    "teammate-quality": (add_teammate_quality_features, ("teammate_qb_quality_signal",)),
    # Unlike the others this one is not a pure function of columns the cache
    # already has -- it reads the weekly snap feed, so it needs the network.
    # It still belongs here rather than in a cache rebuild, for the reason at
    # the top of this file: the point is to change one column and nothing else.
    "snap-exposure": (add_snap_exposure, SNAP_EXPOSURE_FEATURES),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/ffmodel-wf-2025"))
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--feature", choices=sorted(BUILDERS), required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing destination instead of refusing",
    )
    args = parser.parse_args(argv)

    build, expected = BUILDERS[args.feature]
    player_path = args.source / "player_rows.pkl"
    team_path = args.source / "team_rows.pkl"
    for path in (player_path, team_path):
        if not path.exists():
            raise SystemExit(f"no cache at {path}")
    if args.dest.exists() and not args.force:
        raise SystemExit(
            f"{args.dest} already exists. Re-running would silently replace the "
            "frames some completed arm was scored on; pass --force if that is "
            "what you mean"
        )

    player_rows = pd.read_pickle(player_path)
    before = set(player_rows.columns)
    augmented = build(player_rows)

    missing = [name for name in expected if name not in augmented.columns]
    if missing:
        raise SystemExit(
            f"{args.feature} did not produce {missing}. Writing the cache anyway "
            "would hand the model a frame that looks augmented and is not"
        )
    if len(augmented) != len(player_rows):
        raise SystemExit(
            f"row count changed {len(player_rows)} -> {len(augmented)}; the "
            "feature join duplicated or dropped rows"
        )

    args.dest.mkdir(parents=True, exist_ok=True)
    augmented.to_pickle(args.dest / "player_rows.pkl")
    shutil.copyfile(team_path, args.dest / "team_rows.pkl")

    added = sorted(set(augmented.columns) - before)
    print(f"{args.source} -> {args.dest}")
    print(f"  {len(augmented)} player rows, {len(added)} column(s) added: {added}")
    for name in expected:
        values = pd.to_numeric(augmented[name], errors="coerce")
        print(
            f"  {name:26s} missing {values.isna().mean():>6.1%}  "
            f"mean {values.mean():>8.3f}  sd {values.std():>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
