"""Print a short summary of a Wikipedia coaching scrape for a CI job summary.

Kept as its own script rather than an inline heredoc in the workflow YAML: a
multi-line Python heredoc inside a ``run: |`` block scalar is fragile — its
body must stay indented at least as much as the block scalar itself, which a
plain ``<<'PY'`` heredoc does not do, and it silently breaks YAML parsing
rather than failing the step.

    python scripts/summarize_wikipedia_scrape.py data/coaching/wikipedia
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    manifest_path = args.output_dir / "run_manifest.json"
    if not manifest_path.exists():
        print("- no run_manifest.json was produced")
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seasons = manifest.get("seasons") or []
    if seasons:
        print(f"- seasons: {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons)")
    else:
        print("- seasons: none")
    if manifest.get("team_filter"):
        print(f"- team filter: {', '.join(manifest['team_filter'])}")
    print("- rows:")
    for name, count in manifest.get("rows", {}).items():
        print(f"  - {name}: {count:,}")

    review_path = args.output_dir / "review_queue.csv"
    if review_path.exists():
        with review_path.open(encoding="utf-8") as handle:
            rows = sum(1 for _ in handle) - 1
        print(f"- rows needing manual review: {max(rows, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
