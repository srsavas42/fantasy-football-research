"""Pull Open-Meteo conditions for every NFL game, at several forecast lead times.

The research sandbox's egress policy denies Open-Meteo, so this runs on a
GitHub-hosted runner and commits the result back — the same arrangement
`scrape-wikipedia-coaches.yml` already uses for Wikipedia. After a run, any
environment that checks out the branch has the weather without needing to reach
the API itself.

**Why lead time is the point of this script.** Every weather number this package
has measured so far used the conditions nflverse recorded *at* the game, which
is a ceiling: what perfect foreknowledge would have been worth. A live
projection has a forecast, and which forecast depends on when the decision is
made. Waiver claims are due Wednesday; a start/sit call can wait until Sunday
morning. Measured against the 2023-2025 schedule, a Wednesday 16:00 cutoff sits
1.18 days before a Thursday game, 3.88 before a Sunday one and 5.18 before a
Monday one — so a single fixed lead is wrong for about a fifth of the slate, and
the join has to be per game.

Three sources are pulled so the skill curve can be measured rather than assumed:

``observed``
    Open-Meteo's historical-forecast archive at (near) valid time. Not a
    forecast. It is the ceiling, and the control that says whether a coordinate
    is even right.

``lead_N``
    The previous-runs archive at a fixed offset of N days. ``lead_4`` is the
    Wednesday decision for a Sunday game; ``lead_1`` is roughly the day before.
    Open-Meteo archives offsets 1-7, and that archive does **not** reach as far
    back as the historical one, so the script reports the coverage it actually
    got rather than assuming a span.

**It validates itself.** Hand-entered coordinates are the obvious failure mode
here and a wrong one is silent: it returns perfectly plausible weather for the
wrong city. So every fetched ``observed`` row is compared against the
temperature and wind nflverse recorded for the same game, per stadium. A
stadium whose correlation collapses is a coordinate to go and check, and the
run prints a table saying so.

Units are requested in Fahrenheit and mph to match the nflverse columns, so the
comparison is direct rather than a conversion away from being direct.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.config import MANUAL_DATA_DIR
from ffmodel.data import ingest
from ffmodel.data.http import RemoteDataError, get_json

HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
COORDINATES_PATH = MANUAL_DATA_DIR / "stadium_coordinates.csv"
DEFAULT_OUTPUT = Path("data/weather/stadium_hours.parquet")

# What to ask for. `precipitation` is water-equivalent and so does not by itself
# separate rain from snow -- an inch of snow and a light shower can carry the
# same number -- which is why `snowfall` and `weather_code` are requested
# alongside it rather than inferred from temperature.
VARIABLES = (
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "cloud_cover",
    "relative_humidity_2m",
)

# nflverse quotes kickoff in US Eastern; Open-Meteo is asked for UTC so one
# convention holds across the international games too.
GAME_TIMEZONE = "America/New_York"


def load_coordinates() -> pd.DataFrame:
    frame = pd.read_csv(COORDINATES_PATH)
    if frame["stadium_id"].duplicated().any():
        raise SystemExit("stadium_coordinates.csv has duplicate stadium_id rows")
    return frame


def load_games(seasons: range) -> pd.DataFrame:
    schedule = ingest.load_schedules(list(seasons))
    schedule = schedule[schedule["game_type"] == "REG"].copy()
    kickoff = pd.to_datetime(
        schedule["gameday"].astype(str) + " " + schedule["gametime"].astype(str),
        errors="coerce",
    )
    schedule = schedule[kickoff.notna()].copy()
    kickoff = kickoff[kickoff.notna()]
    local = kickoff.dt.tz_localize(GAME_TIMEZONE, ambiguous=True, nonexistent="shift_forward")
    schedule["kickoff_utc"] = local.dt.tz_convert("UTC")
    # The hour Open-Meteo will have a row for.
    schedule["kickoff_hour"] = schedule["kickoff_utc"].dt.floor("h")
    keep = [
        "game_id",
        "season",
        "week",
        "stadium_id",
        "stadium",
        "roof",
        "home_team",
        "away_team",
        "kickoff_utc",
        "kickoff_hour",
    ]
    for column in ("temp", "wind"):
        if column in schedule.columns:
            keep.append(column)
    return schedule[keep].reset_index(drop=True)


# A blocklist, not an allowlist. The first full backfill lost 90 minutes of
# progress to `_ssl.c:999: The handshake operation timed out` -- a plain
# connection-level failure that carries no HTTP status code at all, and every
# real run of this script hit it roughly once per stadium-season. An earlier
# version of this constant listed retryable *status codes* and refused to
# retry anything that did not carry one of them, which is backwards: a
# connection that never completed its handshake has no status to check, so it
# failed the allowlist test and was treated as fatal. The failures worth NOT
# retrying are the ones retrying cannot fix -- the request itself was wrong --
# and those are the only ones this excludes.
NOT_RETRYABLE = ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404")

# Real stalls in practice resolved in a handful of seconds once retried, so a
# short flat gap beats runaway exponential backoff: the cost of a stall is the
# per-call timeout below, paid up to `retries` times, and there is no evidence
# a longer wait between attempts helps.
RETRY_GAP = 2.0

# `urlopen`'s own `timeout` is a *per socket operation* deadline, not a total
# one, so a response that trickles a few bytes at a time can outlast it. This
# is the wall-clock ceiling on the socket wait itself, kept short because the
# observed failure mode (a handshake that never completes) does not become
# more likely to succeed by waiting longer for it.
SOCKET_TIMEOUT = 20.0


def _request(url: str, params: dict, *, retries: int = 3, delay: float) -> dict:
    """One call, with a short retry loop for connection-level failures.

    ``ffmodel.data.http`` is standard-library only by design, so this script
    adds no dependency the rest of the package does not already have.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return get_json(url, params=params, timeout=SOCKET_TIMEOUT)
        except RemoteDataError as error:
            last = error
            if any(code in str(error) for code in NOT_RETRYABLE):
                raise
        time.sleep(RETRY_GAP + delay)
    raise RuntimeError(f"{url} gave up after {retries} attempts: {last}")


PROBE_OUTPUT = Path("data/weather/probe.md")


def probe(
    latitude: float,
    longitude: float,
    day: str,
    leads: tuple[int, ...],
    *,
    label: str = "",
    output: Path = PROBE_OUTPUT,
) -> int:
    """One tiny request per source, printing exactly what came back.

    This is the step that should have run before any backfill. It answers the
    two things the docs cannot: which variables each endpoint actually serves,
    and how far back the previous-runs archive reaches. One day of one stadium
    is a few kilobytes, so a wrong guess costs seconds instead of half an hour.
    """
    sources: list[tuple[str, int | None]] = [("observed", None)]
    sources += [(f"lead_{lead}", lead) for lead in leads]
    failures = 0
    # Written to a file as well as stdout: a workflow log is not reachable from
    # every environment that needs this answer, and the committed file is a
    # durable record of what the API actually served on a given date.
    lines = [
        f"# Open-Meteo probe: {label or 'stadium'} on {day}",
        "",
        f"Coordinates {latitude}, {longitude}. Requested {len(VARIABLES)} variables.",
        "",
    ]
    for name, lead in sources:
        header = f"## {name} ({day})"
        print(f"\n--- {name} ({day}) ---", flush=True)
        lines += ["", header, ""]
        started = time.time()
        try:
            frame = fetch_block(
                latitude, longitude, day, day, lead_days=lead, delay=0.5
            )
        except Exception as error:  # noqa: BLE001 - the point is to report it
            failures += 1
            took = time.time() - started
            print(f"  FAILED after {took:.1f}s: {error}", flush=True)
            lines.append(f"**FAILED** after {took:.1f}s: `{error}`")
            continue
        took = time.time() - started
        if frame.empty:
            failures += 1
            print(f"  empty response after {took:.1f}s", flush=True)
            lines.append(f"**Empty response** after {took:.1f}s.")
            continue
        served = [c for c in frame.columns if c != "time"]
        missing = [v for v in VARIABLES if v not in served]
        row = frame.iloc[len(frame) // 2]
        sample = {k: str(row[k]) for k in served}
        print(f"  {len(frame)} hourly rows in {took:.1f}s", flush=True)
        print(f"  served:  {served}", flush=True)
        print(f"  MISSING: {missing or 'none'}", flush=True)
        print(f"  midday sample: {json.dumps(sample)}", flush=True)
        lines += [
            f"{len(frame)} hourly rows in {took:.1f}s.",
            "",
            f"- **served** ({len(served)}): {', '.join(f'`{c}`' for c in served)}",
            f"- **missing**: {', '.join(f'`{c}`' for c in missing) if missing else 'none'}",
            "",
            "| variable | midday value |",
            "|---|---|",
        ]
        lines += [f"| `{k}` | {v} |" for k, v in sample.items()]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output}", flush=True)
    return failures


def _hourly(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    # The previous-runs endpoint suffixes every variable with the offset it
    # came from; strip it so all sources share one schema.
    frame.columns = [c.split("_previous_day")[0] for c in frame.columns]
    return frame


def fetch_block(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    *,
    lead_days: int | None,
    delay: float,
) -> pd.DataFrame:
    """Hourly rows for one stadium over one date range, at one lead time."""
    variables = VARIABLES
    url = HISTORICAL_URL
    if lead_days is not None:
        variables = tuple(f"{name}_previous_day{lead_days}" for name in VARIABLES)
        url = PREVIOUS_RUNS_URL
    params = {
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }
    return _hourly(_request(url, params, delay=delay))


# A full regular season is under six months of hourly rows for eleven
# variables -- a few hundred KB -- and the unchunked version of this script
# fetched exactly that per stadium-season in a few seconds each, repeatedly,
# with no problem. Monthly chunking was added on the theory that a large
# response was the risk; it was not. The actual failure mode measured in the
# first full backfill was a TLS handshake timeout roughly once per
# stadium-season, unrelated to response size, and chunking to 31 days
# quadrupled the request count and so quadrupled the exposure to it. Left wide
# enough that no in-season range needs a second chunk.
CHUNK_DAYS = 200


def _chunks(start: str, end: str, days: int = CHUNK_DAYS):
    """Split an inclusive date range into spans of at most ``days``."""
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    while first <= last:
        stop = min(first + pd.Timedelta(days=days - 1), last)
        yield first.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")
        first = stop + pd.Timedelta(days=1)


def collect(
    games: pd.DataFrame,
    coordinates: pd.DataFrame,
    *,
    leads: tuple[int, ...],
    delay: float,
    checkpoint: Path | None = None,
) -> pd.DataFrame:
    known = set(coordinates["stadium_id"])
    missing = sorted(set(games["stadium_id"]) - known)
    if missing:
        raise SystemExit(
            f"no coordinates for {len(missing)} stadium_id(s): {missing}. "
            f"Add them to {COORDINATES_PATH}."
        )
    lookup = coordinates.set_index("stadium_id")

    sources: list[tuple[str, int | None]] = [("observed", None)]
    sources += [(f"lead_{lead}", lead) for lead in leads]

    rows = []
    groups = list(games.groupby(["stadium_id", "season"], sort=True))
    for index, ((stadium_id, season), block) in enumerate(groups, start=1):
        site = lookup.loc[stadium_id]
        # One call per stadium-season rather than per game: the range is a few
        # thousand hourly rows either way and this is 1/17th the requests.
        start = block["kickoff_hour"].min().strftime("%Y-%m-%d")
        end = block["kickoff_hour"].max().strftime("%Y-%m-%d")
        print(
            f"[{index}/{len(groups)}] {stadium_id} {season} "
            f"({len(block)} games, {start}..{end})",
            flush=True,
        )
        for label, lead in sources:
            # Printed *before* the call, not after: a heartbeat that only
            # appears on success cannot tell a slow call from a hung one, which
            # is precisely the confusion the first run of this script caused.
            print(f"    {label}: requesting...", end=" ", flush=True)
            began = time.time()
            try:
                hourly = pd.concat(
                    [
                        fetch_block(
                            site["latitude"],
                            site["longitude"],
                            chunk_start,
                            chunk_end,
                            lead_days=lead,
                            delay=delay,
                        )
                        for chunk_start, chunk_end in _chunks(start, end)
                    ],
                    ignore_index=True,
                )
            except Exception as error:  # noqa: BLE001 - reported, not swallowed
                print(f"FAILED after {time.time() - began:.1f}s: {error}", flush=True)
                continue
            if hourly.empty:
                print(f"no rows ({time.time() - began:.1f}s)", flush=True)
                continue
            merged = block.merge(
                hourly, left_on="kickoff_hour", right_on="time", how="left"
            )
            merged["source"] = label
            merged["lead_days"] = lead if lead is not None else 0
            rows.append(merged.drop(columns=["time"], errors="ignore"))
            got = merged["temperature_2m"].notna().mean() if len(merged) else 0.0
            print(
                f"{got:.0%} of kickoff hours matched ({time.time() - began:.1f}s)",
                flush=True,
            )
            time.sleep(delay)
        # Checkpointed after every stadium-season rather than once at the end.
        # The first full backfill ran 90 minutes, got 65% through, hit a step
        # timeout, and kept nothing: the only output path was one write after
        # every group had been collected, so a timeout anywhere discarded
        # everything before it. A parquet write here is a few hundred KB and
        # costs a fraction of a second next to the seconds-to-tens-of-seconds
        # each request already takes.
        if checkpoint is not None and rows:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(rows, ignore_index=True).to_parquet(checkpoint, index=False)
    if not rows:
        raise SystemExit("no weather rows were retrieved at all")
    return pd.concat(rows, ignore_index=True)


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare `observed` against what nflverse recorded, per stadium.

    A wrong coordinate returns entirely plausible weather for the wrong place,
    so the only real check is against an independent record of the same game.
    """
    if "temp" not in frame.columns or "wind" not in frame.columns:
        return pd.DataFrame()
    block = frame[(frame["source"] == "observed") & frame["roof"].isin(["outdoors", "open"])]
    block = block.dropna(subset=["temp", "temperature_2m"])
    if block.empty:
        return pd.DataFrame()
    rows = []
    for stadium_id, part in block.groupby("stadium_id"):
        if len(part) < 10:
            continue
        temp_r = part["temp"].corr(part["temperature_2m"])
        wind_part = part.dropna(subset=["wind", "wind_speed_10m"])
        wind_r = (
            wind_part["wind"].corr(wind_part["wind_speed_10m"])
            if len(wind_part) >= 10
            else float("nan")
        )
        rows.append(
            {
                "stadium_id": stadium_id,
                "stadium": part["stadium"].iloc[0],
                "n": len(part),
                "temp_r": round(float(temp_r), 3),
                "temp_bias": round(float((part["temperature_2m"] - part["temp"]).mean()), 2),
                "wind_r": round(float(wind_r), 3),
            }
        )
    report = pd.DataFrame(rows).sort_values("temp_r")
    report["suspect"] = report["temp_r"] < 0.80
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs=2, default=[2016, 2025])
    parser.add_argument(
        "--leads",
        type=int,
        nargs="*",
        default=[1, 4],
        help="Forecast lead times in days (Open-Meteo archives 1-7). "
        "4 is the Wednesday waiver deadline for a Sunday game; 1 is the day before.",
    )
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stadiums", nargs="*", default=None, help="Limit to these stadium_ids."
    )
    parser.add_argument(
        "--probe",
        metavar="YYYY-MM-DD",
        default=None,
        help="Make one tiny request per source for this date and print what came "
        "back, then exit. Run this before any backfill: it is what says which "
        "variables each endpoint serves and how far back the previous-runs "
        "archive reaches.",
    )
    parser.add_argument(
        "--probe-stadium",
        default="BUF00",
        help="Stadium to probe with (default BUF00, an outdoor cold-weather site).",
    )
    args = parser.parse_args(argv)

    for lead in args.leads:
        if lead not in range(1, 8):
            raise SystemExit(f"lead {lead} is outside Open-Meteo's archived 1-7 days")

    coordinates = load_coordinates()

    if args.probe:
        site = coordinates.set_index("stadium_id").loc[args.probe_stadium]
        print(
            f"probing {args.probe_stadium} ({site['stadium']}) at "
            f"{site['latitude']}, {site['longitude']} on {args.probe}"
        )
        failures = probe(
            site["latitude"],
            site["longitude"],
            args.probe,
            tuple(args.leads),
            label=f"{args.probe_stadium} ({site['stadium']})",
        )
        print(f"\n{failures} of {1 + len(args.leads)} sources failed")
        return 1 if failures == 1 + len(args.leads) else 0

    seasons = range(args.seasons[0], args.seasons[1] + 1)
    games = load_games(seasons)
    if args.stadiums:
        games = games[games["stadium_id"].isin(args.stadiums)]
    print(f"{len(games)} games, {games.stadium_id.nunique()} stadiums, seasons {seasons}")

    frame = collect(
        games, coordinates, leads=tuple(args.leads), delay=args.delay, checkpoint=args.output
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"\nwrote {args.output}: {frame.shape[0]} rows, {frame.shape[1]} columns")

    print("\n=== coverage by source ===")
    coverage = (
        frame.assign(has=frame["temperature_2m"].notna())
        .groupby(["source", "season"])["has"]
        .mean()
        .unstack(0)
    )
    print((coverage * 100).round(1).to_string())

    report = validate(frame)
    if not report.empty:
        print("\n=== coordinate check: fetched vs nflverse-recorded, outdoor games ===")
        print(report.to_string(index=False))
        bad = report[report["suspect"]]
        if not bad.empty:
            print(
                f"\n!! {len(bad)} stadium(s) correlate below 0.80 on temperature. "
                "Check their coordinates before using this data."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
