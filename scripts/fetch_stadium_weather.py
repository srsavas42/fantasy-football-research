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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from ffmodel.config import MANUAL_DATA_DIR, HTTP_TIMEOUT
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


# Transient enough to be worth another attempt: the free tier's rate limiter
# and the usual gateway failures. A 400 means the request itself is wrong and
# retrying it only spends quota.
RETRYABLE = ("429", "500", "502", "503", "504")

# A hard ceiling on any single call. `urlopen`'s own `timeout` is a *per socket
# operation* deadline, not a total one, so a response that trickles a few bytes
# at a time never trips it and the process hangs forever. The first run of this
# script did exactly that: it sat on one request for 35 minutes and had to be
# cancelled. Every call is therefore run in a worker thread and abandoned by the
# clock, which is the only bound that actually holds.
CALL_DEADLINE = 90.0


def _request(
    url: str, params: dict, *, retries: int = 3, delay: float, deadline: float = CALL_DEADLINE
) -> dict:
    """One call, with backoff and a wall-clock deadline that actually bounds it.

    ``ffmodel.data.http`` is standard-library only by design, so this script
    adds no dependency the rest of the package does not already have -- but its
    ``urlopen`` timeout cannot bound a slow-trickling response, so the deadline
    is imposed here instead.
    """
    last: Exception | None = None
    for attempt in range(retries):
        # A fresh executor per attempt: a hung worker cannot be killed, so it is
        # abandoned rather than reused, and `shutdown(wait=False)` lets the main
        # loop move on while it finishes or dies with the process.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                get_json, url, params=params, timeout=min(HTTP_TIMEOUT, 30.0)
            )
            return future.result(timeout=deadline)
        except FutureTimeout:
            last = TimeoutError(f"no response within {deadline:.0f}s")
        except RemoteDataError as error:
            last = error
            if not any(code in str(error) for code in RETRYABLE):
                raise
        finally:
            executor.shutdown(wait=False)
        time.sleep(delay * (2**attempt) + 1.0)
    raise RuntimeError(f"{url} gave up after {retries} attempts: {last}")


def probe(latitude: float, longitude: float, day: str, leads: tuple[int, ...]) -> int:
    """One tiny request per source, printing exactly what came back.

    This is the step that should have run before any backfill. It answers the
    two things the docs cannot: which variables each endpoint actually serves,
    and how far back the previous-runs archive reaches. One day of one stadium
    is a few kilobytes, so a wrong guess costs seconds instead of half an hour.
    """
    sources: list[tuple[str, int | None]] = [("observed", None)]
    sources += [(f"lead_{lead}", lead) for lead in leads]
    failures = 0
    for label, lead in sources:
        print(f"\n--- {label} ({day}) ---", flush=True)
        started = time.time()
        try:
            frame = fetch_block(
                latitude, longitude, day, day, lead_days=lead, delay=0.5
            )
        except Exception as error:  # noqa: BLE001 - the point is to report it
            failures += 1
            print(f"  FAILED after {time.time() - started:.1f}s: {error}", flush=True)
            continue
        took = time.time() - started
        if frame.empty:
            failures += 1
            print(f"  empty response after {took:.1f}s", flush=True)
            continue
        served = [c for c in frame.columns if c != "time"]
        missing = [v for v in VARIABLES if v not in served]
        print(f"  {len(frame)} hourly rows in {took:.1f}s", flush=True)
        print(f"  served:  {served}", flush=True)
        print(f"  MISSING: {missing or 'none'}", flush=True)
        row = frame.iloc[len(frame) // 2]
        print(f"  midday sample: {json.dumps({k: str(row[k]) for k in served})}", flush=True)
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


# One call per stadium-season asked for five months of hourly rows at once. Even
# when that succeeds it is a large response to hold open, and a single stall
# takes the whole stadium-season with it. Monthly chunks are more requests but
# each is small enough to fail fast and retry cheaply.
CHUNK_DAYS = 31


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
            site["latitude"], site["longitude"], args.probe, tuple(args.leads)
        )
        print(f"\n{failures} of {1 + len(args.leads)} sources failed")
        return 1 if failures == 1 + len(args.leads) else 0

    seasons = range(args.seasons[0], args.seasons[1] + 1)
    games = load_games(seasons)
    if args.stadiums:
        games = games[games["stadium_id"].isin(args.stadiums)]
    print(f"{len(games)} games, {games.stadium_id.nunique()} stadiums, seasons {seasons}")

    frame = collect(games, coordinates, leads=tuple(args.leads), delay=args.delay)

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
