# Data-source setup

The acquisition command is `ffmodel-data` (or `python -m ffmodel.data.pull`).
Every remote artifact is written beneath `.cache/ffmodel/raw/` as parquet with
a neighboring `manifest.json`. Mutable feeds include an `as_of` partition; API
keys are never placed in cache paths or manifests.

## 1. Install

Python 3.11 or 3.12 is recommended. With `uv` on Windows PowerShell:

```powershell
uv python install 3.12
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
ffmodel-data doctor
```

Equivalent pip installation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ffmodel-data doctor
```

Override the cache location when desired:

```powershell
$env:FFMODEL_CACHE_DIR = "D:\ffmodel-cache"
```

## 2. nflverse: no account or key

This is the primary NFL feed. `nflreadpy` replaces the archived
`nfl_data_py` dependency and converts to pandas at the package boundary.

Pull the recommended core for several seasons:

```powershell
ffmodel-data bootstrap --seasons 2022 2023 2024 2025
```

Bootstrap downloads the current draft rankings. The full historical rankings
archive is much larger and should be requested deliberately:

```powershell
ffmodel-data nflverse --seasons 2025 --datasets rankings --ranking-type all
```

Add play-by-play explicitly because it is substantially larger:

```powershell
ffmodel-data nflverse --seasons 2022 2023 2024 2025 --datasets pbp
```

Advanced optional pulls:

```powershell
ffmodel-data nflverse --seasons 2018 2019 2020 2021 2022 2023 2024 2025 `
  --datasets ngs_receiving ngs_rushing pfr_rec pfr_rush participation ftn_charting
```

Coverage filters are applied by the CLI. In particular, nflverse's historical
injury feed currently ends after 2024. Participation for recent seasons may
not be available until after the season.

## 3. Sleeper: no account or key

Sleeper supplies current roster status, injury status, practice participation,
and depth order. It does not supply historical snapshots. Run this once per day
during preseason and the regular season; the daily cache becomes your history.

```powershell
ffmodel-data sleeper
```

For unattended Windows collection, create a Task Scheduler task that activates
the repository virtual environment and runs `ffmodel-data sleeper` once daily.
Do not request the full players endpoint repeatedly; the loader intentionally
caches it by UTC date.

## 4. CollegeFootballData: locally cached and quota guarded

Create a free account and key at <https://collegefootballdata.com/key>. Put it
in the Git-ignored project `.env` file; process environment variables take
precedence when both are present:

```dotenv
FFMODEL_CFBD_API_KEY=your-key
FFMODEL_CFBD_MONTHLY_LIMIT=1000
```

Do not commit `.env` or pass a key on the command line. Check both the local
request ledger and CFBD's authoritative account usage with:

```powershell
ffmodel-data cfbd-quota
```

The provider `/info` check is not billed. Each data response is written as
Parquet with a checksum/provenance manifest under
`.cache/ffmodel/raw/cfbd/`. The exact endpoint/year/parameters form the cache
key, so a normal rerun performs zero billable requests. `--refresh` deliberately
replaces cached data and spends calls; avoid it for historical data unless a
correction is expected.

Pull the initial prospect corpus with one bulk request per endpoint/year:

```powershell
ffmodel-data cfbd --seasons 2018 2019 2020 2021 2022 2023 2024 2025 `
  --datasets stats usage roster recruits draft --max-new-requests 40
```

The CLI plans cache misses before pulling, defaults to a 100-new-request cap per
run, compares the plan with CFBD's remaining balance, and maintains a separate
1,000-call monthly local safety cap. The local ledger reserves a call before the
HTTP request, so failed requests are conservatively counted. The provider is
still authoritative if the same key is used outside this repository.

The raw tables intentionally remain provider-shaped. `features/prospects.py`
should later derive dominator, breakout age, and team-normalized market shares
from these cached inputs. College-to-NFL matches that cannot be resolved from
provider IDs belong in `data/manual/player_identity_overrides.csv`; do not put
unreviewed fuzzy matches there.

## 5. The Odds API: deferred

Vegas odds are currently out of scope. Leave `FFMODEL_ODDS_API_KEY` empty and do
not run `ffmodel-data odds`. The adapter remains available for a later phase but
is not included in `bootstrap` or any CFBD pull.

When this source is intentionally enabled later:

1. Obtain a key at <https://the-odds-api.com/>.
2. Set `FFMODEL_ODDS_API_KEY` in the environment as above.
3. Archive current odds:

```powershell
ffmodel-data odds
```

4. Historical snapshots require the provider's historical-data plan:

```powershell
ffmodel-data odds --historical-at 2024-09-04T16:00:00Z
```

Use a consistent cutoff in backtests. A Wednesday projection must not join a
Sunday closing line. The loader emits a long table by event, book, market, and
outcome, with `observed_at` retained.

## 6. Open-Meteo: no key for standard non-commercial use

The loader accepts stadium coordinates and returns hourly values. For a live
forecast, run it at the actual projection cutoff; the acquisition timestamp is
generated by the loader and cannot be backdated:

```powershell
ffmodel-data weather --latitude 42.7738 --longitude -78.7868 `
  --start-date 2026-09-13 --end-date 2026-09-13
```

For a walk-forward backtest, use the Previous Runs endpoint at a fixed lead
time. This example retrieves values predicted four days before valid time:

```powershell
ffmodel-data weather --latitude 42.7738 --longitude -78.7868 `
  --start-date 2025-09-07 --end-date 2025-09-07 --lead-days 4
```

The output includes `available_at = forecast_time - lead_days`. `--historical`
uses Open-Meteo's stitched near-valid-time forecast and is appropriate as a
weather baseline, not as a four-day-ahead predictor. Review Open-Meteo's
commercial-use terms before commercial use.

## 7. Coaching lineage: Wikipedia discovery plus manual verification

The Wikipedia pipeline discovers every team-season represented by the committed
yearly data (1970-2021 by default), retrieves its HC and OC, then retrieves each
coach's structured career history. It uses the MediaWiki API rather than HTML,
keeps an exact revision id/timestamp/source URL for attribution, respects
`maxlag`, sends the project user agent, and caches each revision so an interrupted
run can resume without repeating completed requests.

Install the parsing dependency and run the full committed-data pull:

```powershell
uv sync --extra dev --extra scrape
uv run ffmodel-coaches
```

Useful focused runs are:

```powershell
# Inclusive range; seasons after 2021 discover teams from nflverse schedules.
uv run ffmodel-coaches --seasons 2018:2025

# Review a small slice before the full run.
uv run ffmodel-coaches --seasons 2023 --teams BUF KC

# Rebuild only from already archived MediaWiki responses.
uv run ffmodel-coaches --offline

# Explicitly retrieve current revisions again.
uv run ffmodel-coaches --refresh
```

The default derived-data directory is `data/coaching/wikipedia/` and contains:

- `team_seasons`: the exact coverage discovered from the football data.
- `team_season_assignments`: long-form HC/OC records, including interim and
  midseason changes when Wikipedia documents a week boundary.
- `coach_history`: structured coaching stops for every discovered HC and OC.
- `scheme_sources`: the opening team-season scheme carrier selected by the rule
  "HC if his pre-season history includes OC; otherwise OC." An HC fallback is
  used only when no OC is documented and is always flagged.
- `scheme_lineage`: prior NFL team-seasons for the selected carrier, restricted
  to `prior_season < season`, with the prior team's HC attached when that
  team-season is in the scraped corpus.
- `review_queue.csv`: missing pages, missing OC roles, midseason changes, and
  any selection lacking enough structured evidence.
- `run_manifest.json`: row counts, generation time, rule, source, and license.

Load generated model inputs with:

```python
from ffmodel.data.coaching import (
    load_scheme_lineage,
    load_scheme_sources,
    load_team_season_assignments,
)

assignments = load_team_season_assignments()
scheme = load_scheme_sources()
lineage = load_scheme_lineage()
```

Two review rules matter. First, an OC title does not establish who called plays.
nflverse schedules supply game-level HCs, but there is no dependable open API
for historical play callers. Populate `data/manual/coach_team_period.csv` with
effective dates and a source URL for confirmed callers; those manual periods are
higher-authority overrides. Use `confirmed`, `high`, `medium`, or `low`
confidence. Midseason changes must be separate rows; never assign an interim
caller to a full season.

Second, a prior team's HC is a mentor/context proxy, not proof that the new coach
carried over that system. Use lineage through partial pooling and recency weights,
and require a walk-forward ablation against a team-only baseline before retaining
it. Keep the Wikipedia facts separate from this modeling assumption.

The manual file is validated by:

```python
from ffmodel.data.coaching import load_coaching_periods
coaches = load_coaching_periods()
```

No API key is required for Wikipedia. After a full run, inspect
`review_queue.csv`, research unresolved play-calling responsibility from
sourceable articles/team releases, and add only confirmed effective periods to
the manual file. Wikipedia-derived text is CC BY-SA; revision attribution is
preserved on the assignment, history, scheme-source, and lineage facts and in
the manifest, while raw wikitext remains under the Git-ignored `.cache/`
directory.

## 8. Paid route/charting data

PFF, FTN, and SIS require credentials and their schemas/licenses differ. No
scraper is included because automated redistribution may violate their terms.
If a license is purchased, add a provider adapter that returns:

```text
provider_player_id, game_id, season, week, routes_run, targets,
route_type, alignment, observed_at, source
```

Keep raw paid files outside Git, map IDs through `player_dim`, and add the
provider's license to cache manifests. First run a walk-forward ablation to
confirm true routes improve over the open snap-share plus PBP baseline.

## 9. BlueSky/RSS signal archive: operate as a separate live service

This is intentionally not part of `bootstrap`: it is a projection-time signal
log, not a historical training source. There is no complete retrospective
archive aligned to old NFL prediction cutoffs.

To build it when Phase 9 begins:

1. Create a reviewed list of beat-reporter BlueSky DIDs and RSS feed URLs.
2. Run a long-lived BlueSky Jetstream or AT Protocol firehose consumer on a
   machine that can remain online. Jetstream is simpler; the full Relay/Tap
   path is preferable if durable archival semantics matter.
3. Append, never overwrite, records containing `observed_at`, author DID,
   post URI/CID, text, source, and a content hash. Store raw records separately
   from extracted player/injury/role signals.
4. Add reconnect cursors, deduplication by URI/CID, exponential backoff, and a
   daily dead-letter/replay report before trusting it in live projections.
5. Record every downstream prior adjustment with its confidence, bounded
   effect, decay/expiry time, and the unadjusted model output.

The collector will require an always-on process and operational choices that
cannot be supplied by this repository alone: hosting location, curated account
list, retention policy, and—if LLM extraction is used—an API/provider key and
cost budget. Do not represent a newly started archive as historical evidence.

## 10. Optional supported commercial feed

If daily Sleeper snapshots are operationally insufficient, a supported vendor
such as SportsDataIO can replace the live injury/depth adapter. You must choose
and license a plan, then provide a sample response and API credential. Keep the
adapter behind the same `player_week_availability` contract and preserve the
provider's original update timestamp; do not mix vendor projections into the
observed-injury field.

## Point-in-time rules

- Never overwrite Sleeper, rankings, odds, or weather snapshots.
- Every live row needs `observed_at`; model joins require
  `observed_at <= prediction_time`.
- Keep current injury probability separate from observed historical activity.
- Closing odds and observed weather are future information for earlier cutoffs.
- Keep provider-shaped raw tables separate; only feature builders create joins.
