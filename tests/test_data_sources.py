"""Credentialed/live provider contracts with network responses mocked."""

import pytest
from datetime import datetime, timezone

from ffmodel.data import cfbd, odds, sleeper, weather


def test_sleeper_player_snapshot_is_flat_and_timestamped(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sleeper,
        "get_json",
        lambda *a, **k: {
            "123": {
                "player_id": "123",
                "full_name": "Example Runner",
                "gsis_id": "00-0030000",
                "position": "RB",
                "team": "BUF",
                "injury_status": "Questionable",
                "fantasy_positions": ["RB"],
            }
        },
    )
    out = sleeper.load_players(
        snapshot_at=datetime.now(timezone.utc).date(), refresh=True, cache_dir=tmp_path
    )
    assert out.loc[0, "player_name"] == "Example Runner"
    assert out.loc[0, "sleeper_player_id"] == "123"
    assert out.loc[0, "observed_at"]
    assert isinstance(out.loc[0, "fantasy_positions"], str)


def test_cfbd_requires_key_on_cache_miss(monkeypatch, tmp_path):
    monkeypatch.delenv("FFMODEL_CFBD_API_KEY", raising=False)
    monkeypatch.setattr(cfbd, "project_env_value", lambda name: None)
    with pytest.raises(cfbd.CfbdConfigurationError):
        cfbd.load_player_season_stats(2024, refresh=True, cache_dir=tmp_path)


def test_cfbd_key_is_header_not_cache_parameter(monkeypatch, tmp_path):
    captured = {}

    def fake_get(url, *, params, headers):
        captured.update({"url": url, "params": params, "headers": headers})
        return [{"playerId": 7, "player": "Example Receiver", "stat": 1000}]

    monkeypatch.setattr(cfbd, "get_json", fake_get)
    out = cfbd.load_player_season_stats(
        2024, api_key="secret-key", refresh=True, cache_dir=tmp_path
    )
    assert len(out) == 1
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert "secret-key" not in str(list(tmp_path.rglob("*")))


def test_cfbd_cache_hit_spends_no_additional_call(monkeypatch, tmp_path):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        return [{"playerId": 7, "player": "Example Receiver"}]

    monkeypatch.setattr(cfbd, "get_json", fake_get)
    first = cfbd.load_player_season_stats(
        2024, api_key="secret-key", cache_dir=tmp_path
    )
    second = cfbd.load_player_season_stats(
        2024, api_key="secret-key", cache_dir=tmp_path
    )
    assert len(first) == len(second) == 1
    assert len(calls) == 1
    assert cfbd.local_request_budget(tmp_path)["local_used"] == 1


def test_cfbd_local_monthly_cap_blocks_request(monkeypatch, tmp_path):
    monkeypatch.setenv("FFMODEL_CFBD_MONTHLY_LIMIT", "1")
    monkeypatch.setattr(cfbd, "get_json", lambda *args, **kwargs: [])
    cfbd.load_player_season_stats(
        2023, api_key="secret-key", refresh=True, cache_dir=tmp_path
    )
    with pytest.raises(cfbd.CfbdQuotaError, match="safety limit reached"):
        cfbd.load_player_season_stats(
            2024, api_key="secret-key", refresh=True, cache_dir=tmp_path
        )


def test_odds_are_normalized_to_book_market_outcome(monkeypatch, tmp_path):
    payload = [
        {
            "id": "game-1",
            "sport_key": "americanfootball_nfl",
            "commence_time": "2025-09-07T17:00:00Z",
            "home_team": "Buffalo Bills",
            "away_team": "Baltimore Ravens",
            "bookmakers": [
                {
                    "key": "book",
                    "title": "Book",
                    "markets": [
                        {
                            "key": "spreads",
                            "last_update": "2025-09-03T12:00:00Z",
                            "outcomes": [
                                {"name": "Buffalo Bills", "price": -110, "point": -1.5},
                                {"name": "Baltimore Ravens", "price": -110, "point": 1.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    monkeypatch.setattr(odds, "get_json", lambda *a, **k: payload)
    out = odds.load_nfl_odds(
        api_key="secret-key",
        refresh=True,
        cache_dir=tmp_path,
    )
    assert len(out) == 2
    assert set(out["point"]) == {-1.5, 1.5}
    assert set(out["market"]) == {"spreads"}


def test_weather_hourly_response(monkeypatch, tmp_path):
    monkeypatch.setattr(
        weather,
        "get_json",
        lambda *a, **k: {
            "hourly": {
                "time": ["2025-09-07T16:00", "2025-09-07T17:00"],
                "temperature_2m": [20.0, 19.0],
                "wind_speed_10m": [5.0, 7.0],
            }
        },
    )
    out = weather.load_hourly_forecast(
        42.77,
        -78.79,
        "2025-09-07",
        "2025-09-07",
        refresh=True,
        cache_dir=tmp_path,
    )
    assert len(out) == 2
    assert out["temperature_2m"].tolist() == [20.0, 19.0]
    assert out["observed_at"].notna().all()


def test_weather_previous_run_has_honest_availability(monkeypatch, tmp_path):
    captured = {}

    def fake_get(url, *, params):
        captured["url"] = url
        captured["params"] = params
        return {
            "hourly": {
                "time": ["2025-09-07T17:00"],
                "temperature_2m_previous_day4": [18.0],
            }
        }

    monkeypatch.setattr(weather, "get_json", fake_get)
    out = weather.load_previous_run_forecast(
        42.77,
        -78.79,
        "2025-09-07",
        "2025-09-07",
        lead_days=4,
        variables=("temperature_2m",),
        refresh=True,
        cache_dir=tmp_path,
    )
    assert captured["url"] == weather.PREVIOUS_RUNS_URL
    assert captured["params"]["hourly"] == "temperature_2m_previous_day4"
    assert out.loc[0, "temperature_2m"] == 18.0
    assert str(out.loc[0, "available_at"]).startswith("2025-09-03 17:00")


def test_live_sources_cannot_be_backdated(monkeypatch, tmp_path):
    monkeypatch.setattr(sleeper, "get_json", lambda *a, **k: {})
    with pytest.raises(ValueError, match="cannot retrieve historical"):
        sleeper.load_players(
            snapshot_at="2000-01-01", refresh=True, cache_dir=tmp_path
        )
    with pytest.raises(ValueError, match="cannot be backdated"):
        odds.load_nfl_odds(
            snapshot_at="2000-01-01T00:00:00Z",
            api_key="secret-key",
            refresh=True,
            cache_dir=tmp_path,
        )
