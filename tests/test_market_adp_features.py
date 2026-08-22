"""The ADP join is a name match against an outside file, so it gets tests.

Every failure mode here is silent by nature: a bad join produces a number
rather than an exception, and the number lands in a coefficient. These check the
three that would actually change a projection -- a rank attached to the wrong
player, an unranked player presented as an average one, and a season whose file
is missing being read as a season the market liked nobody in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ffmodel.features.market import (
    ADP_DEPTH,
    ADP_FEATURES,
    add_market_adp_features,
    load_adp,
    read_adp_file,
)


def write_adp(directory: Path, season: int, rows: list[tuple[int, str, str]]) -> None:
    frame = pd.DataFrame(
        [{"Rank": rank, "Player (Bye)": player, "POS": pos} for rank, player, pos in rows]
    )
    frame.to_csv(
        directory / f"FantasyPros_{season}_Overall_ADP_Rankings.csv", index=False
    )


@pytest.fixture
def adp_dir(tmp_path: Path) -> Path:
    write_adp(
        tmp_path,
        2021,
        [
            (1, "Christian McCaffrey  CAR (13)", "RB1"),
            (2, "Derrick Henry", "RB2"),
            (3, "Davante Adams  GB (10)", "WR1"),
            (4, "Travis Kelce  KC (12)", "TE1"),
            (5, "Patrick Mahomes  KC (12)", "QB1"),
            (6, "Justin Tucker  BAL (8)", "K1"),
            (ADP_DEPTH + 5, "Deep Sleeper", "WR90"),
        ],
    )
    return tmp_path


def rows(*players: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2021, "player_name": name, "position": position}
            for name, position in players
        ]
    )


def test_parses_both_player_column_shapes(adp_dir: Path) -> None:
    """The team suffix is on 10% of 2015 rows and 84% of 2026 rows."""
    parsed = read_adp_file(adp_dir / "FantasyPros_2021_Overall_ADP_Rankings.csv")
    keys = set(parsed.key)
    assert "christianmccaffrey" in keys  # "Name TEAM (BYE)"
    assert "derrickhenry" in keys  # bare name


def test_drops_kickers_and_players_past_the_depth_cap(adp_dir: Path) -> None:
    parsed = read_adp_file(adp_dir / "FantasyPros_2021_Overall_ADP_Rankings.csv")
    assert "justintucker" not in set(parsed.key)
    assert "deepsleeper" not in set(parsed.key)


def test_unranked_player_is_placed_past_the_cap_not_at_the_median(
    adp_dir: Path,
) -> None:
    """The failure this encoding exists to prevent.

    The feature matrices fill missing values with the column median. A player
    the market declined to rank would then be handed an average draft position,
    which asserts the opposite of what his absence says.
    """
    out = add_market_adp_features(
        rows(("Christian McCaffrey", "RB"), ("Nobody At All", "WR")),
        directory=adp_dir,
    )
    assert out.loc[0, "adp_drafted"] == 1.0
    assert out.loc[1, "adp_drafted"] == 0.0
    assert out.loc[1, "adp_rank"] == ADP_DEPTH + 1
    assert out.loc[1, "adp_log_rank"] > out["adp_log_rank"].max() - 1e-9
    assert not out[list(ADP_FEATURES)].isna().to_numpy().any()


def test_positional_rank_is_read_from_the_position_column(adp_dir: Path) -> None:
    out = add_market_adp_features(
        rows(("Christian McCaffrey", "RB"), ("Derrick Henry", "RB")),
        directory=adp_dir,
    )
    assert out.loc[0, "adp_position_log_rank"] == pytest.approx(np.log(1))
    assert out.loc[1, "adp_position_log_rank"] == pytest.approx(np.log(2))


def test_position_disagreement_drops_the_rank(adp_dir: Path) -> None:
    """A shared name across positions must lose its rank, not borrow one."""
    strict = add_market_adp_features(rows(("Travis Kelce", "WR")), directory=adp_dir)
    assert strict.loc[0, "adp_drafted"] == 0.0
    loose = add_market_adp_features(
        rows(("Travis Kelce", "WR")), directory=adp_dir, require_position_match=False
    )
    assert loose.loc[0, "adp_drafted"] == 1.0


def test_ambiguous_name_gets_no_rank_rather_than_a_coin_flip(tmp_path: Path) -> None:
    write_adp(
        tmp_path,
        2021,
        [(1, "Mike Williams  LAC (7)", "WR1"), (40, "Mike Williams  NYJ (10)", "WR12")],
    )
    out = add_market_adp_features(rows(("Mike Williams", "WR")), directory=tmp_path)
    assert out.loc[0, "adp_drafted"] == 0.0


def test_both_stints_of_a_traded_player_get_the_same_rank(adp_dir: Path) -> None:
    traded = pd.DataFrame(
        [
            {"season": 2021, "player_name": "Derrick Henry", "position": "RB", "team": "TEN"},
            {"season": 2021, "player_name": "Derrick Henry", "position": "RB", "team": "BAL"},
        ]
    )
    out = add_market_adp_features(traded, directory=adp_dir)
    assert len(out) == 2
    assert out["adp_rank"].nunique() == 1
    assert out.loc[0, "adp_rank"] == 2


def test_missing_season_file_raises(adp_dir: Path) -> None:
    """A silently absent file marks a whole season undrafted."""
    with pytest.raises(FileNotFoundError, match="no ADP list for 2022"):
        load_adp([2021, 2022], adp_dir)


def test_missing_directory_leaves_the_columns_off(tmp_path: Path) -> None:
    """Absent data is not the same as corrupt data.

    A checkout without the ADP folder still builds frames; the models that
    consume the feature raise on the absent column instead.
    """
    out = add_market_adp_features(
        rows(("Derrick Henry", "RB")), directory=tmp_path / "nope"
    )
    assert not set(ADP_FEATURES) & set(out.columns)


def _volume_pipeline():
    from ffmodel.models.volume_season_average import SeasonAverageVolumePipeline

    return SeasonAverageVolumePipeline()


def test_the_availability_arm_reaches_the_availability_model_only() -> None:
    """The promoted arm is availability; the role layers stay on their own flag."""
    frame = pd.DataFrame(
        {name: [0.0] for name in ADP_FEATURES} | {"player_id": ["a"]}
    )
    pipeline = _volume_pipeline()
    pipeline._enable_market_adp_availability(frame)

    assert set(ADP_FEATURES) <= set(pipeline.availability_model.extra_features)
    for model in (
        pipeline.snap_model,
        pipeline.target_model,
        pipeline.carry_model,
        pipeline.workload_model,
    ):
        assert not set(ADP_FEATURES) & set(model.extra_features)


def test_a_frame_without_the_board_fails_loudly_now_that_it_is_a_default() -> None:
    """The columns are required, not optional, and silence is the failure mode.

    ``_matrix`` drops feature names it cannot find. Degrading quietly to the
    baseline would mean a cache built before these columns existed fits a
    different model from the one that cleared the gate and says nothing.
    """
    pipeline = _volume_pipeline()
    assert pipeline.market_adp_availability_features is True

    with pytest.raises(ValueError, match="absent from the player rows"):
        pipeline._enable_market_adp_availability(pd.DataFrame({"player_id": ["a"]}))


def test_the_quarterback_arm_leaves_the_starter_model_alone() -> None:
    """Same question, same inputs -- feeding both would double-count it."""
    frame = pd.DataFrame(
        {name: [0.0] for name in ADP_FEATURES} | {"player_id": ["a"]}
    )
    pipeline = _volume_pipeline()
    pipeline._enable_market_adp_qb(frame)

    assert set(ADP_FEATURES) <= set(pipeline.workload_model.extra_features)
    assert set(ADP_FEATURES) <= set(pipeline.qb_propensity_model.extra_features)
    starter = getattr(pipeline, "qb_starter_model", None)
    if starter is not None:
        assert not set(ADP_FEATURES) & set(getattr(starter, "extra_features", ()))
