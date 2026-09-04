"""The stadium coordinate table, which is hand-entered and fails silently.

A wrong latitude does not raise. It returns entirely plausible weather for the
wrong place, and every downstream number stays believable while being about
another city. The fetcher's real defence is the per-stadium correlation check
against what nflverse recorded for the same game; these tests cover the errors
that can be caught without a network call — a duplicate key, a swapped pair, a
sign error putting a US stadium in the wrong hemisphere.
"""

import pandas as pd
import pytest

from ffmodel.config import MANUAL_DATA_DIR

COORDINATES = MANUAL_DATA_DIR / "stadium_coordinates.csv"

# Every stadium in the table is either in North America or is one of the named
# international sites. Anything outside these boxes is a typo, not a venue.
NORTH_AMERICA = {"lat": (19.0, 48.5), "lon": (-123.0, -70.0)}
INTERNATIONAL = {"LON00", "LON01", "LON02", "GER00", "FRA00", "SAO00", "MEX00"}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(COORDINATES)


def test_the_table_exists_and_is_keyed_on_stadium_id(table):
    assert not table.empty
    assert table["stadium_id"].is_unique, "stadium_id is the join key and must be unique"
    assert table["stadium_id"].notna().all()


def test_no_coordinate_is_missing(table):
    """A blank coordinate would be fetched as a NaN request, not an error."""
    assert table["latitude"].notna().all()
    assert table["longitude"].notna().all()


def test_domestic_stadiums_sit_inside_north_america(table):
    """Catches a sign error or a swapped lat/lon pair, which are the likely slips."""
    domestic = table[~table["stadium_id"].isin(INTERNATIONAL)]
    low, high = NORTH_AMERICA["lat"]
    outside = domestic[~domestic["latitude"].between(low, high)]
    assert outside.empty, f"latitude outside North America: {outside.stadium_id.tolist()}"
    low, high = NORTH_AMERICA["lon"]
    outside = domestic[~domestic["longitude"].between(low, high)]
    assert outside.empty, f"longitude outside North America: {outside.stadium_id.tolist()}"


def test_longitude_is_not_silently_swapped_with_latitude(table):
    """A swap survives the box test for stadiums near the diagonal, so check the
    invariant that separates the two: no NFL venue has |latitude| > |longitude|
    in the Americas, because they all sit far west of the prime meridian."""
    americas = table[~table["stadium_id"].isin({"LON00", "LON01", "LON02", "GER00", "FRA00"})]
    swapped = americas[americas["latitude"].abs() > americas["longitude"].abs()]
    assert swapped.empty, f"latitude/longitude look swapped: {swapped.stadium_id.tolist()}"


def test_no_two_stadiums_share_a_location(table):
    """Two venues at identical coordinates means a row was copied and not edited.

    Genuine near-neighbours exist -- the Georgia Dome and Mercedes-Benz Stadium
    are adjacent -- so this is an exact-duplicate check rather than a distance one.
    """
    pairs = table[["latitude", "longitude"]].round(4)
    duplicated = pairs.duplicated(keep=False)
    assert not duplicated.any(), (
        f"identical coordinates: {table.loc[duplicated, 'stadium_id'].tolist()}"
    )


def test_every_row_carries_a_human_readable_name(table):
    """The note and stadium columns are what a reviewer checks a coordinate against."""
    assert table["stadium"].notna().all()
    assert (table["stadium"].astype(str).str.len() > 3).all()
