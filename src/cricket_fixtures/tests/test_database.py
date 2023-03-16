from typing import List

import pytest

from src.cricket_fixtures.chromosome import Team
from src.cricket_fixtures.cricket_fixtures import Database


@pytest.fixture(name="single_division_teams_db")
def fixture_single_division_teams(data_fixtures_folder) -> List[Team]:
    """Fixture to create a list of teams."""

    csv_file_path = (
        data_fixtures_folder / "raw/cricket_fixtures/single_division_unique_grounds.csv"
    )

    db = Database(csv_file_path)

    return db


@pytest.fixture(name="three_divisions_unique_grounds_teams_db")
def fixture_three_divisions_unique_grounds_teams(data_fixtures_folder) -> List[Team]:
    """Fixture to create a list of teams."""

    csv_file_path = (
        data_fixtures_folder / "raw/cricket_fixtures/three_divisions_unique_grounds.csv"
    )

    db = Database(csv_file_path)

    return db


@pytest.fixture(name="three_divisions_shared_grounds_teams_db")
def fixture_three_divisions_shared_grounds_teams(data_fixtures_folder) -> List[Team]:
    """Fixture to create a list of teams."""

    csv_file_path = (
        data_fixtures_folder / "raw/cricket_fixtures/three_divisions_shared_grounds.csv"
    )

    db = Database(csv_file_path)

    return db


def test_get_team_id_by_name(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the team_id is retrieved by name
    THEN the correct team_id should be returned.
    """

    team_id = single_division_teams_db.get_team_num_by_name("manchester united")
    assert team_id == 1


def test_get_team_name_by_team_num(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the team_name is retrieved by team_num
    THEN the correct team_name should be returned.
    """

    team_name = single_division_teams_db.get_team_name_by_team_num(1)
    assert team_name == "manchester united"


def test_get_ground_num_by_name(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the ground_id is retrieved by name
    THEN the correct ground_id should be returned.
    """

    ground_id = single_division_teams_db.get_ground_num_by_name("old trafford")
    assert ground_id == 1


def test_get_ground_name_by_ground_num(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the ground_name is retrieved by ground_num
    THEN the correct ground_name should be returned.
    """

    ground_name = single_division_teams_db.get_ground_name_by_ground_num(1)
    assert ground_name == "old trafford"


def test_get_division_num_by_name(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the division_id is retrieved by name
    THEN the correct division_id should be returned.
    """

    division_id = single_division_teams_db.get_division_num_by_name("premier league")
    assert division_id == 0


def test_get_division_name_by_division_num(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the division_name is retrieved by division_num
    THEN the correct division_name should be returned.
    """

    division_name = single_division_teams_db.get_division_name_by_division_num(0)
    assert division_name == "premier league"


def test_get_max_games_per_week_in_largest_division(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the max_games_per_week_in_largest_division is retrieved
    THEN the correct max_games_per_week_in_largest_division should be returned.
    """

    max_games_per_week_in_largest_division = (
        single_division_teams_db.get_max_games_per_week_in_largest_division()
    )
    assert max_games_per_week_in_largest_division == 4


def test_num_divisions(three_divisions_shared_grounds_teams_db):
    """GIVEN a list of teams
    WHEN the number of divisions is retrieved
    THEN the correct number of divisions should be returned.
    """

    num_divisions = three_divisions_shared_grounds_teams_db.num_divisions()
    assert num_divisions == 3


def test_get_ground_name_by_team_num(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the ground is retrieved by team_num
    THEN the correct ground should be returned.
    """

    ground = single_division_teams_db.get_ground_name_by_team_num(1)
    assert ground == "old trafford"


def test_get_division_name_by_team_num(single_division_teams_db):
    """GIVEN a list of teams
    WHEN the division is retrieved by team_num
    THEN the correct division should be returned.
    """

    division = single_division_teams_db.get_division_name_by_team_num(1)
    assert division == "premier league"


def test_read_single_division_Dataset(single_division_teams_db):
    assert single_division_teams_db.df["division"].unique() == ["premier league"]


def test_read_three_division_unique_grounds_Dataset(
    three_divisions_unique_grounds_teams_db,
):

    assert set(three_divisions_unique_grounds_teams_db.df["division"].unique()) == set(
        [
            "premier league",
            "championship",
            "division 2",
        ]
    )

    assert len(three_divisions_unique_grounds_teams_db.df["ground"].unique()) == 31


def test_read_three_division_shared_grounds_Dataset(
    three_divisions_shared_grounds_teams_db,
):

    assert set(three_divisions_shared_grounds_teams_db.df["division"].unique()) == set(
        [
            "premier league",
            "championship",
            "division 2",
        ]
    )

    assert len(three_divisions_shared_grounds_teams_db.df["ground"].unique()) == 20
