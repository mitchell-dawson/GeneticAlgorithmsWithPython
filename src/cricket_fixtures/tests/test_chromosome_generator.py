from typing import List

import pytest

from src.cricket_fixtures.chromosome import FixtureListChromosome, Team
from src.cricket_fixtures.cricket_fixtures import FixtureListChromosomeGenerator
from src.cricket_fixtures.database import Database


@pytest.fixture(name="single_division_teams_db")
def fixture_single_division_teams(data_fixtures_folder) -> List[Team]:
    """Fixture to create a list of teams."""

    csv_file_path = (
        data_fixtures_folder / "raw/cricket_fixtures/single_division_unique_grounds.csv"
    )

    db = Database(csv_file_path)

    return db


def test_call_FixtureListChromosomeGenerator(single_division_teams_db):
    """GIVEN a FixtureListChromosomeGenerator
    WHEN the FixtureListChoromosomeGenerator is called
    THEN the method should return a FixtureList filled with fixtures.
    """

    num_divisions = 1
    season_length_in_weeks = len(single_division_teams_db) * 2

    generator = FixtureListChromosomeGenerator(
        single_division_teams_db,
        season_length_in_weeks,
    )

    chromosome = generator()

    assert isinstance(chromosome, FixtureListChromosome)
    assert chromosome.age == 0

    assert chromosome.genes.shape == (
        num_divisions,
        season_length_in_weeks,
        len(single_division_teams_db) // 2,
        3,
    )
