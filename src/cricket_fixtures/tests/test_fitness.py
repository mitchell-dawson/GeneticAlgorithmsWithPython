import numpy as np
import pytest

from src.cricket_fixtures.chromosome import FixtureListChromosome
from src.cricket_fixtures.fitness import (
    GroundClashes,
    GroundUsedTwiceInARow,
    IncorrectNumberOfFixturesBetweenTwoTeams,
    SameTeamsPlayingConsecutively,
    TeamHasMoreThanOneWeekOff,
    TeamHasMoreThanTwoHomeGamesInARow,
    TeamsPlayingMoreThanOnceInAWeek,
)


def get_two_team_one_division_fixture_list_chromosome():
    """A fixture for a two team fixture list chromosome"""

    genes = np.array(
        [
            [
                [[1, 2, 1]],
                [[2, 1, 2]],
            ]
        ]
    )
    assert genes.shape == (1, 2, 1, 3)
    return FixtureListChromosome(genes=genes)


def get_two_team_one_division_with_zeros_fixture_list_chromosome():
    """A fixture for a two team fixture list chromosome"""

    genes = np.array(
        [
            [
                [[1, 2, 1], [0, 0, 0]],
                [[2, 1, 2], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
            ]
        ]
    )
    assert genes.shape == (1, 4, 2, 3)
    return FixtureListChromosome(genes=genes)


def get_two_team_two_division_fixture_list_chromosome():
    """A fixture for a two team fixture list chromosome"""

    genes = np.array(
        [
            [
                [[1, 2, 1]],
                [[2, 1, 2]],
            ],
            [
                [[3, 4, 1]],
                [[4, 3, 2]],
            ],
        ]
    )
    assert genes.shape == (2, 2, 1, 3)
    return FixtureListChromosome(genes=genes)


def get_three_team_one_division_fixture_list_chromosome():
    """A fixture for a three team fixture list chromosome"""

    genes = np.array(
        [
            [
                [[1, 3, 1]],
                [[1, 2, 1]],
                [[2, 1, 2]],
                [[3, 1, 3]],
                [[3, 2, 3]],
                [[2, 3, 2]],
            ]
        ]
    )

    assert genes.shape == (1, 6, 1, 3)
    return FixtureListChromosome(genes=genes)


@pytest.mark.parametrize(
    "chromosome,expected_fitness",
    [
        (get_three_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_with_zeros_fixture_list_chromosome(), 0),
        (get_two_team_two_division_fixture_list_chromosome(), 2),
    ],
)
def test_call_GroundClashes(chromosome, expected_fitness):
    """GIVEN a fixture list chromosome
    WHEN the ground clashes fitness function is called
    THEN the number of ground clashes should be returned.
    """

    fitness = GroundClashes()
    assert fitness(chromosome) == expected_fitness


@pytest.mark.parametrize(
    "chromosome,expected_fitness",
    [
        (get_three_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_with_zeros_fixture_list_chromosome(), 0),
        (get_two_team_two_division_fixture_list_chromosome(), 0),
    ],
)
def test_call_IncorrectNumberOfFixturesBetweenTwoTeams(chromosome, expected_fitness):
    fitness = IncorrectNumberOfFixturesBetweenTwoTeams()
    assert fitness(chromosome) == expected_fitness


@pytest.mark.parametrize(
    "chromosome,expected_fitness",
    [
        (get_three_team_one_division_fixture_list_chromosome(), 2),
        (get_two_team_one_division_fixture_list_chromosome(), 1),
        (get_two_team_one_division_with_zeros_fixture_list_chromosome(), 1),
        (get_two_team_two_division_fixture_list_chromosome(), 2),
    ],
)
def test_call_SameTeamsPlayingConsecutively(chromosome, expected_fitness):
    fitness = SameTeamsPlayingConsecutively()
    assert fitness(chromosome) == expected_fitness


@pytest.mark.parametrize(
    "chromosome,expected_fitness",
    [
        (get_three_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_with_zeros_fixture_list_chromosome(), 0),
        (get_two_team_two_division_fixture_list_chromosome(), 0),
    ],
)
def test_call_TeamsPlayingMoreThanOnceInAWeek(chromosome, expected_fitness):
    fitness = TeamsPlayingMoreThanOnceInAWeek()
    assert fitness(chromosome) == expected_fitness


@pytest.mark.parametrize(
    "chromosome,expected_fitness",
    [
        (get_three_team_one_division_fixture_list_chromosome(), 2),
        (get_two_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_with_zeros_fixture_list_chromosome(), 2),
        (get_two_team_two_division_fixture_list_chromosome(), 0),
    ],
)
def test_call_TeamHasMoreThanOneWeekOff(chromosome, expected_fitness):
    fitness = TeamHasMoreThanOneWeekOff()
    assert fitness(chromosome) == expected_fitness


@pytest.mark.parametrize(
    "chromosome,expected_fitness",
    [
        (get_three_team_one_division_fixture_list_chromosome(), 2),
        (get_two_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_with_zeros_fixture_list_chromosome(), 0),
        (get_two_team_two_division_fixture_list_chromosome(), 0),
    ],
)
def test_call_GroundUsedTwiceInARow(chromosome, expected_fitness):
    fitness = GroundUsedTwiceInARow()
    assert fitness(chromosome) == expected_fitness


@pytest.mark.parametrize(
    "chromosome,expected_fitness",
    [
        (get_three_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_fixture_list_chromosome(), 0),
        (get_two_team_one_division_with_zeros_fixture_list_chromosome(), 0),
        (get_two_team_two_division_fixture_list_chromosome(), 0),
    ],
)
def test_call_TeamHasMoreThanTwoHomeGamesInARow(chromosome, expected_fitness):
    fitness = TeamHasMoreThanTwoHomeGamesInARow()
    assert fitness(chromosome) == expected_fitness
