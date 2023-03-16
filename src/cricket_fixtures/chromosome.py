from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

import numpy as np

from src.genetic import Chromosome


@dataclass(slots=True)
class Team:
    """A cricket team"""

    name: str
    ground: Ground
    division: Division
    team_num: int

    def __eq__(self, other: Team) -> bool:
        return self.team_num == other.team_num

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(slots=True)
class Ground:
    """A cricket ground where fixtures are played"""

    name: str
    ground_num: int

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Ground) -> bool:
        return self.name == other.name

    def __lt__(self, other: Ground) -> bool:
        return self.name < other.name


@dataclass(slots=True)
class Division:
    """A cricket division"""

    name: str
    division_num: int

    def __hash__(self) -> int:
        return hash(self.division_num)


@dataclass
class Fixture:
    """A fixture between two teams"""

    home_team: Team
    away_team: Team

    def __post_init__(self):
        self.ground = self.home_team.ground
        self.division = self.home_team.division

    def __str__(self):
        return f"{self.home_team} vs {self.away_team}"

    def __repr__(self):
        return f"[{self.home_team.team_num}, {self.away_team.team_num}]"

    def __eq__(self, other: Fixture) -> bool:
        return self.home_team == other.home_team and self.away_team == other.away_team

    def __lt__(self, other: Optional[Fixture]) -> bool:

        if other is None:
            return False

        return self.home_team.team_num < other.home_team.team_num

    def has_same_teams(self, other: Fixture) -> bool:
        """Check whether this fixture involves the same teams
        as another fixtures regardless of home or away

        Parameters
        ----------
        other : Fixture
            Another feature to compare to

        Returns
        -------
        bool
            True if other fixtures involves same teams
        """
        return (
            self.home_team == other.home_team and self.away_team == other.away_team
        ) or (self.home_team == other.away_team and self.away_team == other.home_team)


@dataclass
class FixtureListChromosome(Chromosome):
    """Chromosome for the cricket fixtures problem."""

    genes: np.ndarray
    age: int = 0

    def __post_init__(self):
        (
            self.num_divisions,
            self.season_length_in_weeks,
            self.max_games_per_week,
            _,
        ) = self.genes.shape

    def __str__(self) -> str:
        """Return a visualisation of the chromosome."""
        return str(self.genes)

    def __repr__(self) -> str:
        """Return a visualisation of the chromosome."""
        return str(self.genes)

    @staticmethod
    def empty(
        num_divisions,
        season_length_in_weeks: int,
        max_games_per_week_in_largest_division: int,
    ):
        """Create an empty chromosome."""
        return FixtureListChromosome(
            genes=np.zeros(
                (
                    num_divisions,
                    season_length_in_weeks,
                    max_games_per_week_in_largest_division,
                    3,
                ),
                dtype=int,
            )
        )

    def unique_teams(self) -> Set[Team]:
        """Return a set of unique teams in the chromosome."""

        return set(
            [
                fixture.home_team
                for fixture in self.genes.flatten()
                if fixture is not None
            ]
        )

    def is_fixture_set(self, division_num: int, week_num: int, game_num: int) -> bool:
        """Check whether a fixture is set at a given position

        Parameters
        ----------
        division_num : int
            The division number to check
        week_num : int
            The week number to check
        game_num : int
            The game number to check

        Returns
        -------
        bool
            True if a fixture is set at the given position, False otherwise
        """
        return np.all(self.genes[division_num, week_num, game_num] != 0)

    def add_fixture(
        self, fixture: Fixture, division_num: int, week_num: int, game_num: int
    ) -> None:
        """Add a fixture to the chromosome at a given position

        Parameters
        ----------
        fixture : Fixture
            A fixture to add to the chromosome
        division_num : int
            The division number to add the fixture to
        week_num : int
            The week number to add the fixture to
        game_num : int
            The game number to add the fixture to
        """

        self.genes[division_num, week_num, game_num, :] = [
            fixture.home_team.team_num,
            fixture.away_team.team_num,
            fixture.ground.ground_num,
        ]

    def add_fixture_to_empty_slot(self, fixture: Fixture, division_num: int) -> None:
        """Add a fixture to the chromosome at the first empty slot

        Parameters
        ----------
        fixture : Fixture
            A fixture to add to the chromosome
        division_num : int
            The division number to add the fixture to

        Raises
        ------
        IndexError
            If there are no empty slots in the chromosome for the given division
        """

        for game_num in range(self.max_games_per_week):
            for week_num in range(self.season_length_in_weeks):

                if not self.is_fixture_set(division_num, week_num, game_num):
                    self.add_fixture(fixture, division_num, week_num, game_num)
                    return

        raise IndexError("No empty slots, Season is likely too short")


def print_chromosome(chromosome: FixtureListChromosome):

    for division_num in range(chromosome.num_divisions):

        print(f"Division {division_num}")

        for week_num in range(chromosome.season_length_in_weeks):
            print(f"Week {week_num} (Division {division_num})")

            for fixture in chromosome.genes[division_num, week_num, :, :]:
                print(f"{fixture}")

            print("")

        print("")
