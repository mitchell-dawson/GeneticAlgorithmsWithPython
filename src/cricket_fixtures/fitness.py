from __future__ import annotations

import itertools
from abc import abstractmethod
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from src.cricket_fixtures.chromosome import FixtureListChromosome
from src.cricket_fixtures.database import Database
from src.genetic import AbsoluteFitness

HIGH_WEIGHT = 100
MEDIUM_WEIGHT = 5
LOW_WEIGHT = 1


class FixtureListGranularFitness(AbsoluteFitness):
    @abstractmethod
    def report(self, chromosome: FixtureListChromosome) -> List[str]:
        """Returns a list of strings that can be used to report on the fitness
        of the chromosome"""


class FitnessConfig(Dict[str, Tuple[float, FixtureListGranularFitness]]):
    """A dictionary of fitness functions and their weights"""


class TeamsPlayingMoreThanOnceInAWeek(FixtureListGranularFitness):
    """Counts the number of teams playing more than once in a week across the season"""

    def __init__(self, db: Database):
        self.db = db

    def report(self, chromosome: FixtureListChromosome) -> List[str]:
        """Creates a dict where each key is the team name, and each value is the number
        of times that team plays more than once in a week, only for teams who play more
        than once in a week over the season"""

        lines = []

        for division_num in range(chromosome.num_divisions):

            lines.append(f"Division {division_num}\n\n")

            unique_teams = np.unique(chromosome.genes[division_num, :, :, 0:2])
            unique_teams = unique_teams[unique_teams != 0]

            for week_num in range(chromosome.season_length_in_weeks):

                teams_playing_this_week = chromosome.genes[
                    division_num, week_num, :, 0:2
                ]

                team_nums, week_counts = np.unique(
                    teams_playing_this_week[teams_playing_this_week != 0],
                    return_counts=True,
                )

                team_nums = team_nums[week_counts > 1]
                week_counts = week_counts[week_counts > 1]

                if len(team_nums) > 0:
                    lines.append(f"\tWeek {week_num}\n")

                    for team_num, count in zip(team_nums, week_counts):
                        lines.append(
                            f"\t\t{self.db.get_team_name_by_team_num(team_num).title()} plays {count} times\n"
                        )
            lines.append(f"{'='*50}\n")

        return lines

    @staticmethod
    def __call__(chromosome: FixtureListChromosome) -> int:
        """This is the number of teams playing more than once in a week. Counted
        across all weeks of the season

        Returns
        -------
        int
            The number of teams playing more than once in a week across the season
        """
        total_count = 0

        for division in range(chromosome.num_divisions):

            division_count = 0

            for week_num in range(chromosome.season_length_in_weeks):

                teams_playing_this_week = chromosome.genes[division, week_num, :, 0:2]

                _, week_counts = np.unique(
                    teams_playing_this_week[teams_playing_this_week != 0],
                    return_counts=True,
                )

                division_count += np.sum(week_counts - 1)

            total_count += division_count

        return total_count


class GroundClashes(AbsoluteFitness):
    """There cannot be two games at the same ground in a week"""

    def __init__(self, db: Database):
        self.db = db

    def report(self, chromosome: FixtureListChromosome) -> List[str]:
        """Creates a dict where each key is the ground name, and each value is the number
        of times that ground is used more than once in a week, only for grounds which are
        used than once in a week over the season"""

        lines = []

        for week_num in range(chromosome.season_length_in_weeks):

            grounds_used_in_week = chromosome.genes[:, week_num, :, 2]
            grounds_used_in_week = grounds_used_in_week[grounds_used_in_week != 0]

            ground_nums, week_counts = np.unique(
                grounds_used_in_week, return_counts=True
            )

            ground_nums = ground_nums[week_counts > 1]
            week_counts = week_counts[week_counts > 1]

            if len(ground_nums) > 0:

                lines.append(f"\tWeek {week_num}\n")

                for ground_num, count in zip(ground_nums, week_counts):
                    lines.append(
                        f"\t\t{self.db.get_ground_name_by_ground_num(ground_num).title()} is used {count} times\n"
                    )
                lines.append(f"{'='*50}\n")

        return lines

    @staticmethod
    def __call__(chromosome: FixtureListChromosome) -> float:

        total_count = 0

        for week_num in range(chromosome.season_length_in_weeks):

            grounds_used_in_week = chromosome.genes[:, week_num, :, 2]
            grounds_used_in_week = grounds_used_in_week[grounds_used_in_week != 0]

            _, ground_counts = np.unique(grounds_used_in_week, return_counts=True)

            total_count += sum(ground_counts - 1)

        return total_count


class IncorrectNumberOfFixturesBetweenTwoTeams(AbsoluteFitness):
    """Each pair of teams play each other home and away over a season"""

    @staticmethod
    def __call__(chromosome: FixtureListChromosome) -> float:
        total_count = 0

        for division_num in range(chromosome.num_divisions):

            division_count = 0

            fixtures_in_division = chromosome.genes[division_num, :, :, :2]

            teams_in_division = np.unique(fixtures_in_division)
            teams_in_division = teams_in_division[teams_in_division != 0]

            for team_1 in teams_in_division:
                for team_2 in teams_in_division:

                    if team_1 == team_2:
                        continue

                    division_count += np.abs(
                        np.sum(np.equal([team_1, team_2], fixtures_in_division).all(2))
                        - 1
                    )

        return total_count


class SameTeamsPlayingConsecutively(AbsoluteFitness):
    """The home and away leg of the same fixture does not happen in consecutive weeks"""

    @staticmethod
    def __call__(chromosome: FixtureListChromosome) -> float:

        total_count = 0

        for division_num in range(chromosome.num_divisions):

            division_count = 0

            # count the number of consecutive fixtures between them
            # if not 0, add to fitness
            for week_1 in range(chromosome.season_length_in_weeks - 1):

                week_2 = week_1 + 1

                # get division fixtures in week 1
                week_1_fixtures = chromosome.genes[division_num, week_1, :]

                # get division fixtures in week 2
                week_2_fixtures = chromosome.genes[division_num, week_2, :]

                week_consecutive_fixtures = [
                    [
                        int(
                            fixture_1[0] == fixture_2[1]
                            and fixture_1[1] == fixture_2[0]
                        )
                        for fixture_1 in week_1_fixtures
                        if fixture_1[0] != 0
                    ]
                    for fixture_2 in week_2_fixtures
                    if fixture_2[0] != 0
                ]

                division_count += np.sum(week_consecutive_fixtures)

            total_count += division_count

        return total_count


class TeamHasMoreThanOneWeekOff(AbsoluteFitness):
    """A team does not have two weeks off in a row"""

    def __call__(self, chromosome: FixtureListChromosome) -> float:
        total_count = 0

        for division_num in range(chromosome.num_divisions):

            division_count = self.count_teams_has_more_than_one_week_off_in_division(
                chromosome, division_num
            )

            total_count += division_count

        return total_count

    def count_teams_has_more_than_one_week_off_in_division(
        self, chromosome, division_num
    ):

        division_count = 0

        # get unique teams in division
        unique_teams_in_division = np.unique(chromosome.genes[division_num, :, :, 0:2])

        unique_teams_in_division = unique_teams_in_division[
            unique_teams_in_division != 0
        ]

        # for each team
        for team in unique_teams_in_division:

            longest_team_break = self.count_longest_team_break(
                chromosome, division_num, team
            )

            if longest_team_break > 1:
                division_count += longest_team_break - 1

        return division_count

    def count_longest_team_break(self, chromosome, division_num, team):
        longest_team_break = 0
        current_team_break = 0

        for week_num in range(chromosome.season_length_in_weeks):

            weekly_fixtures = chromosome.genes[division_num, week_num, :]

            team_is_playing_in_week = any(
                [
                    fixture[0] == team or fixture[1] == team
                    for fixture in weekly_fixtures
                    if fixture[0] != 0
                ]
            )

            if team_is_playing_in_week:
                current_team_break = 0

            else:
                current_team_break += 1

                if current_team_break > longest_team_break:
                    longest_team_break = current_team_break

        return longest_team_break


class GroundUsedTwiceInARow(AbsoluteFitness):
    """A ground is not used more than once in two weeks
    Each venue should have at least one week rest in between games
    """

    def __call__(self, chromosome: FixtureListChromosome) -> float:
        total_count = 0

        for week_num_1 in range(chromosome.season_length_in_weeks - 1):

            week_count = 0

            week_num_2 = week_num_1 + 1

            week_1_grounds = chromosome.genes[:, week_num_1, :, 2].flatten()
            week_2_grounds = chromosome.genes[:, week_num_2, :, 2].flatten()

            week_1_grounds = week_1_grounds[week_1_grounds != 0]
            week_2_grounds = week_2_grounds[week_2_grounds != 0]

            week_count = len(
                list((Counter(week_1_grounds) & Counter(week_2_grounds)).elements())
            )

            total_count += week_count

        return total_count


class TeamHasMoreThanTwoHomeGamesInARow(AbsoluteFitness):
    """A team should not play three home games in a row"""

    def __call__(self, chromosome: FixtureListChromosome) -> float:

        total_count = 0

        for division_num in range(chromosome.num_divisions):

            division_count = 0

            # get unique teams in division
            unique_teams_in_division = np.unique(
                chromosome.genes[division_num, :, :, 0:2].flatten()
            )

            unique_teams_in_division = unique_teams_in_division[
                unique_teams_in_division != 0
            ]

            for team in unique_teams_in_division:

                team_count = 0

                most_consecutive_home_games = 0
                current_consecutive_home_games = 0

                for week_num in range(chromosome.season_length_in_weeks):

                    # find all fixtures in division in week
                    week_fixtures = [
                        fixture
                        for fixture in chromosome.genes[division_num, week_num, :]
                        if fixture is not None
                    ]

                    # if team has a home fixture
                    team_has_home_fixture = team in [
                        fixture[0] for fixture in week_fixtures
                    ]

                    # if team has a away fixture
                    team_has_away_fixture = team in [
                        fixture[1] for fixture in week_fixtures
                    ]

                    if team_has_home_fixture:
                        current_consecutive_home_games += 1

                        if current_consecutive_home_games > most_consecutive_home_games:
                            most_consecutive_home_games = current_consecutive_home_games

                    elif team_has_away_fixture:
                        current_consecutive_home_games = 0
                    else:
                        # team has no fixture in week
                        pass

                if most_consecutive_home_games > 2:
                    team_count += most_consecutive_home_games - 2

                division_count += team_count

            total_count += division_count

        return total_count


class FixtureListFitness(AbsoluteFitness):
    """Fitness function for a fixture list chromosome"""

    chromosome: FixtureListChromosome

    def __init__(self, db, fitness_config: FitnessConfig) -> None:

        self.fitness_config = fitness_config

        self.unique_teams_in_division = [
            db.get_team_nums_in_division(division_num)
            for division_num in db.division_nums()
        ]

        self.unique_pairs_of_teams_in_division = [
            list(itertools.combinations(teams, 2))
            for teams in self.unique_teams_in_division
        ]

        multi_division_pairs_with_duplicates = [
            list(itertools.product(teams, teams))
            for teams in self.unique_teams_in_division
        ]

        self.multi_division_pairs = []

        for division_pairs_with_duplicates in multi_division_pairs_with_duplicates:
            self.multi_division_pairs.append(
                [
                    division_pairs
                    for division_pairs in division_pairs_with_duplicates
                    if division_pairs[0] != division_pairs[1]
                ]
            )

        self.empty_fixture_count_dict = [
            {
                f"{team_1}_{team_2}": 0
                for team_1, team_2 in self.multi_division_pairs[division_num]
            }
            for division_num in db.division_nums()
        ]

    def report(self, chromosome: FixtureListChromosome) -> Dict[str, List[str]]:

        return {
            fitness_name: fitness_fn.report(chromosome)
            for fitness_name, (weight, fitness_fn) in self.fitness_config.items()
        }

    def __call__(self, chromosome: FixtureListChromosome) -> float:
        """Return the fitness of a chromosome. The fitness is the absolute difference
        between the sum of the rows, columns, and diagonals and the expected sum.

        Parameters
        ----------
        chromosome : FixtureListChromosome
            The chromosome to evaluate.

        Returns
        -------
        int
            The fitness of the chromosome.
        """

        return -sum(
            [
                weight * fitness_fn(chromosome)
                for weight, fitness_fn in self.fitness_config.values()
            ]
        )
