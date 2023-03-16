from __future__ import annotations

import copy
import logging
import random
from abc import abstractmethod

import numpy as np

from src.cricket_fixtures.chromosome import FixtureListChromosome
from src.genetic import Mutation


class FixtureListMutation(Mutation):
    """Mutation function for the magic squares problem."""

    @abstractmethod
    def mutate(self, chromosome: FixtureListChromosome) -> FixtureListChromosome:
        """Mutate a chromosome."""

    def __call__(self, parent: FixtureListChromosome) -> FixtureListChromosome:
        """Mutate a chromosome by swapping two random elements. Pick two random
        indices, looping until the indices are distinct, and swap the elements at those
        indices.

        Parameters
        ----------
        parent : FixtureListChromosome
            The parent chromosome to mutate.

        Returns
        -------
        FixtureListChromosome
            The mutated chromosome.
        """

        copy_of_parent = copy.deepcopy(parent)
        return self.mutate(copy_of_parent)

    def choose_two_fixtures(self, child):

        # Choose a random game
        week_num_1 = self.choose_random_week(child)
        game_num_1 = self.choose_random_game(child)

        # choose a random game from another week
        while True:
            week_num_2 = self.choose_random_week(child)
            if week_num_1 != week_num_2:
                break

        game_num_2 = self.choose_random_game(child)

        return week_num_1, game_num_1, week_num_2, game_num_2

    @staticmethod
    def choose_random_week(child):
        return random.randint(0, child.season_length_in_weeks - 1)

    @staticmethod
    def choose_random_game(child):
        return random.randint(0, child.max_games_per_week - 1)

    @staticmethod
    def swap_two_fixtures(
        division_num, week_num_1, week_num_2, game_num_1, game_num_2, child
    ):

        fixture_1 = np.copy(child.genes[division_num, week_num_1, game_num_1, :])
        fixture_2 = np.copy(child.genes[division_num, week_num_2, game_num_2, :])

        child.genes[division_num, week_num_1, game_num_1, :] = fixture_2
        child.genes[division_num, week_num_2, game_num_2, :] = fixture_1

        return child


class SwapUpToNFixturesMutation(FixtureListMutation):
    """Mutation function which swaps up to N fixtures"""

    def __init__(self, max_num_to_swap: int):
        self.max_num_to_swap = max_num_to_swap

    def mutate(self, chromosome: FixtureListChromosome) -> FixtureListChromosome:

        # for each division
        divisions = random.sample(
            range(chromosome.num_divisions),
            k=random.randrange(1, chromosome.num_divisions + 1),
        )

        # for division_num in range(chromosome.num_divisions):
        for division_num in divisions:

            for _ in range(random.randrange(1, self.max_num_to_swap + 1)):

                (
                    week_num_1,
                    game_num_1,
                    week_num_2,
                    game_num_2,
                ) = self.choose_two_fixtures(chromosome)

                logging.debug(
                    "Swapping week {%d}, game {%d} with week {%d}, game {%d}",
                    week_num_1,
                    game_num_1,
                    week_num_2,
                    game_num_2,
                )

                chromosome = self.swap_two_fixtures(
                    division_num,
                    week_num_1,
                    week_num_2,
                    game_num_1,
                    game_num_2,
                    chromosome,
                )

        return chromosome


class TheNicolaSwitchProcesserMutation(FixtureListMutation):
    """Mutation function which swaps up to N fixtures"""

    def mutate(self, chromosome: FixtureListChromosome) -> FixtureListChromosome:

        for division_num in range(chromosome.num_divisions):

            team_nums, week_nums = self.highlight_duplicate_in_division(
                chromosome, division_num
            )

            for zip_ in zip(team_nums, week_nums):

                team_num, week_num = zip_

                duplicate_fixtures, game_nums = self.find_duplicate_fixtures(
                    chromosome,
                    division_num,
                    week_num,
                    team_num,
                )

                for ii, duplicate_fixture in enumerate(duplicate_fixtures):

                    for other_week_num in range(chromosome.season_length_in_weeks):

                        if other_week_num == week_num:
                            continue

                        if self.team_plays_in_week(
                            chromosome,
                            division_num,
                            other_week_num,
                            duplicate_fixture[0],
                        ):
                            continue

                        if self.team_plays_in_week(
                            chromosome,
                            division_num,
                            other_week_num,
                            duplicate_fixture[1],
                        ):
                            continue

                        for other_game_num in range(chromosome.max_games_per_week):

                            if (
                                chromosome.genes[
                                    division_num, other_week_num, other_game_num, 0
                                ]
                                != 0
                            ):
                                continue

                            chromosome = self.swap_two_fixtures(
                                division_num,
                                week_num,
                                other_week_num,
                                game_nums[ii],
                                other_game_num,
                                chromosome,
                            )

                            return chromosome
        return chromosome

    def team_plays_in_week(self, chromosome, division_num, week_num, team_num):
        return np.any(chromosome.genes[division_num, week_num, :, 0:2] == team_num)

    def find_duplicate_fixtures(self, chromosome, division_num, week_num, team_num):
        return (
            chromosome.genes[
                division_num,
                week_num,
                np.any(
                    chromosome.genes[division_num, week_num, :, 0:2] == team_num, axis=1
                ),
                :,
            ],
            np.where(
                np.any(
                    chromosome.genes[division_num, week_num, :, 0:2] == team_num, axis=1
                )
            )[0],
        )

    def highlight_duplicate_in_division(self, chromosome, division_num):

        unique_teams = np.unique(chromosome.genes[division_num, :, :, 0:2])
        unique_teams = unique_teams[unique_teams != 0]

        team_nums = []
        week_nums = []

        for week_num in range(chromosome.season_length_in_weeks):

            teams_playing_this_week = chromosome.genes[division_num, week_num, :, 0:2]

            week_team_nums, week_counts = np.unique(
                teams_playing_this_week[teams_playing_this_week != 0],
                return_counts=True,
            )

            week_team_nums = week_team_nums[week_counts > 1]
            week_counts = week_counts[week_counts > 1]

            # if can't find any duplicates, skip to next week
            if len(week_counts) == 0:
                continue

            # if can find duplicates, find the teams that are playing more than once
            for team_num in week_team_nums:
                team_nums.append(team_num)
                week_nums.append(week_num)

        return team_nums, week_nums
