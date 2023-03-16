from __future__ import annotations

import itertools
import random

from src.cricket_fixtures.chromosome import Fixture, FixtureListChromosome, Team
from src.cricket_fixtures.database import Database
from src.genetic import ChromosomeGenerator


class FixtureListChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, db: Database, season_length_in_weeks: int) -> None:
        self.season_length_in_weeks = season_length_in_weeks
        self.db = db

    def __call__(self) -> FixtureListChromosome:
        """Generate a starting chromosome for the cricket fixture list problem. The
        chromosome is an array of fixtures, with each fixture containing a home team and
        an away team. The fixture list is first initialised as empty, then all possible
        combinations of teams are added to the fixture list, with each team playing
        home and away against each other.

        Returns
        -------
        FixtureListChromosome
            The starting chromosome.
        """

        chromosome = FixtureListChromosome.empty(
            self.db.num_divisions(),
            self.season_length_in_weeks,
            self.db.get_max_games_per_week_in_largest_division(),
        )

        for division_num in range(self.db.num_divisions()):

            # find teams in this division
            division_teams = self.db.get_team_nums_in_division(division_num)

            division_team_pairs = list(itertools.combinations(division_teams, 2))

            random.shuffle(division_team_pairs)

            for team_1_num, team_2_num in division_team_pairs:

                team_1_division_name = self.db.get_division_name_by_team_num(team_1_num)
                team_2_division_name = self.db.get_division_name_by_team_num(team_2_num)

                if team_1_division_name != team_2_division_name:
                    continue

                team_1_name = self.db.get_team_name_by_team_num(team_1_num)

                team_1 = Team(
                    team_1_name,
                    self.db.get_ground_by_team_num(team_1_num),
                    self.db.get_division_by_team_num(team_1_num),
                    team_num=team_1_num,
                )

                team_2_name = self.db.get_team_name_by_team_num(team_2_num)

                team_2 = Team(
                    team_2_name,
                    self.db.get_ground_by_team_num(team_2_num),
                    self.db.get_division_by_team_num(team_2_num),
                    team_num=team_2_num,
                )

                chromosome.add_fixture_to_empty_slot(
                    Fixture(team_1, team_2), division_num
                )

                chromosome.add_fixture_to_empty_slot(
                    Fixture(team_2, team_1), division_num
                )

        return chromosome
