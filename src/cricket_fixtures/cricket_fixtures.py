from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Sequence, Union

from src.cricket_fixtures.chromosome import FixtureListChromosome, print_chromosome
from src.cricket_fixtures.chromosome_generator import FixtureListChromosomeGenerator
from src.cricket_fixtures.database import Database
from src.cricket_fixtures.fitness import (
    HIGH_WEIGHT,
    LOW_WEIGHT,
    FixtureListFitness,
    GroundClashes,
    TeamsPlayingMoreThanOnceInAWeek,
)
from src.cricket_fixtures.mutation import (
    FixtureListMutation,
    SwapUpToNFixturesMutation,
    TheNicolaSwitchProcesserMutation,
)
from src.cricket_fixtures.report_writers import (
    FitnessDetailsReportWriter,
    FitnessReportReportWriter,
    FormattedGenesReportWriter,
    RawGenesReportWriter,
    ReportWriter,
)
from src.cricket_fixtures.stopping_criteria import FixtureListStoppingCriteria
from src.genetic import (
    AgeAnnealing,
    FitnessStagnationDetector,
    Mutation,
    Runner,
    StoppingCriteria,
)

random.seed(42)


class FixtureListRunner(Runner):
    def __init__(
        self,
        chromosome_generator: FixtureListChromosomeGenerator,
        fitness: FixtureListFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
        age_annealing: AgeAnnealing,
        fitness_stagnation_detector: FitnessStagnationDetector,
        report_writers: Sequence[ReportWriter],
    ):
        super().__init__(
            chromosome_generator,
            fitness,
            stopping_criteria,
            mutate,
            age_annealing,
            fitness_stagnation_detector,
        )
        self.report_writers = report_writers

    def display(self, candidate):
        # time_diff = time.time() - self.start_time

        # logging.info("fitness=%s", self.fitness(candidate))

        for report_writer in self.report_writers:
            report_writer.write(candidate)


def cricket_fixtures_problem(
    db: Database,
    season_length_in_weeks: int,
    output_folder: Path,
    fitness_stagnation_limit: Union[float, int] = float("inf"),
    age_limit: float = float("inf"),
) -> FixtureListChromosome:

    target = 0  # all league fixture criteria are met

    fitness_config = {
        "teams_playing_more_than_once_in_a_week": (
            HIGH_WEIGHT,
            TeamsPlayingMoreThanOnceInAWeek(db),
        ),
        "ground_clashes": (LOW_WEIGHT, GroundClashes(db)),
    }

    fitness = FixtureListFitness(db, fitness_config)
    chromosome_generator = FixtureListChromosomeGenerator(db, season_length_in_weeks)
    stopping_criteria = FixtureListStoppingCriteria(target, fitness)
    age_annealing = AgeAnnealing(age_limit=age_limit)

    mutation = SwapUpToNFixturesMutation(2)
    mutation = (
        TheNicolaSwitchProcesserMutation()
    )  # <- this is the best mutation I have found so far

    fitness_stagnation_detector = FitnessStagnationDetector(
        fitness, fitness_stagnation_limit
    )

    report_writers = [
        FormattedGenesReportWriter(db, output_folder),
        RawGenesReportWriter(output_folder),
        FitnessReportReportWriter(fitness, db, output_folder),
        FitnessDetailsReportWriter(fitness, db, output_folder),
    ]

    runner = FixtureListRunner(
        chromosome_generator,
        fitness,
        stopping_criteria,
        mutation,
        age_annealing,
        fitness_stagnation_detector,
        report_writers,
    )

    best = runner.run()
    return best


def check_season_length_validity(db: Database, season_length_in_weeks: int):

    num_division_teams = [
        len(db.get_team_nums_in_division(division_num))
        for division_num in db.division_nums()
    ]

    max_division_size = max(num_division_teams)

    min_season_length_in_weeks = 2 * (max_division_size - 1)

    if not season_length_in_weeks >= min_season_length_in_weeks:
        raise ValueError(
            f"Season length must be at least {min_season_length_in_weeks} weeks."
        )


def main():

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    season_length_in_weeks = 27
    age_limit = 2000

    root_folder = Path(
        "/Users/mitchell.dawson/CodeProjects/GeneticAlgorithmsWithPython/"
    )

    output_folder = root_folder / "data/processed/cricket_fixtures/outputs"

    csv_file_path = root_folder / "data_fixtures/raw/cricket_fixtures/cricket_clubs.csv"
    # csv_file_path = (
    #     root_folder
    #     / "data_fixtures/raw/cricket_fixtures/three_divisions_shared_grounds.csv"
    # )
    # csv_file_path = (
    #     root_folder
    #     / "data_fixtures/raw/cricket_fixtures/three_divisions_unique_grounds.csv"
    # )
    # csv_file_path = (
    #     root_folder
    #     / "data_fixtures/raw/cricket_fixtures/single_division_unique_grounds.csv"
    # )

    db = Database(csv_file_path)

    check_season_length_validity(db, season_length_in_weeks)

    best = cricket_fixtures_problem(
        db, season_length_in_weeks, output_folder, age_limit=age_limit
    )

    print_chromosome(best)

    logging.info("best=%s", best)


if __name__ == "__main__":
    main()
