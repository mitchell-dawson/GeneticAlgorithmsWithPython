from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.cricket_fixtures.chromosome import FixtureListChromosome
from src.cricket_fixtures.database import Database
from src.cricket_fixtures.fitness import FixtureListFitness


class ReportWriter(ABC):

    output_folder: Path
    output_file_name: str

    @abstractmethod
    def get_formatted_lines(self, candidate: FixtureListChromosome) -> List[str]:
        """Get the formatted lines to write to the output file."""

    def write(self, candidate: FixtureListChromosome):

        output_lines = self.get_formatted_lines(candidate)

        output_file = self.output_folder / self.output_file_name

        self.write_output_lines_to_file(output_lines, output_file)

    @staticmethod
    def write_output_lines_to_file(output_lines: List[str], output_file: str):

        with open(output_file, "w", encoding="utf-8") as out_f:
            for output_line in output_lines:
                out_f.write(output_line)


class RawGenesReportWriter(ReportWriter):

    output_file_name = "raw_genes.txt"

    def __init__(self, output_folder: Path):
        self.output_folder = output_folder

    def get_formatted_lines(self, candidate: FixtureListChromosome) -> List[str]:

        lines = []

        for division_num in range(candidate.num_divisions):
            lines.append(f"Division {division_num}\n\n")

            for week_num in range(candidate.season_length_in_weeks):
                lines.append(f"\tWeek {week_num}\n")

                for game_num in range(candidate.max_games_per_week):

                    fixture = candidate.genes[division_num, week_num, game_num]

                    if fixture[0] == 0:
                        continue

                    lines.append(f"\t\t{fixture}\n")

                if week_num != candidate.season_length_in_weeks - 1:
                    lines.append(f"\t{'- '*20}\n")
                else:
                    lines.append("\n")

            lines.append(f"{'='*50}\n")

        return lines


class FormattedGenesReportWriter(ReportWriter):

    output_file_name = "formatted_genes.txt"

    def __init__(self, db: Database, output_folder: Path):
        self.db = db
        self.output_folder = output_folder

        self.max_team_name_length = max(
            len(team_name) for team_name in self.db.get_team_names()
        )

    def pad_team_name(self, team_name: str) -> str:
        return team_name.ljust(self.max_team_name_length)

    def get_formatted_lines(self, candidate: FixtureListChromosome) -> List[str]:

        lines = []

        for division_num in range(candidate.num_divisions):
            lines.append(f"Division {division_num}\n\n")

            for week_num in range(candidate.season_length_in_weeks):
                lines.append(f"\tWeek {week_num}\n")

                for game_num in range(candidate.max_games_per_week):

                    fixture = candidate.genes[division_num, week_num, game_num]

                    if fixture[0] == 0:
                        continue

                    home_team_name = self.pad_team_name(
                        self.db.get_team_name_by_team_num(fixture[0]).title()
                    )
                    away_team_name = self.pad_team_name(
                        self.db.get_team_name_by_team_num(fixture[1]).title()
                    )
                    ground_name = self.db.get_ground_name_by_ground_num(
                        fixture[2]
                    ).title()

                    lines.append(
                        f"\t\t{home_team_name} vs {away_team_name} at {ground_name}\n"
                    )

                if week_num != candidate.season_length_in_weeks - 1:
                    lines.append(f"\t{'- '*20}\n")
                else:
                    lines.append("\n")

            lines.append(f"{'='*50}\n")

        return lines


class FitnessReportReportWriter(ReportWriter):

    output_file_name = "fitness_report.txt"

    def __init__(self, fitness: FixtureListFitness, db: Database, output_folder: Path):

        self.fitness = fitness
        self.db = db
        self.output_folder = output_folder

    def get_formatted_lines(self, candidate: FixtureListChromosome) -> List[str]:

        lines = []

        lines.append(f"Candidate Fitness: {self.fitness(candidate)}\n")

        return lines


class FitnessDetailsReportWriter(ReportWriter):
    output_file_name = "fitness_details.txt"

    def __init__(self, fitness: FixtureListFitness, db: Database, output_folder: Path):

        self.fitness = fitness
        self.db = db
        self.output_folder = output_folder

    def get_formatted_lines(self, candidate: FixtureListChromosome) -> List[str]:

        lines = []

        report = self.fitness.report(candidate)

        for report_type, report_lines in report.items():
            lines.append(f"{report_type}\n\n")
            lines.extend(report_lines)
            lines.append("\n\n\n")
            lines.append("*" * 50)
            lines.append("*" * 50)
            lines.append("*" * 50)
            lines.append("\n\n\n")

        return lines

        # logging.info(
        #     "teams_playing_more_than_once_in_a_week=%d",
        #     self.fitness.teams_playing_more_than_once_in_a_week(candidate),
        # )
        # # logging.info(
        # #     "incorrect_number_of_fixtures_between_two_teams=%d",
        # #     self.fitness.incorrect_number_of_fixtures_between_two_teams(candidate),
        # # )
        # logging.info("ground_clashes=%d", self.fitness.ground_clashes(candidate))
        # logging.info(
        #     "teams_playing_eachother_consecutively=%f",
        #     self.fitness.teams_playing_eachother_consecutively(candidate),
        # )
        # logging.info(
        #     "teams_has_more_than_one_week_off=%f",
        #     self.fitness.teams_has_more_than_one_week_off(candidate),
        # )
        # logging.info(
        #     "ground_used_more_than_once_in_two_weeks=%f",
        #     self.fitness.ground_used_more_than_once_in_two_weeks(candidate),
        # )
        # logging.info(
        #     "teams_with_more_than_two_home_games_in_a_row=%f",
        #     self.fitness.teams_with_more_than_two_home_games_in_a_row(candidate),
        # # )
        # logging.info("time=%.2f", time_diff)
