from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cricket_fixtures.chromosome import Division, Ground


class Database:
    def __init__(self, csv_file_path: Path) -> None:
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(self.csv_file_path)
        self._num_divisions = None
        self._division_nums = None

    def __len__(self) -> int:
        return len(self.df)

    def get_team_names(self) -> list[str]:
        return self.df["name"].values

    def get_team_num_by_name(self, name: str) -> int:
        return self.df[self.df.name == name]["team_num"].values[0]

    def get_team_name_by_team_num(self, team_num: int) -> str:
        return self.df[self.df.team_num == team_num]["name"].values[0]

    def get_ground_num_by_name(self, ground_name: str) -> int:
        return self.df[self.df.ground == ground_name]["ground_num"].values[0]

    def get_ground_name_by_ground_num(self, ground_num: int) -> str:
        return self.df[self.df.ground_num == ground_num]["ground"].values[0]

    def get_division_num_by_name(self, division_name: str) -> int:
        return self.df[self.df.division == division_name]["division_num"].values[0]

    def get_division_name_by_division_num(self, division_num: int) -> str:
        return self.df[self.df.division_num == division_num]["division"].values[0]

    def get_max_games_per_week_in_largest_division(self) -> int:
        max_teams_in_a_division = self.df.groupby("division_num").size().max()
        return max_teams_in_a_division // 2

    def get_ground_by_team_num(self, team_num: int) -> Ground:
        ground_name = self.df[self.df.team_num == team_num]["ground"].values[0]
        ground_num = self.df[self.df.team_num == team_num]["ground_num"].values[0]
        return Ground(ground_name, ground_num)

    def get_division_by_team_num(self, team_num: int) -> Division:
        division_name = self.df[self.df.team_num == team_num]["division"].values[0]
        division_num = self.df[self.df.team_num == team_num]["division_num"].values[0]
        return Division(division_name, division_num)

    def num_divisions(self) -> int:

        if not self._num_divisions:
            self._num_divisions = self.df.division_num.nunique()

        return self._num_divisions

    def get_ground_name_by_team_num(self, team_num: int) -> str:
        ground_name = self.df[self.df.team_num == team_num]["ground"].values[0]
        return ground_name

    def get_division_num_by_team_num(self, team_num: int) -> str:
        division_name = self.df[self.df.team_num == team_num]["division_num"].values[0]
        return division_name

    def get_division_name_by_team_num(self, team_num: int) -> str:
        division_num = self.df[self.df.team_num == team_num]["division"].values[0]
        return division_num

    def get_team_nums_in_division(self, division_num):
        return list(self.df[self.df.division_num == division_num]["team_num"])

    def division_nums(self):

        if self._division_nums is None:
            self._division_nums = self.df.division_num.unique()

        return self._division_nums
