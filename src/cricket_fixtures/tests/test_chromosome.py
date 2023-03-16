import numpy as np

from src.cricket_fixtures.chromosome import (
    Division,
    Fixture,
    FixtureListChromosome,
    Ground,
    Team,
)

#########################################################################
### Team ################################################################
#########################################################################


def test_init_Team():
    """GIVEN a team with a name, ground, division and team number
    WHEN the Team is initialized
    THEN the name, ground, division and team number should be set.
    """

    name = "Test Team"
    team_num = 1
    ground = Ground("Test Ground", 2)
    division = Division("Test Division", 3)
    team = Team(name, ground, division, team_num)
    assert team.name == name
    assert team.ground == ground
    assert team.division == division
    assert team.team_num == team_num


def test_equal_Team():
    """GIVEN two teams with the same team number
    WHEN the teams are compared
    THEN the teams should be equal, even if they have different names, grounds or
        divisions.
    """

    team_1_name = "Test Team 1"
    team_1_ground = Ground("Test Ground 1", 1)
    team_1_division = Division("Test Division 1", 1)
    team_1_team_num = 1
    team_1 = Team(team_1_name, team_1_ground, team_1_division, team_1_team_num)

    team_2_name = "Test Team 2"
    team_2_ground = Ground("Test Ground 2", 2)
    team_2_division = Division("Test Division 2", 2)
    team_2_team_num = 1
    team_2 = Team(team_2_name, team_2_ground, team_2_division, team_2_team_num)

    team_3_name = "Test Team 1"
    team_3_ground = Ground("Test Ground 1", 1)
    team_3_division = Division("Test Division 1", 1)
    team_3_team_num = 2
    team_3 = Team(team_3_name, team_3_ground, team_3_division, team_3_team_num)

    assert team_1 == team_2
    assert team_1 != team_3


#########################################################################
### Fixture #############################################################
#########################################################################


def test_init_Fixture():
    """GIVEN a Fixture
    WHEN the Fixture is initialized with a home team and an away team
    THEN the home team and away team should be set.
    """

    team_1_name = "Test Team 1"
    team_1_ground = Ground("Test Ground 1", 1)
    team_1_division = Division("Test Division 1", 1)
    team_1_team_num = 1
    team_1 = Team(team_1_name, team_1_ground, team_1_division, team_1_team_num)

    team_2_name = "Test Team 2"
    team_2_ground = Ground("Test Ground 2", 2)
    team_2_division = Division("Test Division 2", 2)
    team_2_team_num = 1
    team_2 = Team(team_2_name, team_2_ground, team_2_division, team_2_team_num)

    fixture = Fixture(home_team=team_1, away_team=team_2)

    assert fixture.home_team == team_1
    assert fixture.away_team == team_2
    assert fixture.ground == team_1.ground
    assert fixture.division == team_1.division


def test_has_same_teams_Fixture():
    """GIVEN a Fixture
    WHEN the has_same_teams method is called with a fixture
    THEN the method should return True if the teams involved in the fixtures are
        the same.
    """
    ground = Ground("Test Ground", 1)
    division = Division("Test Division", 1)

    team_1 = Team("Team 1", ground, division, 1)
    team_2 = Team("Team 2", ground, division, 2)
    team_3 = Team("Team 3", ground, division, 3)

    fixture1 = Fixture(home_team=team_1, away_team=team_2)
    fixture2 = Fixture(home_team=team_1, away_team=team_2)
    fixture3 = Fixture(home_team=team_2, away_team=team_1)
    fixture4 = Fixture(home_team=team_1, away_team=team_3)

    assert fixture1.has_same_teams(fixture2)
    assert fixture1.has_same_teams(fixture3)
    assert not fixture1.has_same_teams(fixture4)


#########################################################################
### FixtureList #########################################################
#########################################################################


def test_empty_FixtureListChromosome():
    """GIVEN a FixtureList initialized with an empty list
    WHEN the FixtureList is initialized
    THEN the FixtureList should contain zeros and be of the correct shape.
    """

    num_divisions = 3
    largest_division_size = 5
    season_length_in_weeks = 7

    # this is the most games that can be played in a week
    max_games_per_week_in_largest_league = largest_division_size // 2

    fixture_list = FixtureListChromosome.empty(
        num_divisions, season_length_in_weeks, max_games_per_week_in_largest_league
    )

    assert np.all(fixture_list.genes == 0)
    assert fixture_list.genes.shape == (
        num_divisions,
        season_length_in_weeks,
        max_games_per_week_in_largest_league,
        3,
    )


def test_attributes_FixtureListChromosome():
    """GIVEN a FixtureList with genes
    WHEN attributes of the chromosome are queried
    THEN the correct values are returned
    """
    num_divisions = 3
    largest_division_size = 5
    season_length_in_weeks = 7

    # this is the most games that can be played in a week
    max_games_per_week_in_largest_league = largest_division_size // 2

    fixture_list = FixtureListChromosome.empty(
        num_divisions, season_length_in_weeks, max_games_per_week_in_largest_league
    )

    assert fixture_list.num_divisions == num_divisions
    assert fixture_list.season_length_in_weeks == season_length_in_weeks
    assert fixture_list.max_games_per_week == max_games_per_week_in_largest_league


def test_false_is_fixture_set():
    """GIVEN a FixtureList
    WHEN no_fixture_set is called
    THEN the method should return false if there is no in the fixture list.
    """

    num_divisions = 3
    largest_division_size = 8
    season_length_in_weeks = 7

    # this is the most games that can be played in a week
    max_games_per_week_in_largest_league = largest_division_size // 2

    fixture_list = FixtureListChromosome.empty(
        num_divisions, season_length_in_weeks, max_games_per_week_in_largest_league
    )

    assert not fixture_list.is_fixture_set(0, 0, 0)
    assert not fixture_list.is_fixture_set(0, 2, 1)


def test_true_is_fixture_set():
    """GIVEN a FixtureList
    WHEN no_fixture_set is called
    THEN the method should return true if there is a fixture in the fixture list.
    """
    num_divisions = 4
    largest_division_size = 8
    season_length_in_weeks = 7

    # this is the most games that can be played in a week
    max_games_per_week_in_largest_league = largest_division_size // 2

    fixture_list = FixtureListChromosome.empty(
        num_divisions, season_length_in_weeks, max_games_per_week_in_largest_league
    )

    division_num = 3
    week_num = 0
    game_num = 1

    fixture_list.genes[division_num, week_num, game_num] = [1, 2, 1]
    assert fixture_list.is_fixture_set(division_num, week_num, game_num)


def test_add_fixture():
    """GIVEN a FixtureList
    WHEN add_fixture is called
    THEN the method should add the team ids of the fixture to the fixture list.
    """
    num_divisions = 4
    largest_division_size = 8
    season_length_in_weeks = 7

    # this is the most games that can be played in a week
    max_games_per_week_in_largest_league = largest_division_size // 2

    fixture_list = FixtureListChromosome.empty(
        num_divisions, season_length_in_weeks, max_games_per_week_in_largest_league
    )

    division_num = 3
    week_num = 0
    game_num = 1

    fixture = Fixture(
        home_team=Team(
            "home", Ground("home ground", 1), Division("test division", division_num), 3
        ),
        away_team=Team(
            "away", Ground("away ground", 1), Division("test division", division_num), 2
        ),
    )

    fixture_list.add_fixture(fixture, division_num, week_num, game_num)
    assert fixture_list.is_fixture_set(division_num, week_num, game_num)

    assert np.array_equal(
        fixture_list.genes[division_num, week_num, game_num, :],
        [
            fixture.home_team.team_num,
            fixture.away_team.team_num,
            fixture.ground.ground_num,
        ],
    )
