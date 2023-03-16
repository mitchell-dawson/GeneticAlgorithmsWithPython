import random

import numpy as np

from src.cricket_fixtures.chromosome import FixtureListChromosome
from src.cricket_fixtures.mutation import (
    SwapUpToNFixturesMutation,
    TheNicolaSwitchProcesserMutation,
)


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


def get_hard_test_case():

    genes = np.array(
        [
            [
                [
                    [16, 10, 16],
                    [11, 15, 11],
                    [18, 19, 18],
                    [20, 13, 20],
                    [17, 9, 17],
                    [0, 0, 0],
                ],
                [
                    [19, 18, 19],
                    [20, 15, 20],
                    [10, 12, 10],
                    [12, 9, 12],
                    [16, 17, 16],
                    [14, 11, 14],
                ],
                [
                    [16, 11, 16],
                    [9, 17, 9],
                    [20, 16, 20],
                    [13, 19, 13],
                    [10, 15, 10],
                    [14, 18, 14],
                ],
                [
                    [16, 18, 16],
                    [10, 17, 10],
                    [20, 9, 20],
                    [15, 13, 15],
                    [11, 14, 11],
                    [12, 19, 12],
                ],
                [
                    [13, 18, 13],
                    [16, 15, 16],
                    [17, 14, 17],
                    [9, 18, 9],
                    [18, 10, 18],
                    [19, 20, 19],
                ],
                [
                    [11, 9, 11],
                    [10, 19, 10],
                    [12, 14, 12],
                    [16, 20, 16],
                    [15, 17, 15],
                    [18, 14, 18],
                ],
                [
                    [12, 11, 12],
                    [9, 13, 9],
                    [10, 16, 10],
                    [17, 15, 17],
                    [18, 11, 18],
                    [14, 20, 14],
                ],
                [
                    [10, 13, 10],
                    [16, 9, 16],
                    [20, 12, 20],
                    [17, 11, 17],
                    [15, 11, 15],
                    [14, 12, 14],
                ],
                [
                    [18, 13, 18],
                    [12, 15, 12],
                    [10, 14, 10],
                    [17, 16, 17],
                    [9, 20, 9],
                    [0, 0, 0],
                ],
                [
                    [15, 12, 15],
                    [19, 9, 19],
                    [14, 19, 14],
                    [18, 20, 18],
                    [16, 13, 16],
                    [17, 10, 17],
                ],
                [
                    [10, 20, 10],
                    [19, 12, 19],
                    [17, 18, 17],
                    [16, 14, 16],
                    [13, 11, 13],
                    [15, 19, 15],
                ],
                [
                    [10, 11, 10],
                    [18, 16, 18],
                    [13, 12, 13],
                    [17, 20, 17],
                    [19, 10, 19],
                    [9, 14, 9],
                ],
                [
                    [20, 17, 20],
                    [13, 15, 13],
                    [9, 11, 9],
                    [16, 12, 16],
                    [14, 10, 14],
                    [0, 0, 0],
                ],
                [
                    [14, 17, 14],
                    [9, 16, 9],
                    [15, 20, 15],
                    [12, 18, 12],
                    [11, 19, 11],
                    [13, 10, 13],
                ],
                [
                    [15, 18, 15],
                    [9, 10, 9],
                    [11, 12, 11],
                    [19, 14, 19],
                    [17, 13, 17],
                    [0, 0, 0],
                ],
                [
                    [11, 13, 11],
                    [15, 16, 15],
                    [9, 12, 9],
                    [14, 13, 14],
                    [20, 10, 20],
                    [17, 19, 17],
                ],
                [
                    [18, 15, 18],
                    [12, 16, 12],
                    [13, 14, 13],
                    [11, 17, 11],
                    [20, 19, 20],
                    [10, 9, 10],
                ],
                [
                    [10, 18, 10],
                    [12, 17, 12],
                    [14, 15, 14],
                    [19, 13, 19],
                    [20, 11, 20],
                    [16, 19, 16],
                ],
                [
                    [19, 11, 19],
                    [15, 14, 15],
                    [13, 9, 13],
                    [20, 18, 20],
                    [18, 12, 18],
                    [0, 0, 0],
                ],
                [
                    [12, 13, 12],
                    [19, 17, 19],
                    [20, 14, 20],
                    [11, 16, 11],
                    [18, 17, 18],
                    [15, 9, 15],
                ],
                [
                    [15, 10, 15],
                    [17, 12, 17],
                    [13, 20, 13],
                    [18, 9, 18],
                    [11, 18, 11],
                    [19, 16, 19],
                ],
                [
                    [14, 9, 14],
                    [11, 20, 11],
                    [13, 16, 13],
                    [19, 15, 19],
                    [12, 10, 12],
                    [0, 0, 0],
                ],
                [
                    [11, 10, 11],
                    [13, 17, 13],
                    [14, 16, 14],
                    [9, 19, 9],
                    [12, 20, 12],
                    [9, 15, 9],
                ],
            ]
        ]
    )

    assert genes.shape == (1, 23, 6, 3)
    return FixtureListChromosome(genes=genes)


def test_call_TheNicolaSwitchProcesserMutation():
    random.seed(1)

    parent = get_hard_test_case()

    mutation = TheNicolaSwitchProcesserMutation()

    child = mutation(parent)


def test_call_SwapUpToNFixturesMutation():
    random.seed(1)

    max_num_fixtures_to_swap = 2

    parent = get_three_team_one_division_fixture_list_chromosome()

    mutation = SwapUpToNFixturesMutation(max_num_fixtures_to_swap)

    child = mutation(parent)

    assert child.genes.shape == (1, 6, 1, 3)
    assert child.genes[0, 0, 0, 0] == 1

    assert np.array_equal(
        parent.genes, get_three_team_one_division_fixture_list_chromosome().genes
    )
    assert not np.array_equal(child.genes, parent.genes)

    assert np.array_equal(
        child.genes,
        np.array(
            [
                [
                    [[1, 3, 1]],
                    [[1, 2, 1]],
                    [[3, 2, 3]],
                    [[3, 1, 3]],
                    [[2, 1, 2]],
                    [[2, 3, 2]],
                ]
            ]
        ),
    )
