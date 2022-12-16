import numpy as np
import pytest

from scripts.ch08.magic_squares import (
    MagicSquare,
    MagicSquaresChromosome,
    MagicSquaresChromosomeGenerator,
    MagicSquaresFitness,
    MagicSquaresMutation,
    magic_squares,
)


@pytest.fixture(name="true_magic_square")
def fixture_true_magic_square() -> MagicSquare:
    """Create a magic square.

    Returns
    -------
    MagicSquare
        A magic square.

    """
    return MagicSquare(np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]]))


#########################################################################
### MagicSquare #########################################################
#########################################################################


def test_is_magic_true_MagicSquare(true_magic_square: MagicSquare):
    """GIVEN a magic square
    WHEN the is_magic method is called
    THEN the method should return True.
    """
    assert true_magic_square.is_magic()


def test_is_magic_false_duplicates_MagicSquare():
    """GIVEN a non-magic square with duplicates
    WHEN the is_magic method is called
    THEN the method should return False.
    """

    magic_square = MagicSquare(np.array([[8, 1, 6], [3, 5, 7], [4, 9, 3]]))
    assert not magic_square.is_magic()


def test_is_magic_false_bad_sum_MagicSquare():
    """GIVEN a non-magic square with a bad sum
    WHEN the is_magic method is called
    THEN the method should return False.
    """

    magic_square = MagicSquare(np.array([[8, 1, 6], [3, 5, 7], [4, 2, 9]]))
    assert not magic_square.is_magic()


def test_is_magic_false_non_square_MagicSquare():
    """GIVEN a non-magic square that is not square
    WHEN the is_magic method is called
    THEN the method should return False.
    """

    magic_square = MagicSquare(
        np.array([[8, 1, 6], [3, 5, 7], [4, 2, 9], [10, 11, 12]])
    )
    assert not magic_square.is_magic()


def test_is_magic_false_negative_number_MagicSquare():
    """GIVEN a non-magic square that contains a negative number
    WHEN the is_magic method is called
    THEN the method should return False.
    """

    magic_square = MagicSquare(np.array([[-1, 1, 6], [3, 5, 7], [4, 2, 0]]))
    assert not magic_square.is_magic()


def test_is_magic_false_MagicSquare():
    """GIVEN a non-magic square that contains a float
    WHEN the is_magic method is called
    THEN the method should return False.
    """

    magic_square = MagicSquare(np.array([[8, 1, 6], [3, 5, 7], [4, 9.0, 2]]))
    assert not magic_square.is_magic()


#########################################################################
### MagicSquaresFitness #################################################
#########################################################################


def test_call_true_MagicSquaresFitness(true_magic_square: MagicSquare):
    """GIVEN a magic square
    WHEN the MagicSquaresFitness class is called
    THEN the method should return 0.
    """

    fitness = MagicSquaresFitness()

    chromosome = MagicSquaresChromosome(genes=true_magic_square)

    assert fitness(chromosome) == 0


def test_call_on_diagonal_change_MagicSquaresFitness():
    """GIVEN a non-magic square with a change on the major diagonal
    WHEN the MagicSquaresFitness class is called
    THEN the method should return the expected value.
    """

    fitness = MagicSquaresFitness()

    chromosome = MagicSquaresChromosome(
        genes=MagicSquare((np.array([[9, 1, 6], [3, 5, 7], [4, 9, 2]])))
    )

    expected_row_fitness = -1
    expected_column_fitness = -1
    expected_major_diagonal_fitness = -1
    expected_minor_diagonal_fitness = 0

    assert fitness(chromosome) == (
        expected_row_fitness
        + expected_column_fitness
        + expected_major_diagonal_fitness
        + expected_minor_diagonal_fitness
    )


def test_call_off_diagonal_change_MagicSquaresFitness():
    """GIVEN a non-magic square with a change off the diagonal
    WHEN the MagicSquaresFitness class is called
    THEN the method should return the expected value.
    """

    fitness = MagicSquaresFitness()

    magic_square = MagicSquaresChromosome(
        genes=MagicSquare(np.array([[8, 2, 6], [3, 5, 7], [4, 9, 2]]))
    )

    expected_row_fitness = -1
    expected_column_fitness = -1
    expected_major_diagonal_fitness = 0
    expected_minor_diagonal_fitness = 0

    assert fitness(magic_square) == (
        expected_row_fitness
        + expected_column_fitness
        + expected_major_diagonal_fitness
        + expected_minor_diagonal_fitness
    )


#########################################################################
### MagicSquaresChromosomeGenerator #####################################
#########################################################################


@pytest.mark.parametrize(
    "side_length, expected",
    [
        (2, np.array([[1, 2], [3, 4]])),
        (3, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        (4, np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])),
    ],
)
def test_call_MagicSquaresChromosomeGenerator(side_length: int, expected: np.ndarray):

    chromosome_generator = MagicSquaresChromosomeGenerator(side_length)

    magic_square = chromosome_generator()
    assert np.array_equal(magic_square.genes, expected)


#########################################################################
### MagicSquaresMutation ################################################
#########################################################################


def test_call_MagicSquaresMutation():
    """GIVEN a magic square
    WHEN the MagicSquaresMutation class is called
    THEN the method should return a magic square.
    """

    mutation = MagicSquaresMutation(side_length=3)

    magic_square = MagicSquaresChromosome(
        MagicSquare(np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]]))
    )

    mutated_magic_square = mutation(magic_square)

    assert not np.array_equal(mutated_magic_square.genes, magic_square.genes)
    assert np.sum(np.array((mutated_magic_square.genes != magic_square.genes))) == 2


#########################################################################
### magic_squares #######################################################
#########################################################################


def test_magic_squares():
    side_length = 3
    best = magic_squares(side_length, age_limit=50)
    assert best.genes.is_magic()
