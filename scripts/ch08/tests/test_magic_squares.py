import numpy as np
import pytest

from scripts.ch08.magic_squares import MagicSquare


@pytest.fixture(name="true_magic_square")
def fixture_true_magic_square() -> MagicSquare:
    """Create a magic square.

    Returns
    -------
    MagicSquare
        A magic square.

    """
    return MagicSquare(np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]]))


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
