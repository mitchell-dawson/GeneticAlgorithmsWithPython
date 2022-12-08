import numpy as np


class MagicSquare(np.ndarray):
    """A magic square is a square matrix of numbers where the sum of the numbers in
    each row, column, and diagonal is the same. This class represents a magic square
    and provides methods to check if a matrix is a magic square and to calculate the
    fitness of a magic square.
    """

    def __new__(cls, input_array: np.ndarray) -> "MagicSquare":
        """Create a new magic square from a numpy array."""
        obj = np.asarray(input_array).view(cls)
        return obj

    def is_magic(self) -> bool:
        """Check if the matrix is a magic square."""
        if self.shape[0] != self.shape[1]:
            return False

        if not np.issubdtype(self.dtype, np.integer):
            return False

        if np.any(self < 1):
            return False

        if self.contains_duplicates:
            return False

        if not np.all(self.sum_of_rows == self.expected_sum):
            return False

        if not np.all(self.sum_of_columns == self.expected_sum):
            return False

        if self.sum_of_major_diagonal != self.expected_sum:
            return False

        if self.sum_of_minor_diagonal != self.expected_sum:
            return False

        return True

    @property
    def side_length(self) -> int:
        """Calculate the side length of the matrix."""
        return self.shape[0]

    @property
    def expected_sum(self) -> int:
        """Calculate the expected sum of each the rows, columns, and diagonals."""
        return int(self.side_length * (self.side_length**2 + 1) / 2)

    @property
    def contains_duplicates(self) -> bool:
        """Check if the matrix contains duplicates."""
        return np.unique(self).size < self.size

    @property
    def sum_of_rows(self) -> np.ndarray:
        """Calculate the sum of the rows."""
        return np.array(np.sum(self, axis=1))

    @property
    def sum_of_columns(self) -> np.ndarray:
        """Calculate the sum of the columns."""
        return np.array(np.sum(self, axis=0))

    @property
    def sum_of_major_diagonal(self) -> np.ndarray:
        """Calculate the sum of the major diagonal."""
        return np.array(np.sum(np.diag(self)))

    @property
    def sum_of_minor_diagonal(self) -> np.ndarray:
        """Calculate the sum of the minor diagonal."""
        return np.array(np.sum(np.diag(np.fliplr(self))))
