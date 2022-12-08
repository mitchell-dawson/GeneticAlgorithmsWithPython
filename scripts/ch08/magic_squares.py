from __future__ import annotations

import logging
import random
import time
from copy import deepcopy

import numpy as np

from src.genetic import (
    AbsoluteFitness,
    AgeAnnealing,
    AgeAnnealingRunner,
    Chromosome,
    ChromosomeGenerator,
    Mutation,
    StoppingCriteria,
)


class MagicSquare(np.ndarray):
    """A magic square is a square matrix of numbers where the sum of the numbers in
    each row, column, and diagonal is the same. This class represents a magic square
    and provides methods to check if a matrix is a magic square and to calculate the
    fitness of a magic square.
    """

    def __new__(cls, input_array: np.ndarray) -> MagicSquare:
        """Create a new magic square."""
        return np.asarray(input_array).view(cls)

    def __str__(self) -> str:
        """return a visualisation of the chessboard"""
        return "\n" + "\n".join(
            ["".join([str(value).rjust(3) for value in row]) for row in self]
        )

    def __repr__(self) -> str:
        """return a visualisation of the chessboard"""
        return "\n" + "\n".join(
            ["".join([str(value).rjust(3) for value in row]) for row in self]
        )

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


class MagicSquaresChromosome(Chromosome):
    """Chromosome for the magic squares problem."""

    def __init__(self, genes: MagicSquare, age: int = 0) -> None:
        """Create a new chromosome.

        Parameters
        ----------
        genes : MagicSquare
            The genes of the chromosome.
        """
        self.genes = genes
        self.age = age

    def __str__(self) -> str:
        """Return a visualisation of the chromosome."""
        return str(self.genes)

    def __repr__(self) -> str:
        """Return a visualisation of the chromosome."""
        return str(self.genes)


class MagicSquaresFitness(AbsoluteFitness):
    """Fitness function of the magic squares problem."""

    chromosome: MagicSquaresChromosome

    def __init__(self) -> None:
        pass

    def __call__(self, chromosome: MagicSquaresChromosome) -> int:
        """Return the fitness of a chromosome. The fitness is the absolute difference
        between the sum of the rows, columns, and diagonals and the expected sum.

        Parameters
        ----------
        chromosome : MagicSquaresChromosome
            The chromosome to evaluate.

        Returns
        -------
        int
            The fitness of the chromosome.
        """

        self.chromosome = chromosome

        return -(
            self.row_fitness()
            + self.column_fitness()
            + self.minor_diagonal_fitness()
            + self.major_diagonal_fitness()
        )

    def row_fitness(self) -> int:
        """Return the fitness of the rows. The fitness is the absolute difference
        between the sum of the rows and the expected sum.

        Returns
        -------
        int
            The fitness of the rows.
        """
        return abs(
            self.chromosome.genes.sum_of_rows - self.chromosome.genes.expected_sum
        ).sum()

    def column_fitness(self) -> int:
        """Return the fitness of the columns. The fitness is the absolute difference
        between the sum of the columns and the expected sum.

        Returns
        -------
        int
            The fitness of the columns.
        """
        return abs(
            self.chromosome.genes.sum_of_columns - self.chromosome.genes.expected_sum
        ).sum()

    def minor_diagonal_fitness(self) -> int:
        """Return the fitness of the minor diagonal. The fitness is the absolute
        difference between the sum of the minor diagonal and the expected sum.

        Returns
        -------
        int
            The fitness of the minor diagonal.
        """
        return abs(
            self.chromosome.genes.sum_of_minor_diagonal
            - self.chromosome.genes.expected_sum
        )

    def major_diagonal_fitness(self) -> int:
        """Return the fitness of the major diagonal. The fitness is the absolute
        difference between the sum of the major diagonal and the expected sum.

        Returns
        -------
        int
            The fitness of the major diagonal.
        """
        return abs(
            self.chromosome.genes.sum_of_major_diagonal
            - self.chromosome.genes.expected_sum
        )


class MagicSquaresMutation(Mutation):
    """Mutation function for the magic squares problem."""

    def __init__(self, side_length: int):
        super().__init__()
        self.side_length = side_length

    def __call__(self, parent: MagicSquaresChromosome) -> MagicSquaresChromosome:
        """Mutate a chromosome by swapping two random elements. Pick two random
        indices, looping until the indices are distinct, and swap the elements at those
        indices.

        Parameters
        ----------
        parent : MagicSquaresChromosome
            The parent chromosome to mutate.

        Returns
        -------
        MagicSquaresChromosome
            The mutated chromosome.
        """

        child = deepcopy(parent)
        child = self.swap_two_genes(child)
        return child

    def swap_two_genes(self, child: MagicSquaresChromosome):
        """Swap two random numbers in the magic square"""

        idx_1 = (0, 0)
        idx_2 = (0, 0)

        while idx_1 == idx_2:
            idx_1 = (
                random.randint(0, self.side_length - 1),
                random.randint(0, self.side_length - 1),
            )

            idx_2 = (
                random.randint(0, self.side_length - 1),
                random.randint(0, self.side_length - 1),
            )

            child.genes[idx_1], child.genes[idx_2] = (
                child.genes[idx_2],
                child.genes[idx_1],
            )
        return child


class MagicSquaresStoppingCriteria(StoppingCriteria):
    """Stopping criteria for the magic squares problem."""

    def __init__(self, target: int, fitness: MagicSquaresFitness):
        self.target = target
        self.fitness = fitness

    def __call__(self, chromosome: MagicSquaresChromosome) -> bool:
        """Return true if can stop the genetic algorithm."""
        return self.fitness(chromosome) >= self.target


class MagicSquaresChromosomeGenerator(ChromosomeGenerator):
    """Generate a starting chromosome for the magic squares problem.
    The starting chromosome is a square matrix with the numbers 1 to n^2
    in order
    """

    def __init__(self, side_length: int) -> None:
        self.side_length = side_length

    def __call__(self) -> MagicSquaresChromosome:
        return MagicSquaresChromosome(
            MagicSquare(
                np.arange(1, (self.side_length**2) + 1).reshape(self.side_length, -1)
            )
        )


class MagicSquaresRunner(AgeAnnealingRunner):
    def __init__(
        self,
        chromosome_generator: MagicSquaresChromosomeGenerator,
        fitness: MagicSquaresFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
        age_annealing: AgeAnnealing,
    ):
        self.chromosome_generator = chromosome_generator
        self.fitness = fitness
        self.stopping_criteria = stopping_criteria
        self.mutate = mutate
        self.age_annealing = age_annealing

    def display(self, candidate):
        time_diff = time.time() - self.start_time

        logging.debug("candidate=%s", candidate)
        logging.debug("fitness=%s", self.fitness(candidate))
        logging.info("time=%f", time_diff)


def magic_squares(side_length: int, age_limit: int = 50) -> MagicSquaresChromosome:

    target = 0  # no difference between target and sum of the rows, columns, diagonals

    fitness = MagicSquaresFitness()
    chromosome_generator = MagicSquaresChromosomeGenerator(side_length)
    stopping_criteria = MagicSquaresStoppingCriteria(target, fitness)
    mutate = MagicSquaresMutation(side_length)
    age_annealing = AgeAnnealing(age_limit=age_limit)

    runner = MagicSquaresRunner(
        chromosome_generator,
        fitness,
        stopping_criteria,
        mutate,
        age_annealing=age_annealing,
    )

    best = runner.run()
    print(best)
    return best


def main():

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.DEBUG)

    side_length = 3
    age_limt = 50

    best = magic_squares(side_length, age_limit=age_limt)


if __name__ == "__main__":
    main()
