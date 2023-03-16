"""you should be able to see the queens problem as a variant of a scheduling or 
resource allocation problem, with the rows and columns representing resources, and 
the queens being people or things that require or consume resources, possibly with 
additional constraints (i.e. the diagonal attacks). With a different test class 
and genes you could solve a number of scheduling or resource allocation problems 
with the current engine. For example: baking pizzas in ovens, one pizza per oven. The 
rows could become ovens - possibly with special features like cooks-faster or 
-slower or only small pizzas. The columns become time slots, and the queens become 
pizzas with features like size, required cooking time, value, bonus for faster 
delivery, etc. Now you can try to maximize the number of pizzas, or the amount 
of money earned.
"""
from __future__ import annotations

import logging
import random
import time
from typing import List, Union

from src.genetic import (
    AbsoluteFitness,
    AgeAnnealing,
    Chromosome,
    ChromosomeGenerator,
    FitnessStagnationDetector,
    GeneSet,
    Mutation,
    Runner,
    StoppingCriteria,
)


class EightQueensChromosome(Chromosome):
    """Chromosome for the OneMax problem is a string of 1s and 0s."""

    def __init__(self, genes: List[int], age: int = 0):
        self.genes = genes
        self.age = age

    def __str__(self) -> str:
        return f"Chromosome({self.genes}, age={self.age})"

    def __repr__(self) -> str:
        return f"Chromosome({self.genes}, age={self.age})"

    def __eq__(self, other: EightQueensChromosome) -> bool:
        return self.genes == other.genes


class Board:
    def __init__(self, genes, size):
        board = [["."] * size for _ in range(size)]
        for index in range(0, len(genes), 2):
            row = genes[index]
            column = genes[index + 1]
            board[column][row] = "Q"
        self._board = board

    def get(self, row, column):
        return self._board[column][row]

    def print(self):
        for i in reversed(range(len(self._board))):
            print(" ".join(self._board[i]))


class EightQueensFitness(AbsoluteFitness):
    """Fitness function of a chromosome for the sort numbers problem."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, chromosome: EightQueensChromosome) -> float:
        """Return a fitness score for the queen positions. The score is the
        number of pairs of queens that are attacking each other.

        Parameters
        ----------
        chromosome : EightQueensChromosome
            The chromosome to evaluate

        Returns
        -------
        float
            a fitness score for the queen positions, equal to -(number of queens
            under attack). The higher the score, the better the fitness.
        """

        board = Board(chromosome.genes, self.size)

        rows_with_queens = set()
        cols_with_queens = set()
        north_east_diagonals_with_queens = set()
        south_east_diagonals_with_queens = set()

        for row in range(self.size):
            for col in range(self.size):
                if board.get(row, col) == "Q":
                    rows_with_queens.add(row)
                    cols_with_queens.add(col)
                    north_east_diagonals_with_queens.add(row + col)
                    south_east_diagonals_with_queens.add(self.size - 1 - row + col)

        total = -(
            self.size
            - len(rows_with_queens)
            + self.size
            - len(cols_with_queens)
            + self.size
            - len(north_east_diagonals_with_queens)
            + self.size
            - len(south_east_diagonals_with_queens)
        )

        return total


class EightQueensMutation(Mutation):
    def __init__(self, fitness: EightQueensFitness, gene_set: GeneSet):
        self.fitness = fitness
        self.gene_set = gene_set

    def __call__(self, parent: EightQueensChromosome) -> EightQueensChromosome:
        """Mutate a chromosome by swapping the value of a chromosome's gene with
        another random gene from the gene set. Mutation in this case is the
        changing of a row or column position of a single queen.

        Parameters
        ----------
        parent : EightQueensChromosome
            The chromosome to mutate.

        Returns
        -------
        EightQueensChromosome
            the mutated chromosome
        """

        child_genes = parent.genes.copy()
        index = random.randrange(0, len(parent.genes))
        new_gene, alternate = random.sample(self.gene_set, 2)
        child_genes[index] = alternate if new_gene == child_genes[index] else new_gene

        return EightQueensChromosome(child_genes)


class EightQueensStoppingCriteria(StoppingCriteria):
    def __init__(self, target: int, fitness: EightQueensFitness):
        self.target = target
        self.fitness = fitness

    def __call__(self, chromosome: EightQueensChromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        return self.fitness(chromosome) >= self.target


class EightQueensChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, gene_set, length) -> None:
        self.gene_set = gene_set
        self.length = length

    def __call__(self) -> EightQueensChromosome:
        """Generate a chromosome from the gene set

        In this task a chromosome is a list of numbers that represent the
        positions of the queens in the board. For a board of size nxn, the chromsome
        will have 2n genes of the form [row1, col1, row2, col2, ..., rown, coln]
        """

        genes = []
        while len(genes) < self.length:
            sampleSize = min(self.length - len(genes), len(self.gene_set))
            genes.extend(random.sample(self.gene_set, sampleSize))
        return EightQueensChromosome(genes)


class EightQueensRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: EightQueensFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
        age_annealing: AgeAnnealing,
        fitness_stagnation_detector: FitnessStagnationDetector,
        size: int,
    ):
        super().__init__(
            chromosome_generator,
            fitness,
            stopping_criteria,
            mutate,
            age_annealing,
            fitness_stagnation_detector,
        )
        self.size = size

    def display(self, candidate: EightQueensChromosome):
        time_diff = time.time() - self.start_time

        board = Board(candidate.genes, self.size)
        board.print()
        return f"time = {time_diff}"


def eight_queens(
    size: int,
    fitness_stagnation_limit: Union[float, int] = float("inf"),
    age_limit: float = float("inf"),
) -> EightQueensChromosome:
    """Run the genetic algorithm to find the solution for the eight queens problem.

    Parameters
    ----------
    size : int
        The size of the board (size x size)

    Returns
    -------
    EightQueensChromosome
        The chromosome with the solution. The genes of the chromosome are the
        positions of the queens in the board. For a board of size nxn, the chromosome
        will have 2n genes of the form [row1, col1, row2, col2, ..., rown, coln]
    """

    # the target is to have no queens under attack from eachother
    target = 0

    # the genes are the potential column and row positions of the queens
    gene_set = list(range(size))

    fitness = EightQueensFitness(size)
    chromosome_generator = EightQueensChromosomeGenerator(gene_set, 2 * size)
    stopping_criteria = EightQueensStoppingCriteria(target, fitness)
    mutate = EightQueensMutation(fitness, gene_set)
    age_annealing = AgeAnnealing(age_limit=age_limit)

    fitness_stagnation_detector = FitnessStagnationDetector(
        fitness, fitness_stagnation_limit
    )

    runner = EightQueensRunner(
        chromosome_generator,
        fitness,
        stopping_criteria,
        mutate,
        age_annealing,
        fitness_stagnation_detector,
        size,
    )

    best = runner.run()
    return best


def main():

    logging_format = (
        "[%(levelname)8s :%(filename)15s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.DEBUG)

    size = 8
    eight_queens(size)


if __name__ == "__main__":
    main()
