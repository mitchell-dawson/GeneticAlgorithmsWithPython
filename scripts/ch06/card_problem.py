"""We start off with the set of cards, Ace and 2-10, which we separate into 2 groups of 
5 cards. The cards in each group have different constraints. The card values in one 
group must have a product of 360. The card values in the other group must sum to 36. 
We can only use each card once.
"""


from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, List, Tuple, Type

import numpy as np

from src.genetic import (
    ChromosomeGenerator,
    GeneSet,
    Mutation,
    RelativeFitness,
    Runner,
    StoppingCriteria,
)


class Chromosome(List[int]):
    """Chromosome for the card problem."""

    @property
    def left_group(self) -> List[int]:
        return self[:5]

    @property
    def right_group(self) -> List[int]:
        return self[5:]

    @property
    def num_duplicates(self) -> int:
        return len(self) - len(set(self))


class CardProblemFitness(RelativeFitness):
    """Fitness function of a chromosome for the card problem. The fitness is the sum of
    the absolute difference between the sum of the cards in the first group and 36 and
    the absolute difference between the product of the cards in the second group and
    360.
    """

    def __init__(self, chromosome: Chromosome) -> None:
        self.chromosome = chromosome

    def __gt__(self, other: CardProblemFitness) -> bool:

        if self.count_duplicates() != other.count_duplicates():
            return self.count_duplicates() < self.count_duplicates()

        return self.total_difference() < other.total_difference()

    def total_difference(self) -> int:
        return self.sum_of_cards_fitness() + self.product_of_cards_fitness()

    def count_duplicates(self) -> int:
        """Return the number of duplicates in a list of cards.

        Parameters
        ----------
        cards : List[int]
            The cards to evaluate.

        Returns
        -------
        int
            The number of duplicates in the list of cards.
        """
        return len(self.chromosome) - len(set(self.chromosome))

    def sum_of_cards_fitness(self, target: int = 36) -> int:
        """Return the fitness of some cards from the chromosome. The fitness is the
        absolute difference between the sum of the cards and the target.

        Parameters
        ----------
        target : int, optional
            The target value, by default 36

        Returns
        -------
        int
            The fitness of the cards.
        """
        return np.abs(np.sum(self.chromosome.left_group) - target)

    def product_of_cards_fitness(self, target: int = 360) -> int:
        """Return the fitness of some cards from the chromosome. The fitness is the
        absolute difference between the product of the cards and the target.

        Parameters
        ----------
        target : int, optional
            the target value, by default 360

        Returns
        -------
        int
            the fitness of the cards.
        """
        return np.abs(np.prod(self.chromosome.right_group) - target)


class CardProblemMutation(Mutation):
    """Mutation function for the graph colouring problem."""

    def __init__(self, fitness: CardProblemFitness, gene_set: GeneSet):
        super().__init__(fitness, gene_set)

    def __call__(self, parent: Chromosome) -> Chromosome:

        child = parent.copy()

        if len(child) == len(set(child)):
            count = random.randint(1, 4)
            while count > 0:
                count -= 1
                indexA, indexB = random.sample(range(len(child)), 2)
                child[indexA], child[indexB] = child[indexB], child[indexA]
        else:
            indexA = random.randrange(0, len(child))
            indexB = random.randrange(0, len(self.gene_set))
            child[indexA] = child[indexB]

        return Chromosome(child)


class CardProblemStoppingCriteria(StoppingCriteria):
    def __init__(self, target: int, fitness: Type[CardProblemFitness]):
        self.target = target
        self.fitness = fitness

    def __call__(self, chromosome: Chromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        return (self.fitness(chromosome).total_difference() == 0) and (
            self.fitness(chromosome).count_duplicates() == 0
        )


class CardProblemChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, gene_set) -> None:
        self.gene_set = gene_set

    def __call__(self) -> Chromosome:
        return Chromosome(self.gene_set)


class CardProblemRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: Type[CardProblemFitness],
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
    ):
        self.chromosome_generator = chromosome_generator
        self.fitness = fitness
        self.stopping_criteria = stopping_criteria
        self.mutate = mutate

    def display(self, candidate):
        time_diff = time.time() - self.start_time

        logging.debug("candidate=%s", candidate)

        logging.debug("left sum=%s", np.sum(candidate.left_group))
        logging.debug("right product=%s", np.prod(candidate.right_group))
        logging.debug("duplicates=%s", candidate.num_duplicates)

        logging.debug("time=%f", time_diff)


def card_problem(gene_set: Tuple[int]) -> Chromosome:

    target = 0  # no duplicates, sum of left cards = 36, product of right cards = 360

    fitness = CardProblemFitness
    chromosome_generator = CardProblemChromosomeGenerator(gene_set)
    stopping_criteria = CardProblemStoppingCriteria(target, fitness)
    mutate = CardProblemMutation(fitness, gene_set)

    runner = CardProblemRunner(chromosome_generator, fitness, stopping_criteria, mutate)

    best = runner.run()
    print(best)
    return best


def main():

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    gene_set = list(range(1, 11))

    card_problem(gene_set)


if __name__ == "__main__":
    main()
