from __future__ import annotations

import random
import time
from typing import List

from src.genetic import (
    AbsoluteFitness,
    ChromosomeGenerator,
    GeneSet,
    Mutation,
    Runner,
    StoppingCriteria,
)


class Chromosome:
    """Chromosome for the OneMax problem is a string of 1s and 0s."""

    def __init__(self, genes: List[str]):
        self.genes = genes

    def __repr__(self) -> str:
        return f"Chromosome({self.genes})"

    def __eq__(self, other: Chromosome) -> bool:
        return self.genes == other.genes


class GuessPasswordFitness(AbsoluteFitness):
    def __init__(self, target: str) -> None:
        self.target = target

    def __call__(self, chromosome: Chromosome) -> float:
        """Return a fitness score for the guess. The higher the score, the better the
        guess. The score is the sum of the number of characters in the guess that
        match the corresponding character in the target.

        Parameters
        ----------
        chromosome : Chromosome
            The guess to score

        Returns
        -------
        float
            The fitness score
        """

        return sum(
            1
            for expected, actual in zip(self.target, chromosome.genes)
            if expected == actual
        )


class GuessPasswordMutation(Mutation):
    def __init__(self, fitness: GuessPasswordFitness, gene_set: GeneSet):
        super().__init__(fitness, gene_set)

    def __call__(self, parent):
        index = random.randrange(0, len(parent.genes))
        child_genes = list(parent.genes)
        new_gene, alternate = random.sample(self.gene_set, 2)
        child_genes[index] = alternate if new_gene == child_genes[index] else new_gene
        genes = "".join(child_genes)
        return Chromosome(genes)


class GuessPasswordStoppingCriteria(StoppingCriteria):
    def __init__(self, target, fitness):
        self.target = target
        self.fitness = fitness

    @property
    def optimal_fitness(self) -> float:
        return len(self.target)

    def __call__(self, chromosome: Chromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        return self.fitness(chromosome) >= len(self.target)


class GuessPasswordChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, gene_set, length) -> None:
        self.gene_set = gene_set
        self.length = length

    def __call__(self) -> Chromosome:
        genes = []
        while len(genes) < self.length:
            sampleSize = min(self.length - len(genes), len(self.gene_set))
            genes.extend(random.sample(self.gene_set, sampleSize))
        genes = "".join(genes)
        return Chromosome(genes)


class GuessPasswordRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: GuessPasswordFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
    ):
        self.chromosome_generator = chromosome_generator
        self.fitness = fitness
        self.stopping_criteria = stopping_criteria
        self.mutate = mutate

    def display(self, candidate):
        timeDiff = time.time() - self.start_time
        print("{}\t{}\t{}".format(candidate.genes, self.fitness(candidate), timeDiff))


def guess_password(target, gene_set) -> Chromosome:

    fitness = GuessPasswordFitness(target)
    chromosome_generator = GuessPasswordChromosomeGenerator(gene_set, len(target))
    stopping_criteria = GuessPasswordStoppingCriteria(target, fitness)
    mutate = GuessPasswordMutation(fitness, gene_set)

    runner = GuessPasswordRunner(
        chromosome_generator, fitness, stopping_criteria, mutate
    )

    best = runner.run()
    return best


def main():

    target = "Hello World!"
    gene_set = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."

    guess_password(target, gene_set)


if __name__ == "__main__":
    main()
