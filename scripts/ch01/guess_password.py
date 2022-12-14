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


class GuessPasswordChromosome(Chromosome):
    def __init__(self, genes: str, age: int = 0) -> None:
        self.genes = genes
        self.age = age

    def __str__(self) -> str:
        """Return a visualisation of the chromosome."""
        return str(self.genes)

    def __repr__(self) -> str:
        return f"{self.genes}"

    def __eq__(self, other: GuessPasswordChromosome) -> bool:
        return self.genes == other.genes


class GuessPasswordFitness(AbsoluteFitness):
    def __init__(self, target: str) -> None:
        self.target = target

    def __call__(self, chromosome: GuessPasswordChromosome) -> float:
        """Return a fitness score for the guess. The higher the score, the better the
        guess. The score is the sum of the number of characters in the guess that
        match the corresponding character in the target.

        Parameters
        ----------
        chromosome : GuessPasswordChromosome
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
        return GuessPasswordChromosome(genes)


class GuessPasswordStoppingCriteria(StoppingCriteria):
    def __init__(self, target, fitness):
        self.target = target
        self.fitness = fitness

    @property
    def optimal_fitness(self) -> float:
        return len(self.target)

    def __call__(self, chromosome: GuessPasswordChromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        return self.fitness(chromosome) >= len(self.target)


class GuessPasswordChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, gene_set, length) -> None:
        self.gene_set = gene_set
        self.length = length

    def __call__(self) -> GuessPasswordChromosome:
        genes = []
        while len(genes) < self.length:
            sampleSize = min(self.length - len(genes), len(self.gene_set))
            genes.extend(random.sample(self.gene_set, sampleSize))
        genes = "".join(genes)
        return GuessPasswordChromosome(genes)


class GuessPasswordRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: GuessPasswordFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
        age_annealing: AgeAnnealing,
        fitness_stagnation_detector: FitnessStagnationDetector,
    ):
        super().__init__(
            chromosome_generator=chromosome_generator,
            fitness=fitness,
            stopping_criteria=stopping_criteria,
            mutate=mutate,
            age_annealing=age_annealing,
            fitness_stagnation_detector=fitness_stagnation_detector,
        )

    def display(self, candidate):
        time_diff = time.time() - self.start_time
        logging.debug("genes=%s", candidate.genes)
        logging.debug("fitness=%s", self.fitness(candidate))
        logging.debug("time=%.5f", time_diff)


def guess_password(
    target,
    gene_set,
    fitness_stagnation_limit: Union[float, int] = float("inf"),
    age_limit: float = float("inf"),
) -> GuessPasswordChromosome:

    fitness = GuessPasswordFitness(target)
    chromosome_generator = GuessPasswordChromosomeGenerator(gene_set, len(target))
    stopping_criteria = GuessPasswordStoppingCriteria(target, fitness)
    mutate = GuessPasswordMutation(fitness, gene_set)
    age_annealing = AgeAnnealing(age_limit=age_limit)

    fitness_stagnation_detector = FitnessStagnationDetector(
        fitness, fitness_stagnation_limit
    )

    runner = GuessPasswordRunner(
        chromosome_generator,
        fitness,
        stopping_criteria,
        mutate,
        age_annealing,
        fitness_stagnation_detector,
    )

    best = runner.run()
    return best


def main():

    random.seed(0)

    logging_format = (
        "[%(levelname)8s :%(filename)25s:%(lineno)4s - %(funcName)30s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    target = "Hello World!"
    gene_set = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."

    guess_password(target, gene_set)


if __name__ == "__main__":
    main()
