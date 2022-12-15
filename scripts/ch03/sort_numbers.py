from __future__ import annotations

import random
import time
from typing import List, Type, Union

import numpy as np

from src.genetic import (
    AgeAnnealing,
    Chromosome,
    ChromosomeGenerator,
    FitnessStagnationDetector,
    GeneSet,
    Mutation,
    RelativeFitness,
    Runner,
    StoppingCriteria,
)


class SortNumbersChromosome(Chromosome):
    """Chromosome for the sort numbers problem is a list of numbers."""

    def __init__(self, genes: List[int], age: int = 0):
        self.genes = genes
        self.age = age

    def __str__(self) -> str:
        return f"Chromosome({self.genes}, age={self.age})"

    def __repr__(self) -> str:
        return f"Chromosome({self.genes}, age={self.age})"

    def __eq__(self, other: Chromosome) -> bool:
        return self.genes == other.genes


class SortNumbersFitness(RelativeFitness):
    """Fitness function of a chromosome for the sort numbers problem."""

    def __init__(self, chromosome: SortNumbersChromosome) -> None:
        self.chromosome = chromosome

    def __gt__(self, other: SortNumbersFitness) -> bool:

        if self.numbers_in_sequence_count() != other.numbers_in_sequence_count():
            return self.numbers_in_sequence_count() > other.numbers_in_sequence_count()

        return (
            self.total_size_of_gaps_in_sequence()
            < other.total_size_of_gaps_in_sequence()
        )

    def total_size_of_gaps_in_sequence(self) -> float:
        """Return the total size of the gaps in the sequence.

        Returns
        -------
        float
            The total size of the gaps in the sequence
        """

        return sum(
            [
                np.abs(self.chromosome.genes[ii] - self.chromosome.genes[ii - 1])
                for ii in range(1, len(self.chromosome.genes))
            ]
        )

    def numbers_in_sequence_count(self) -> float:
        """Return the number of numbers in sequence.

        Returns
        -------
        float
            The number of numbers in sequence
        """

        return len(
            [
                ii
                for ii in range(1, len(self.chromosome.genes))
                if self.chromosome.genes[ii] > self.chromosome.genes[ii - 1]
            ]
        )


class SortNumbersMutation(Mutation):
    def __init__(self, fitness: Type[SortNumbersFitness], gene_set: GeneSet):
        super().__init__(fitness, gene_set)

    def __call__(self, parent):
        child_genes = parent.genes.copy()
        index = random.randrange(0, len(parent.genes))
        new_gene, alternate = random.sample(self.gene_set, 2)
        child_genes[index] = alternate if new_gene == child_genes[index] else new_gene

        # a simple option to improve performance here is to set the part of the domain
        # that we know should be true to the correct value
        child_genes[0] = min(self.gene_set)
        child_genes[-1] = max(self.gene_set)

        return SortNumbersChromosome(child_genes)


class SortNumbersStoppingCriteria(StoppingCriteria):
    def __init__(self, target: int, fitness: Type[SortNumbersFitness]):
        self.target = target
        self.fitness = fitness

    def __call__(self, chromosome: SortNumbersChromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        return self.fitness(chromosome).numbers_in_sequence_count() >= self.target


class SortNumbersChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, gene_set, length) -> None:
        self.gene_set = gene_set
        self.length = length

    def __call__(self) -> SortNumbersChromosome:
        genes = []
        while len(genes) < self.length:
            sampleSize = min(self.length - len(genes), len(self.gene_set))
            genes.extend(random.sample(self.gene_set, sampleSize))
        return SortNumbersChromosome(genes)


class SortNumbersRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: Type[SortNumbersFitness],
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
        age_annealing: AgeAnnealing,
        fitness_stagnation_detector: FitnessStagnationDetector,
    ):
        super().__init__(
            chromosome_generator,
            fitness,
            stopping_criteria,
            mutate,
            age_annealing,
            fitness_stagnation_detector,
        )

    def display(self, candidate):
        timeDiff = time.time() - self.start_time

        numbers_in_sequence_count = self.fitness(candidate).numbers_in_sequence_count()
        total_size_of_gaps_in_sequence = self.fitness(
            candidate
        ).total_size_of_gaps_in_sequence()

        print(
            "{}:\tIn sequence={}\tGap size={}\ttime={}".format(
                candidate.genes,
                numbers_in_sequence_count,
                total_size_of_gaps_in_sequence,
                timeDiff,
            )
        )


def sort_numbers(
    length,
    fitness_stagnation_limit: Union[float, int] = float("inf"),
    age_limit: float = float("inf"),
) -> SortNumbersChromosome:

    target = length - 1

    gene_set = list(range(length))

    fitness = SortNumbersFitness
    chromosome_generator = SortNumbersChromosomeGenerator(gene_set, length)
    stopping_criteria = SortNumbersStoppingCriteria(target, fitness)
    mutate = SortNumbersMutation(fitness, gene_set)
    age_annealing = AgeAnnealing(age_limit=age_limit)

    fitness_stagnation_detector = FitnessStagnationDetector(
        fitness, fitness_stagnation_limit
    )

    runner = SortNumbersRunner(
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

    length = 10
    sort_numbers(length)


if __name__ == "__main__":
    main()
