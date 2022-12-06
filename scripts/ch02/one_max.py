from __future__ import annotations

import logging
import random
import time

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

    def __init__(self, genes: str):
        self.genes = genes

    def __repr__(self) -> str:
        return f"Chromosome({self.genes})"

    def __eq__(self, other: Chromosome) -> bool:
        return self.genes == other.genes


class OneMaxFitness(AbsoluteFitness):
    def __call__(self, chromosome: Chromosome) -> float:
        logging.debug("Calculating fitness for %s", chromosome)
        fitness = chromosome.genes.count("1")
        logging.debug("%s has fitness %.1f", chromosome, fitness)
        return fitness


class OneMaxMutation(Mutation):
    def __init__(self, fitness: AbsoluteFitness, gene_set: GeneSet):
        super().__init__(fitness, gene_set)

    def __call__(self, parent):

        logging.debug("- " * 30)
        logging.debug("Mutating parent: %s", parent)

        index = random.randrange(0, len(parent.genes))
        logging.debug("Index to mutate: %s", index)

        child_genes = list(parent.genes)
        new_gene, alternate = random.sample(self.gene_set, 2)
        logging.debug("New_gene: %d, alternate: %d", new_gene, alternate)

        child_genes[index] = alternate if new_gene == child_genes[index] else new_gene
        genes = "".join([str(X) for X in child_genes])
        chromosome = Chromosome(genes)
        logging.debug("Mutated child created: %s", chromosome)

        logging.debug("Mutation complete")
        logging.debug("- " * 30)

        return chromosome


class OneMaxStoppingCriteria(StoppingCriteria):
    def __init__(self, target: int, fitness: AbsoluteFitness):
        self.target = target
        self.fitness = fitness

    @property
    def optimal_fitness(self) -> float:
        return self.target

    def __call__(self, chromosome: Chromosome) -> bool:
        return self.fitness(chromosome) >= self.target


class OneMaxChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, gene_set, length) -> None:
        self.gene_set = gene_set
        self.length = length

    def __call__(self) -> Chromosome:
        genes = []
        while len(genes) < self.length:
            sampleSize = min(self.length - len(genes), len(self.gene_set))
            genes.extend(random.sample(self.gene_set, sampleSize))
        genes = "".join([str(X) for X in genes])
        return Chromosome(genes)


class OneMaxRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: AbsoluteFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
    ):
        self.chromosome_generator = chromosome_generator
        self.fitness = fitness
        self.stopping_criteria = stopping_criteria
        self.mutate = mutate

    def display(self, candidate: Chromosome):
        time_diff = time.time() - self.start_time
        return f"time = {time_diff}"


def one_max(target, gene_set):

    fitness = OneMaxFitness()
    chromosome_generator = OneMaxChromosomeGenerator(gene_set, target)
    stopping_criteria = OneMaxStoppingCriteria(target, fitness)
    mutation = OneMaxMutation(fitness, gene_set)

    runner = OneMaxRunner(
        chromosome_generator,
        fitness,
        stopping_criteria,
        mutation,
    )

    best = runner.run()
    return best


def main():

    logging_format = (
        "[%(levelname)8s :%(filename)15s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.DEBUG)

    target = 30
    gene_set = [0, 1]
    one_max(target, gene_set)


if __name__ == "__main__":
    main()
