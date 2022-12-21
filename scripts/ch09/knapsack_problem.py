from __future__ import annotations

import logging
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union

import numpy as np

from src.genetic import (
    AgeAnnealing,
    Chromosome,
    ChromosomeGenerator,
    FitnessStagnationDetector,
    Mutation,
    RelativeFitness,
    Runner,
    StoppingCriteria,
)


@dataclass(eq=True, frozen=True)
class Resource:
    """A dataclass to represent a single resource in the knapsack problem"""

    name: str
    value: int
    weight: float
    volume: float


@dataclass
class ResourceQuantity:
    """A dataclass to represent the quantity of a resource in the knapsack problem"""

    resource: Resource
    quantity: int


class KnapsackChromosome(Chromosome):
    """Chromosome for the magic squares problem."""

    def __init__(self, genes: List[ResourceQuantity], age: int = 0) -> None:
        """Create a new chromosome.

        Parameters
        ----------
        genes : List[Resource]
            A list of resources
        age : int, optional
            The age of the chromosome, by default 0
        """
        self.genes = genes
        self.age = age

    def __len__(self) -> int:
        """Return the length of the chromosome."""
        return len(self.genes)

    def __getitem__(self, index: int):
        """Return the gene at the given index."""
        return self.genes[index]

    def __str__(self) -> str:
        """Return a visualisation of the chromosome."""
        return str(self.genes)

    def __repr__(self) -> str:
        """Return a visualisation of the chromosome."""
        return str(self.genes)

    @property
    def total_weight(self) -> float:
        """Return the total weight of the chromosome."""
        return sum(
            [
                resource_quantity.resource.weight * resource_quantity.quantity
                for resource_quantity in self.genes
            ]
        )

    @property
    def total_volume(self) -> float:
        """Return the total volume of the chromosome."""
        return sum(
            [
                resource_quantity.resource.volume * resource_quantity.quantity
                for resource_quantity in self.genes
            ]
        )

    @property
    def total_value(self) -> float:
        """Return the total value of the chromosome."""
        return sum(
            [
                resource_quantity.resource.value * resource_quantity.quantity
                for resource_quantity in self.genes
            ]
        )


class KnapsackFitness(RelativeFitness):
    """Fitness function of the magic knapsack problem."""

    def __init__(self, chromosome: KnapsackChromosome) -> None:
        self.chromosome = chromosome

    def __gt__(self, other: KnapsackFitness) -> bool:
        """Return true if the fitness of the current chromosome is better than
        the fitness of the other chromosome. In the knapsack problem the fitness
        is the total value of the resources in the knapsack. If the total value
        is the same then the fitness is the total weight of the resources in the
        knapsack. If the total weight is the same then the fitness is the total
        volume of the resources in the knapsack.
        """

        if self.chromosome.total_value != other.chromosome.total_value:
            return self.chromosome.total_value > other.chromosome.total_value

        if self.chromosome.total_weight != other.chromosome.total_weight:
            return self.chromosome.total_weight < other.chromosome.total_weight

        return self.chromosome.total_volume < other.chromosome.total_volume


class KnapsackStoppingCriteria(StoppingCriteria):
    """Stopping criteria for the magic squares problem."""

    def __call__(self, chromosome: KnapsackChromosome) -> bool:
        """Return true if can stop the genetic algorithm. In the knapsack problem
        the stopping criteria will be handled by the fitness stagnation detector
        """
        return False


class KnapsackChromosomeGenerator(ChromosomeGenerator):
    """Generate a starting chromosome for the magic squares problem.
    The starting chromosome is a square matrix with the numbers 1 to n^2
    in order
    """

    def __init__(
        self, gene_set: List[Resource], max_weight: float, max_volume: float
    ) -> None:
        self.gene_set = gene_set
        self.max_weight = max_weight
        self.max_volume = max_volume

    def __call__(self) -> KnapsackChromosome:

        genes = []
        remaining_weight, remaining_volume = self.max_weight, self.max_volume

        for _ in range(random.randrange(1, len(self.gene_set))):

            new_gene = self.add(
                genes, self.gene_set, remaining_weight, remaining_volume
            )

            if new_gene is not None:
                genes.append(new_gene)
                remaining_weight -= new_gene.quantity * new_gene.resource.weight
                remaining_volume -= new_gene.quantity * new_gene.resource.volume

        return KnapsackChromosome(genes, age=0)

    def add(
        self,
        genes: List[ResourceQuantity],
        gene_set: List[Resource],
        max_weight: float,
        max_volume: float,
    ):

        used_items = {resource_quantity.resource for resource_quantity in genes}

        item = random.choice(gene_set)

        while item in used_items:
            item = random.choice(gene_set)

        max_quantity = self.max_quantity(item, max_weight, max_volume)

        return ResourceQuantity(item, max_quantity) if max_quantity > 0 else None

    def max_quantity(self, item: Resource, max_weight: float, max_volume: float):

        return min(
            int(max_weight / item.weight) if item.weight > 0 else sys.maxsize,
            int(max_volume / item.volume) if item.volume > 0 else sys.maxsize,
        )


class KnapsackRunner(Runner):
    def __init__(
        self,
        chromosome_generator: KnapsackChromosomeGenerator,
        fitness: KnapsackFitness,
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
        time_diff = time.time() - self.start_time

        logging.debug("candidate=%s", candidate)
        logging.debug("fitness=%s", self.fitness(candidate))
        logging.info("time=%f", time_diff)


def fill_knapsack(
    items: List[Resource],
    fitness_stagnation_limit: Union[float, int] = float("inf"),
    age_limit: float = float("inf"),
) -> KnapsackChromosome:

    target = 0  # no difference between target and sum of the rows, columns, diagonals

    fitness = KnapsackFitness()
    chromosome_generator = KnapsackChromosomeGenerator(side_length)
    stopping_criteria = KnapsackStoppingCriteria(target, fitness)
    mutation = KnapsackMutation(side_length)
    age_annealing = AgeAnnealing(age_limit=age_limit)

    fitness_stagnation_detector = FitnessStagnationDetector(
        fitness, fitness_stagnation_limit
    )
    runner = KnapsackRunner(
        chromosome_generator,
        fitness,
        stopping_criteria,
        mutation,
        age_annealing,
        fitness_stagnation_detector,
    )

    best = runner.run()
    return best


def main():

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    max_weight = 10
    max_volume = 4
    age_limit = 50

    gene_set = [
        Resource("Flour", 1680, 0.265, 0.41),
        Resource("Butter", 1440, 0.5, 0.13),
        Resource("Sugar", 1840, 0.441, 0.29),
    ]

    max_weight = 10
    max_volume = 4

    # optimal = get_fitness(
    #     [
    #         ItemQuantity(items[0], 1),
    #         ItemQuantity(items[1], 14),
    #         ItemQuantity(items[2], 6),
    #     ]
    # )

    # self.fill_knapsack(items, maxWeight, maxVolume, optimal)

    # best = fill_knapsack(items, , age_limit=age_limit)


if __name__ == "__main__":
    main()
