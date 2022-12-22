from __future__ import annotations

import logging
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Type, Union

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

    def max_quantity(self, max_weight: float, max_volume: float):
        """Return the maximum number of the item that can be added to the chromosome
        without exceeding the maximum weight or volume.

        Parameters
        ----------
        item : Resource
            The item to add to the chromosome
        max_weight : float
            The maximum weight allowed of the chromosome
        max_volume : float
            The maximum volume allowed of the chromosome

        Returns
        -------
        int
            The maximum number of the item that can be added to the chromosome
            without exceeding the maximum weight or volume
        """

        return min(
            int(max_weight / self.weight) if self.weight > 0 else sys.maxsize,
            int(max_volume / self.volume) if self.volume > 0 else sys.maxsize,
        )


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

    def remaining_weight(
        self,
        max_weight: float,
    ) -> float:
        """Calculate the remaining weight of the knapsack.

        Parameters
        ----------
        max_weight : float
            The maximum weight of the knapsack.

        Returns
        -------
        float
            The remaining weight of the knapsack.
        """

        return max_weight - self.total_weight

    def remaining_volume(
        self,
        max_volume: float,
    ) -> float:
        """Calculate the remaining volume of the knapsack.

        Parameters
        ----------
        max_volume : float
            The maximum volume of the knapsack.

        Returns
        -------
        float
            The remaining volume of the knapsack.
        """

        return max_volume - self.total_volume


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


class KnapsackMutation(Mutation):
    """Mutation function of the knapsack problem."""

    def __init__(
        self,
        gene_set: List[Resource],
        max_weight: float,
        max_volume: float,
    ):
        super().__init__(fitness=None, gene_set=gene_set)
        self.max_weight = max_weight
        self.max_volume = max_volume
        self.remaining_weight = max_weight
        self.remaining_volume = max_volume

    def __call__(self, parent: KnapsackChromosome):

        genes = deepcopy(parent.genes)

        if self.choose_to_remove_item(genes):
            genes = self.remove_random_item(genes)
            # We don’t immediately return when removing an item because we know
            # removing an item reduces the fitness.

        if self.choose_to_add_item(genes):
            genes = self.add_random_item(genes)
            return KnapsackChromosome(genes, parent.age)

        # If we get to this point, we have removed an item and not added a new item.
        # So now we will change the quantity of an existing item.
        self.change_item_quantity(genes)

        genes = sorted(genes, key=lambda x: x.resource.name)

        return KnapsackChromosome(genes, parent.age)

    def change_item_quantity(self, genes):
        idx = random.randrange(0, len(genes))
        resource_to_change = genes[idx].resource

        # add an amount between 1 and the maximum amount that can be added
        # without exceeding the maximum weight or volume.
        max_quantity = resource_to_change.max_quantity(
            KnapsackChromosome(genes).remaining_weight(self.max_weight),
            KnapsackChromosome(genes).remaining_volume(self.max_volume),
        )

        if max_quantity > 0:
            new_quantity = random.randint(1, max_quantity)
            genes[idx] = ResourceQuantity(resource_to_change, new_quantity)
        else:
            # if the maximum quantity is zero, then we remove the gene.
            del genes[idx]

    def choose_to_change_item(self, genes: List[ResourceQuantity]) -> bool:
        """Return true if a random item should be changed in the knapsack.

        Then we give the algorithm a chance to pick a different item type.

        Parameters
        ----------
        genes : List[ResourceQuantity]
            The list of resources in the knapsack.

        Returns
        -------
        bool
            True if a random item should be changed in the knapsack.
        """

        rng = random.randint(0, 4) == 0
        not_all_resources_added = len(genes) < len(self.gene_set)
        return not_all_resources_added and rng

    def add_random_item(self, genes):

        chromosome = KnapsackChromosome(genes)

        new_gene = add(
            genes,
            self.gene_set,
            chromosome.remaining_weight(self.max_weight),
            chromosome.remaining_volume(self.max_volume),
        )

        if new_gene is not None:
            genes.append(new_gene)

        return genes

    def choose_to_add_item(self, genes: List[ResourceQuantity]) -> bool:
        """Return true if a new item should be added to the knapsack.
        The conditions are that the knapsack is not full and that a random
        the knapsack is not full or that not all items have been added to the
        knapsack and that a random number is 0.

        We’ll always add if the length is zero and when there is weight or volume
        available. Otherwise, if we haven’t used all the item types, we’ll give the
        algorithm a small chance of adding another item type.

        Parameters
        ----------
        genes : List[ResourceQuantity]
            A list of resources and their quantities

        Returns
        -------
        bool
            True if a new item should be added to the knapsack.
        """

        chromosome = KnapsackChromosome(genes)

        weight_ok = chromosome.remaining_weight(self.max_weight) > 0
        volume_ok = chromosome.remaining_volume(self.max_volume) > 0

        knapsack_empty = len(genes) == 0
        not_all_resources_added = len(genes) < len(self.gene_set)
        rng = random.randint(0, 100) == 0

        return (weight_ok or volume_ok) and (
            knapsack_empty or (not_all_resources_added and rng)
        )

    @staticmethod
    def remove_random_item(
        genes: List[ResourceQuantity],
    ) -> List[ResourceQuantity]:
        """Remove a random item from the knapsack.

        Parameters
        ----------
        genes : List[ResourceQuantity]
            A list of resources and their quantities in the knapsack

        Returns
        -------
        List[ResourceQuantity]
            A list of resources and their quantities in the knapsack
        """

        idx = random.randrange(0, len(genes))
        del genes[idx]
        return genes

    @staticmethod
    def choose_to_remove_item(genes: List[ResourceQuantity]) -> bool:
        """Return true if the mutation should remove an item from the knapsack. We
        to remove an item from the knapsack if the knapsack is not empty and if
        the random number generator returns 0 (happens with probability = 0.1).

        We need to give it a small chance of removing an item from the knapsack. We’ll
        only do that if we have more than one item so that the knapsack is never empty.

        Parameters
        ----------
        genes : List[ResourceQuantity]
            A list of resources and their quantities in the knapsack

        Returns
        -------
        bool
            True if the mutation should remove an item from the knapsack
        """

        rng = random.randint(0, 10) == 0
        return (len(genes) > 1) and rng


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

            new_gene = add(genes, self.gene_set, remaining_weight, remaining_volume)

            if new_gene is not None:
                genes.append(new_gene)
                remaining_weight -= new_gene.quantity * new_gene.resource.weight
                remaining_volume -= new_gene.quantity * new_gene.resource.volume

        return KnapsackChromosome(genes, age=0)


class KnapsackRunner(Runner):
    def __init__(
        self,
        chromosome_generator: KnapsackChromosomeGenerator,
        fitness: Type[KnapsackFitness],
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
        logging.debug("total_weight=%f", candidate.total_weight)
        logging.debug("total_volume=%f", candidate.total_volume)
        logging.info("time=%f", time_diff)


def add(
    genes: List[ResourceQuantity],
    gene_set: List[Resource],
    max_weight: float,
    max_volume: float,
) -> Union[ResourceQuantity, None]:
    """Choose a random item from the gene set and add it to the chromosome. Add
    as many of the item as possible without exceeding the maximum weight or
    volume.

    When adding an item we’re going to exclude item types that are already in the
    knapsack from our options because we don’t want to have to sum multiple groups of
    a particular item type. Then we pick a random item and add as much of that item to
    the knapsack as we can.

    Parameters
    ----------
    genes : List[ResourceQuantity]
        A list of resources already in the chromosome
    max_weight : float
        The maximum weight allowed of the chromosome
    max_volume : float
        The maximum volume allowed of the chromosome

    Returns
    -------
    ResourceQuantity
        A resource quantity object for the new item
    """

    used_resources = {resource_quantity.resource for resource_quantity in genes}

    resource = random.choice(gene_set)

    while resource in used_resources:
        resource = random.choice(gene_set)

    max_quantity = resource.max_quantity(max_weight, max_volume)

    return ResourceQuantity(resource, max_quantity) if max_quantity > 0 else None


def fill_knapsack(
    gene_set: List[Resource],
    max_weight: float,
    max_volume: float,
    fitness_stagnation_limit: Union[float, int] = float("inf"),
    age_limit: float = float("inf"),
) -> KnapsackChromosome:

    fitness = KnapsackFitness
    chromosome_generator = KnapsackChromosomeGenerator(gene_set, max_weight, max_volume)
    stopping_criteria = KnapsackStoppingCriteria()
    mutation = KnapsackMutation(gene_set, max_weight, max_volume)
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
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)25s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    max_weight = 10
    max_volume = 4
    fitness_stagnation_limit = 1000

    gene_set = [
        Resource("Flour", 1680, 0.265, 0.41),  # 1
        Resource("Butter", 1440, 0.5, 0.13),  # 14
        Resource("Sugar", 1840, 0.441, 0.29),  # 6
    ]

    best = fill_knapsack(
        gene_set,
        max_weight,
        max_volume,
        fitness_stagnation_limit=fitness_stagnation_limit,
    )

    logging.info("best=%s", best)
    logging.info("total_weight=%f", best.total_weight)
    logging.info("total_volume=%f", best.total_volume)
    logging.info("total_value=%f", best.total_value)


if __name__ == "__main__":
    main()
