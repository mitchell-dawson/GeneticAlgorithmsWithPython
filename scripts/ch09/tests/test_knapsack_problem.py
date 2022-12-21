import random
from typing import List

import numpy as np
import pytest

from scripts.ch09.knapsack_problem import (
    KnapsackChromosome,
    KnapsackChromosomeGenerator,
    KnapsackFitness,
    KnapsackStoppingCriteria,
    Resource,
    ResourceQuantity,
)


@pytest.fixture(name="resources")
def fixture_resources() -> List[Resource]:
    """Create a list of resources.

    Returns
    -------
    List[Resource]
        A list of resources.

    """
    return [
        Resource("Flour", 1680, 0.265, 0.41),
        Resource("Butter", 1440, 0.5, 0.13),
        Resource("Sugar", 1840, 0.441, 0.29),
    ]


@pytest.fixture(name="resource_quantities")
def fixture_resource_quantities(resources) -> List[ResourceQuantity]:
    """Create a list of resource quantities. The quantity of each resource is
    1,2,3.

    Parameters
    ----------
    resources : List[Resource]
        A list of resources.

    Returns
    -------
    List[ResourceQuantity]
        A list of resource quantities.

    """
    return [
        ResourceQuantity(resource, amount + 1)
        for amount, resource in enumerate(resources)
    ]


def test_init_Resource():

    name = "Flour"
    value = 1680
    weight = 0.265
    volume = 0.41

    resource = Resource(name, value, weight, volume)

    assert resource.name == name
    assert resource.value == value
    assert resource.weight == weight
    assert resource.volume == volume


def test_init_ResourceQuantity():
    """GIVEN a resource
    WHEN a ResourceQuantity is created
    THEN the resource and quantity are set
    """

    name = "Flour"
    value = 1680
    weight = 0.265
    volume = 0.41

    resource = Resource(name, value, weight, volume)
    quantity = 3

    resource_quantity = ResourceQuantity(resource, quantity)

    assert resource_quantity.resource == resource
    assert resource_quantity.quantity == quantity


def test_init_KnapsackChromosome(resource_quantities):
    """GIVEN a list of resource quantities
    WHEN a KnapsackChromosome is created
    THEN the genes and age are set
    """

    genes = resource_quantities
    age = 3

    chromosome = KnapsackChromosome(genes, age)

    assert chromosome.genes == genes
    assert chromosome.age == age


def test_len_KnapsackChromosome(resource_quantities):
    """GIVEN a list of resource quantities
    WHEN the length of the chromosome is calculated
    THEN the correct length is returned
    """

    chromosome = KnapsackChromosome(resource_quantities)
    assert len(chromosome) == len(resource_quantities)


def test_getitem_KnapsackChromosome(resource_quantities):
    """GIVEN a list of resource quantities
    WHEN a gene is retrieved from the chromosome
    THEN the correct gene is returned
    """

    chromosome = KnapsackChromosome(resource_quantities)
    assert chromosome[0] == resource_quantities[0]
    assert chromosome[1] == resource_quantities[1]
    assert chromosome[2] == resource_quantities[2]


def test_total_weight_KnapsackChromosome(resource_quantities):
    """GIVEN a list of resource_quantities
    WHEN the total weight of the chromosome is calculated
    THEN the correct total weight is returned
    """
    chromosome = KnapsackChromosome(resource_quantities)
    assert chromosome.total_weight == 1 * 0.265 + 2 * 0.5 + 3 * 0.441


def test_total_value_KnapsackChromosome(resource_quantities):
    """GIVEN a list of resource_quantities
    WHEN the total value of the chromosome is calculated
    THEN the correct total value is returned
    """

    chromosome = KnapsackChromosome(resource_quantities)
    assert chromosome.total_value == 1 * 1680 + 2 * 1440 + 3 * 1840


def test_total_volume_KnapsackChromosome(resource_quantities):
    """GIVEN a list of resource_quantities
    WHEN the total volume of the chromosome is calculated
    THEN the correct total volume is returned
    """
    chromosome = KnapsackChromosome(resource_quantities)
    assert chromosome.total_volume == 1 * 0.41 + 2 * 0.13 + 3 * 0.29


def test_different_total_value_KnapsackFitness():
    """GIVEN two chromosomes with different total values
    WHEN the fitness of the chromosomes is compared
    THEN the chromosome with the higher total value is greater
    """

    chromosome_1 = KnapsackChromosome(
        [
            ResourceQuantity(Resource("Flour", 1680, 0.265, 0.41), 1),
            ResourceQuantity(Resource("Butter", 1440, 0.5, 0.13), 2),
            ResourceQuantity(Resource("Sugar", 1840, 0.441, 0.29), 3),
        ]
    )

    # note value for flour is higher in chromosome_2
    chromosome_2 = KnapsackChromosome(
        [
            ResourceQuantity(Resource("Flour", 1700, 0.265, 0.41), 1),
            ResourceQuantity(Resource("Butter", 1440, 0.5, 0.13), 2),
            ResourceQuantity(Resource("Sugar", 1840, 0.441, 0.29), 3),
        ]
    )

    assert KnapsackFitness(chromosome_2) > KnapsackFitness(chromosome_1)


def test_different_total_weight_KnapsackFitness():
    """GIVEN two chromosomes with different total weights
    WHEN the fitness of the chromosomes is compared
    THEN the chromosome with the lower total weight is greater
    """

    chromosome_1 = KnapsackChromosome(
        [
            ResourceQuantity(Resource("Flour", 1680, 0.265, 0.41), 1),
            ResourceQuantity(Resource("Butter", 1440, 0.5, 0.13), 2),
            ResourceQuantity(Resource("Sugar", 1840, 0.441, 0.29), 3),
        ]
    )

    # note weight for flour is lower in chromosome_2
    chromosome_2 = KnapsackChromosome(
        [
            ResourceQuantity(Resource("Flour", 1680, 0.25, 0.41), 1),
            ResourceQuantity(Resource("Butter", 1440, 0.5, 0.13), 2),
            ResourceQuantity(Resource("Sugar", 1840, 0.441, 0.29), 3),
        ]
    )

    assert KnapsackFitness(chromosome_2) > KnapsackFitness(chromosome_1)


def test_different_total_volume_KnapsackFitness():
    """GIVEN two chromosomes with different total volumes
    WHEN the fitness of the chromosomes is compared
    THEN the chromosome with the lower total volume is greater
    """

    chromosome_1 = KnapsackChromosome(
        [
            ResourceQuantity(Resource("Flour", 1680, 0.265, 0.41), 1),
            ResourceQuantity(Resource("Butter", 1440, 0.5, 0.13), 2),
            ResourceQuantity(Resource("Sugar", 1840, 0.441, 0.29), 3),
        ]
    )

    # note volume for flour is lower in chromosome_2
    chromosome_2 = KnapsackChromosome(
        [
            ResourceQuantity(Resource("Flour", 1680, 0.265, 0.4), 1),
            ResourceQuantity(Resource("Butter", 1440, 0.5, 0.13), 2),
            ResourceQuantity(Resource("Sugar", 1840, 0.441, 0.29), 3),
        ]
    )

    assert KnapsackFitness(chromosome_2) > KnapsackFitness(chromosome_1)


def test_call_KnapsackStoppingCriteria(resources):
    """GIVEN a KnapsackStoppingCriteria
    WHEN the knapack problem stopping criteria is called
    THEN a False value is returned
    """

    stopping_criteria = KnapsackStoppingCriteria()

    for resource in resources:
        assert stopping_criteria(resource) is False


def test_init_KnapsackChromosomeGenerator(resources):
    """GIVEN a KnapsackChromosomeGenerator
    WHEN a KnapsackChromosomeGenerator is created
    THEN the max_weight, max_volume are set
    """
    max_weight = 50
    max_volume = 100

    chromosome_generator = KnapsackChromosomeGenerator(
        resources, max_weight, max_volume
    )

    assert chromosome_generator.gene_set == resources
    assert chromosome_generator.max_weight == max_weight
    assert chromosome_generator.max_volume == max_volume


def test_call_KnapsackChromosomeGenerator(resources):
    """GIVEN a KnapsackChromosomeGenerator
    WHEN the knapack problem chromosome generator is called
    THEN a KnapsackChromosome is returned
    """
    random.seed(1)

    max_weight = 50
    max_volume = 100

    chromosome_generator = KnapsackChromosomeGenerator(
        resources, max_weight, max_volume
    )

    chromosome = chromosome_generator()

    assert isinstance(chromosome, KnapsackChromosome)
    assert len(chromosome) == 1
    assert chromosome[0].resource.name == "Sugar"
    assert np.isclose(chromosome.total_weight, 49.833)
    assert np.isclose(chromosome.total_volume, 32.77)
    assert chromosome.total_value == 207920


def test_add_KnapsackChromosomeGenerator(resources):
    """GIVEN a KnapsackChromosomeGenerator
    WHEN the add method is called
    THEN a resource quantity is returned
    """
    random.seed(1)

    max_weight = 50
    max_volume = 100

    chromosome_generator = KnapsackChromosomeGenerator(
        resources, max_weight, max_volume
    )

    chromosome_generator.add()

    assert False


def test_max_quantity_high_volume_KnapsackChromosomeGenerator(resources):
    """GIVEN a KnapsackChromosomeGenerator
    WHEN the max_quantity method is called on a resource with a high volume
    THEN a valid quantity is returned
    """
    random.seed(1)

    max_weight = 50
    max_volume = 100

    chromosome_generator = KnapsackChromosomeGenerator(
        resources, max_weight, max_volume
    )

    chromosome_generator.max_quantity()

    assert False


def test_max_quantity_high_weight_KnapsackChromosomeGenerator(resources):
    """GIVEN a KnapsackChromosomeGenerator
    WHEN the max_quantity method is called on a resource with a high weight
    THEN a valid quantity is returned
    """
    random.seed(1)

    max_weight = 50
    max_volume = 100

    chromosome_generator = KnapsackChromosomeGenerator(
        resources, max_weight, max_volume
    )

    chromosome_generator.max_quantity()

    assert False


def test_cookies_fill_knapsack(self):
    # items = [
    #     Resource("Flour", 1680, 0.265, 0.41),
    #     Resource("Butter", 1440, 0.5, 0.13),
    #     Resource("Sugar", 1840, 0.441, 0.29),
    # ]
    # maxWeight = 10
    # maxVolume = 4
    # optimal = get_fitness(
    #     [
    #         ItemQuantity(items[0], 1),
    #         ItemQuantity(items[1], 14),
    #         ItemQuantity(items[2], 6),
    #     ]
    # )
    # self.fill_knapsack(items, maxWeight, maxVolume, optimal)
    assert False
