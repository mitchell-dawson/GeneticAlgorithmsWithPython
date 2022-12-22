import random
from typing import List

import numpy as np
import pytest

from scripts.ch09.knapsack_problem import (
    KnapsackChromosome,
    KnapsackChromosomeGenerator,
    KnapsackFitness,
    KnapsackMutation,
    KnapsackStoppingCriteria,
    Resource,
    ResourceQuantity,
    add,
    fill_knapsack,
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


def test_max_quantity_Resource(resources):
    """GIVEN a resource
    WHEN the max_quantity method is called with a max weight and max volume
    THEN the maximum quantity of the resource that fits within the max weight and
        max volume constraints is returned
    """
    random.seed(1)

    max_weight = 50
    max_volume = 100

    for resource in resources:

        max_quantity = resource.max_quantity(max_weight, max_volume)

        assert resource.weight * max_quantity <= max_weight
        assert resource.volume * max_quantity <= max_volume

        assert (resource.weight * (max_quantity + 1) > max_weight) or (
            resource.volume * (max_quantity + 1) > max_volume
        )


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


def test_add(resources):
    """GIVEN a list of genes and a list of resources
    WHEN the add method is called
    THEN a resource quantity is returned that fits within the max weight and max
        volume constraints
    """
    random.seed(1)

    max_weight = 50
    max_volume = 100

    genes = []

    resource_quantity = add(genes, resources, max_weight, max_volume)

    # the resource should fit within the max weight and max volume constraints
    total_weight = resource_quantity.resource.weight * resource_quantity.quantity
    total_volume = resource_quantity.resource.volume * resource_quantity.quantity

    assert total_weight <= max_weight
    assert total_volume <= max_volume

    # show that adding one more of the same resource would exceed the max weight or
    # max volume
    total_weight_with_one_more = resource_quantity.resource.weight * (
        resource_quantity.quantity + 1
    )

    total_volume_with_one_more = resource_quantity.resource.volume * (
        resource_quantity.quantity + 1
    )

    assert (
        total_weight_with_one_more > max_weight
        or total_volume_with_one_more > max_volume
    )


def test_choose_to_remove_item_empty_KnapsackMutation():
    """GIVEN a KnapsackMutation
    WHEN the choose_to_remove_item method is called on an empty list
    THEN False is returned as there is nothing to remove
    """

    genes = []
    assert not KnapsackMutation.choose_to_remove_item(genes)


def test_choose_to_remove_item_single_item_KnapsackMutation():
    """GIVEN a KnapsackMutation
    WHEN the choose_to_remove_item method is called on list of a single item
    THEN False is returned as then there would be nothing in the knapsack
    """

    genes = [ResourceQuantity(Resource("Flour", 1680, 0.265, 0.41), 1)]
    assert not KnapsackMutation.choose_to_remove_item(genes)


def test_choose_to_remove_item_KnapsackMutation(resource_quantities):
    """GIVEN a KnapsackMutation
    WHEN the choose_to_remove_item method is called on a list of genes
    THEN True is returned as there is something to remove
    """

    random.seed(1)

    decisions_to_remove = [
        KnapsackMutation.choose_to_remove_item(resource_quantities)
        for ii in range(10000)
    ]

    # roughly 10% of the time, an item should be removed
    assert sum(decisions_to_remove) == 914


def test_init_KnapsackMutation(resources):

    max_weight = 50
    max_volume = 100

    mutation = KnapsackMutation(resources, max_weight, max_volume)

    assert mutation.gene_set == resources
    assert mutation.max_weight == max_weight
    assert mutation.max_volume == max_volume


def test_remaining_weight_KnapsackChromosome():
    """GIVEN a KnapsackChromosome
    WHEN the remaining_weight method is called with a max_weight
    THEN the correct remaining weight is returned
    """

    max_weight = 50

    genes = [ResourceQuantity(Resource("Flour", 1680, 0.265, 0.41), 3)]

    assert np.isclose(
        KnapsackChromosome(genes).remaining_weight(max_weight), max_weight - 3 * 0.265
    )


def test_remaining_weight_KnapsackChromosome():
    """GIVEN a KnapsackChromosome
    WHEN the remaining_volumne method is called with a max_volumne
    THEN the correct remaining volumne is returned
    """

    max_volume = 100

    genes = [ResourceQuantity(Resource("Flour", 1680, 0.265, 0.41), 3)]

    assert np.isclose(
        KnapsackChromosome(genes).remaining_volume(max_volume), max_volume - 3 * 0.41
    )


def test_remove_random_item_KnapsackMutation(resources, resource_quantities):
    """GIVEN a KnapsackMutation
    WHEN the remove_random_item method is called on a list of genes
    THEN a random item is removed from the list
    """

    random.seed(1)

    max_weight = 50
    max_volume = 100

    mutation = KnapsackMutation(resources, max_weight, max_volume)

    genes = resource_quantities.copy()
    assert len(genes) == 3

    mutation.remove_random_item(genes)

    assert len(genes) == 2
    assert genes[0].resource.name == "Butter"
    assert genes[1].resource.name == "Sugar"

    KnapsackMutation.remove_random_item(genes)

    assert len(genes) == 1
    assert genes[0].resource.name == "Sugar"

    KnapsackMutation.remove_random_item(genes)

    assert len(genes) == 0


def test_choose_to_add_item_empty_KnapsackMutation(resources):
    """GIVEN a KnapsackMutation
    WHEN the choose_to_add_item method is called on an empty list
    THEN True is returned as there is something to add
    """

    random.seed(1)

    max_weight = 50
    max_volume = 100

    mutation = KnapsackMutation(resources, max_weight, max_volume)

    decisions_to_add = [mutation.choose_to_add_item([]) for ii in range(100)]

    assert sum(decisions_to_add) == 100


def test_choose_to_add_item_not_all_items_KnapsackMutation(
    resources, resource_quantities
):
    """GIVEN a KnapsackMutation
    WHEN the choose_to_add_item method is called on a list of genes
    THEN True is returned as there is something to add
    """

    random.seed(1)

    max_weight = 50
    max_volume = 100

    mutation = KnapsackMutation(resources, max_weight, max_volume)

    decisions_to_add = [
        mutation.choose_to_add_item(resource_quantities[:2]) for ii in range(10000)
    ]

    # roughly 1% of the time, we should choose to add an item should be added
    assert sum(decisions_to_add) == 83


def test_add_random_item_KnapsackMutation(resources):
    """GIVEN a KnapsackMutation
    WHEN the add_random_item method is called on a list of genes
    THEN a random item is added to the list
    """

    random.seed(1)

    max_weight = 50
    max_volume = 100

    mutation = KnapsackMutation(resources, max_weight, max_volume)

    genes = []
    assert len(genes) == 0

    genes = mutation.add_random_item(genes)

    assert len(genes) == 1
    assert genes[0].resource.name == "Flour"


def test_choose_to_change_item_full_KnapsackMutation(resources, resource_quantities):
    """GIVEN a KnapsackMutation
    WHEN the choose_to_change_item method is called on a full list of genes
    THEN False is returned as there is nothing to change
    """

    random.seed(1)

    max_weight = 50
    max_volume = 100

    mutation = KnapsackMutation(resources, max_weight, max_volume)

    assert not mutation.choose_to_change_item(resource_quantities)


def test_choose_to_change_item_empty_KnapsackMutation(resources, resource_quantities):
    """GIVEN a KnapsackMutation
    WHEN the choose_to_change_item method is called on an empty list
    THEN True is returned as there is something to change
    """

    random.seed(2)

    max_weight = 50
    max_volume = 100

    mutation = KnapsackMutation(resources, max_weight, max_volume)

    decisions_to_add = [
        mutation.choose_to_change_item(resource_quantities[:2]) for ii in range(10000)
    ]

    # roughly 20% of the time, we should choose to add an item should be added
    assert sum(decisions_to_add) == 2032


def test_call_KnapsackMutation(resources, resource_quantities):

    random.seed(2)

    max_weight = 50
    max_volume = 100

    parent = KnapsackChromosome(resource_quantities.copy())

    mutation = KnapsackMutation(resources, max_weight, max_volume)

    child = mutation(parent)

    assert isinstance(child, KnapsackChromosome)
    assert child.age == parent.age
    assert child.genes != parent.genes
    assert len(parent.genes) == 3
    assert len(child.genes) == 2


def test_cookies_fill_knapsack():

    random.seed(1)

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

    assert best.total_value == 29200
    assert best.total_weight <= max_weight
    assert best.total_volume <= max_volume
