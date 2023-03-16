from __future__ import annotations

import logging
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple

import networkx as nx

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
from src.graph import NodeColouredGraph


class GraphColouringChromosome:
    """Chromosome for the graph colouring problem."""

    def __init__(self, genes: NodeColouredGraph, age: int = 0):
        self.genes = genes
        self.age = age


class GraphColoringFitness(AbsoluteFitness):
    """Fitness function of a chromosome for the sort numbers problem."""

    chromosome: GraphColouringChromosome

    def __init__(self) -> None:
        pass

    def __call__(self, chromosome: GraphColouringChromosome) -> int:
        """Return the fitness of a chromosome. The fitness is 0 minus the number of
        edges that have the same colour node at each end. The higher the fitness, the
        fewer nodes have the same colour as an adjacent node.

        Parameters
        ----------
        chromosome : GraphColouringChromosome
            The chromosome to evaluate.

        Returns
        -------
        int
            The fitness of the chromosome.
        """

        return -sum(
            [
                chromosome.genes.node_colour_match(node1, node2)
                for node1, node2 in chromosome.genes.edges
            ]
        )


class GraphColoringMutation(Mutation):
    """Mutation function for the graph colouring problem."""

    def __init__(self, fitness: GraphColoringFitness, gene_set: GeneSet):
        self.fitness = fitness
        self.gene_set = gene_set

    def __call__(self, parent: GraphColouringChromosome) -> GraphColouringChromosome:

        child = deepcopy(parent)

        # pick a random node and a random colour
        node = random.choice(list(child.genes.nodes))
        new_colour = random.choice(list(self.gene_set))

        # mutate the node
        child.genes.nodes[node]["colour"] = new_colour

        return child


class GraphColoringStoppingCriteria(StoppingCriteria):
    def __init__(self, target: int, fitness: GraphColoringFitness):
        self.target = target
        self.fitness = fitness

    def __call__(self, chromosome: GraphColouringChromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        return self.fitness(chromosome) >= self.target


class GraphColoringChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, gene_set, base_graph: NodeColouredGraph) -> None:
        self.gene_set = gene_set
        self.base_graph = base_graph

    def __call__(self) -> GraphColouringChromosome:

        graph = deepcopy(self.base_graph)

        for node in graph.nodes:
            random_colour = random.choice(list(self.gene_set))
            graph.nodes[node]["colour"] = random_colour

        return GraphColouringChromosome(graph, age=0)


class GraphColoringRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: GraphColoringFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
        age_annealing: AgeAnnealing,
        stagnation_detector: FitnessStagnationDetector,
    ):
        super().__init__(
            chromosome_generator,
            fitness,
            stopping_criteria,
            mutate,
            age_annealing,
            stagnation_detector,
        )

    def display(self, candidate):
        time_diff = time.time() - self.start_time

        logging.debug("candidate=%s", candidate.genes.colour_dict)
        logging.debug("fitness=%s", self.fitness(candidate))

        logging.info("time=%f", time_diff)


def graph_colouring(
    base_graph: GraphColouringChromosome,
    gene_set: Tuple[str],
    fitness_stagnation_limit: Union[float, int] = float("inf"),
    age_limit: float = float("inf"),
) -> GraphColouringChromosome:

    target = 0  # no edges with the same colour at each end

    fitness = GraphColoringFitness()
    chromosome_generator = GraphColoringChromosomeGenerator(gene_set, base_graph)
    stopping_criteria = GraphColoringStoppingCriteria(target, fitness)
    mutate = GraphColoringMutation(fitness, gene_set)
    age_annealing = AgeAnnealing(age_limit=age_limit)

    fitness_stagnation_detector = FitnessStagnationDetector(
        fitness, fitness_stagnation_limit
    )
    runner = GraphColoringRunner(
        chromosome_generator,
        fitness,
        stopping_criteria,
        mutate,
        age_annealing,
        fitness_stagnation_detector,
    )

    best = runner.run()
    print(best)
    return best


def main():

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.INFO)

    col_file_path = Path(
        "/Users/mitchell.dawson/CodeProjects/GeneticAlgorithmsWithPython/data_fixtures/raw/graph_colouring/adjacent_states.col"
    )

    output_folder = Path(
        "/Users/mitchell.dawson/CodeProjects/GeneticAlgorithmsWithPython/data/processed/graph_colouring"
    )

    base_graph = NodeColouredGraph().from_col_file(col_file_path)

    gene_set = ["red", "blue", "green", "yellow"]

    best = graph_colouring(base_graph, gene_set)

    positions = nx.planar_layout(best.genes)
    positions = nx.spring_layout(best.genes, pos=positions, seed=100, k=1)

    colour_map = []

    for node in best.genes:
        colour_map.append(best.genes.nodes[node]["colour"])

    best.genes.plot(output_folder, positions=positions, colour_map=colour_map)


if __name__ == "__main__":
    main()
