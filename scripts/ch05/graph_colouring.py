from __future__ import annotations

import copy
import logging
import random
import time
from pathlib import Path
from typing import Any, Tuple

from src.genetic import (
    AbsoluteFitness,
    ChromosomeGenerator,
    GeneSet,
    Mutation,
    Runner,
    StoppingCriteria,
)
from src.graph import NodeColouredGraph


class Chromosome(NodeColouredGraph):
    """Chromosome for the graph colouring problem."""


class GraphColoringFitness(AbsoluteFitness):
    """Fitness function of a chromosome for the sort numbers problem."""

    chromosome: Chromosome

    def __init__(self) -> None:
        pass

    def __call__(self, chromosome: Chromosome) -> int:
        """Return the fitness of a chromosome. The fitness is 0 minus the number of
        edges that have the same colour node at each end. The higher the fitness, the
        fewer nodes have the same colour as an adjacent node.

        Parameters
        ----------
        chromosome : Chromosome
            The chromosome to evaluate.

        Returns
        -------
        int
            The fitness of the chromosome.
        """

        return -sum(
            [
                chromosome.node_colour_match(node1, node2)
                for node1, node2 in chromosome.edges
            ]
        )


class GraphColoringMutation(Mutation):
    """Mutation function for the graph colouring problem."""

    def __init__(self, fitness: GraphColoringFitness, gene_set: GeneSet):
        super().__init__(fitness, gene_set)

    def __call__(self, parent: Chromosome) -> Chromosome:
        child = parent.copy(as_view=False)

        # pick a random node and a random colour
        node = random.choice(list(child.nodes))
        new_colour = random.choice(list(self.gene_set))

        # mutate the node
        child.nodes[node]["colour"] = new_colour

        return child


class GraphColoringStoppingCriteria(StoppingCriteria):
    def __init__(self, target: int, fitness: GraphColoringFitness):
        self.target = target
        self.fitness = fitness

    def __call__(self, chromosome: Chromosome) -> bool:
        """Return true if can stop the genetic algorithm."""

        return self.fitness(chromosome) >= self.target


class GraphColoringChromosomeGenerator(ChromosomeGenerator):
    def __init__(self, gene_set, base_graph: Chromosome) -> None:
        self.gene_set = gene_set
        self.base_graph = base_graph

    def __call__(self) -> Chromosome:

        graph = self.base_graph.copy(as_view=False)

        for node in graph.nodes:
            random_colour = random.choice(list(self.gene_set))
            graph.nodes[node]["colour"] = random_colour

        return graph


class GraphColoringRunner(Runner):
    def __init__(
        self,
        chromosome_generator: ChromosomeGenerator,
        fitness: GraphColoringFitness,
        stopping_criteria: StoppingCriteria,
        mutate: Mutation,
    ):
        self.chromosome_generator = chromosome_generator
        self.fitness = fitness
        self.stopping_criteria = stopping_criteria
        self.mutate = mutate

    def display(self, candidate):
        time_diff = time.time() - self.start_time

        logging.debug("candidate=%s", candidate.colour_dict)
        logging.debug("fitness=%s", self.fitness(candidate))

        logging.info("time=%f", time_diff)


def graph_colouring(base_graph: Chromosome, gene_set: Tuple[str]) -> Chromosome:

    target = 0  # no edges with the same colour at each end

    fitness = GraphColoringFitness()
    chromosome_generator = GraphColoringChromosomeGenerator(gene_set, base_graph)
    stopping_criteria = GraphColoringStoppingCriteria(target, fitness)
    mutate = GraphColoringMutation(fitness, gene_set)

    runner = GraphColoringRunner(
        chromosome_generator, fitness, stopping_criteria, mutate
    )

    best = runner.run()
    print(best)
    return best


def main():

    logging_format = (
        "[%(levelname)8s :%(filename)20s:%(lineno)4s - %(funcName)10s] %(message)s"
    )
    logging.basicConfig(format=logging_format, level=logging.DEBUG)

    col_file_path = Path(
        "/Users/mitchell.dawson/CodeProjects/GeneticAlgorithmsWithPython/data_fixtures/raw/graph_colouring/adjacent_states.col"
    )

    base_graph = NodeColouredGraph().from_col_file(col_file_path)

    gene_set = ["red", "blue", "green", "yellow"]

    graph_colouring(base_graph, gene_set)


if __name__ == "__main__":
    main()
