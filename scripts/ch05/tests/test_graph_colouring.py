import random

import networkx as nx
import pytest

from scripts.ch05.graph_colouring import (
    GraphColoringChromosomeGenerator,
    GraphColoringFitness,
    GraphColoringMutation,
    GraphColouringChromosome,
    graph_colouring,
)
from src.graph import NodeColouredGraph


@pytest.fixture
def simple_col_file(data_fixtures_folder):
    return data_fixtures_folder / "raw/graph_colouring/simple.col"


def test_call_GraphColoringFitness():

    fitness = GraphColoringFitness()

    graph = NodeColouredGraph()
    graph.add_node(1, colour="red")
    graph.add_node(2, colour="red")
    graph.add_node(3, colour="blue")
    graph.add_node(4, colour="blue")
    graph.add_node(5, colour="green")

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)

    assert fitness(GraphColouringChromosome(graph)) == -2


def test_call_GraphColoringMutation():

    random.seed(1)

    fitness = GraphColoringFitness()
    gene_set = ["red", "blue", "green"]

    mutation = GraphColoringMutation(fitness, gene_set)

    parent = NodeColouredGraph()
    parent.add_node(1, colour="red")
    parent.add_node(2, colour="red")
    parent.add_node(3, colour="blue")

    parent = GraphColouringChromosome(parent)
    child = mutation(parent)

    assert child.genes.colour_dict == {1: "green", 2: "red", 3: "blue"}
    assert parent.genes.colour_dict == {1: "red", 2: "red", 3: "blue"}


def test_call_GraphColoringChromosomeGenerator():

    random.seed(1)

    graph = NodeColouredGraph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    gene_set = ["red", "blue", "green"]

    generator = GraphColoringChromosomeGenerator(gene_set, graph)

    chromosome = generator()

    assert chromosome.genes.colour_dict == {
        1: "red",
        2: "green",
        3: "red",
    }


def test_graph_colouring(simple_col_file):

    base_graph = NodeColouredGraph().from_col_file(simple_col_file)
    gene_set = ["red", "blue", "green", "yellow"]
    best = graph_colouring(base_graph, gene_set)

    assert set(best.genes.colour_dict.values()) == set(gene_set)
    assert best.genes.colour_dict["4"] == best.genes.colour_dict["5"]
