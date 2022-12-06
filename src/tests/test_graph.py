import pytest

from src.graph import Graph


@pytest.fixture
def simple_col_file(data_fixtures_folder):
    return data_fixtures_folder / "raw/graph_coloring/simple.col"


@pytest.fixture
def adjacent_states_col_file(data_fixtures_folder):
    return data_fixtures_folder / "raw/graph_coloring/adjacent_states.col"

def test_add_node_Graph():
    """GIVEN a graph
    WHEN a node is added
    THEN the node is in the graph
    """
    
    graph = Graph()
    graph.add_node(1)
    graph.add_node('a')

    assert 1 in graph.nodes
    assert 'a' in graph.nodes

def test_add_node_with_attributes_Graph():
    """GIVEN a graph
    WHEN a node is added with attributes
    THEN the node is in the graph with the attributes
    """
    
    graph = Graph()
    graph.add_node(1, color='red')
    graph.add_node('a', color='green')

    assert 1 in graph.nodes
    assert 'a' in graph.nodes
    assert graph.nodes[1]['color'] == 'red'
    assert graph.nodes['a']['color'] == 'green'

def test_add_edge_Graph():
    """GIVEN a graph
    WHEN an edge is added
    THEN the edge is in the graph
    """
    
    graph = Graph()
    graph.add_node(1)
    graph.add_node('a')
    graph.add_edge(1, 'a')

    assert len(graph.edges) == 1
    assert (1, 'a') in graph.edges
    assert ('a', 1) in graph.edges

def test_adj_Graph():
    """GIVEN a graph with several nodes and edges
    WHEN the adjacency list is accessed
    THEN the adjacency list is correct
    """
    
    graph = Graph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_node(3)

    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    assert graph.adj[1] == {2: {}}
    assert graph.adj[2] == {1: {}, 3: {}}
    assert graph.adj[3] == {2: {}}

def test_from_col_file__simple_Graph(simple_col_file):
    """GIVEN a col file for a simple graph
    WHEN a graph is created using the col file
    THEN the graph has the correct number of nodes and edges
    """
    graph = Graph.from_col_file(simple_col_file)
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 9

def test_from_col_file_states_Graph(adjacent_states_col_file):
    """GIVEN a col file containing states
    WHEN a graph is created using the col file
    THEN the graph has the correct number of nodes and edges
    """
    graph = Graph.from_col_file(adjacent_states_col_file)
    assert len(graph.nodes) == 51
    assert len(graph.edges) == 107