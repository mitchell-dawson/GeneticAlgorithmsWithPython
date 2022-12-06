from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
from matplotlib import pyplot as plt


class Graph(nx.Graph):
    """A graph, represented as a set of nodes and a set of edges."""

    def set_node_attribute(self, node: Any, attribute: str, value: Any) -> None:
        """Set an attribute of a node.

        Parameters
        ----------
        node : Any
            The node to set the attribute of.
        attribute : str
            The attribute to set.
        value : Any
            The value to set the attribute to.

        Raises
        ------
        ValueError
            If the node is not in the graph.
        """
        if node not in self.nodes:
            raise ValueError(f"Node {node} not in graph")

        nx.set_node_attributes(self, value, attribute)

    def from_col_file(self, file_name: Path) -> Graph:
        """Read a graph from a .col file. This is an alternative constructor.

        Parameters
        ----------
        file_name : str
            The name of the file to read.

        Returns
        -------
        Graph
            The graph read from the file.
        """

        assert file_name.suffix == ".col"

        with open(file_name, "r") as col_file:
            lines = col_file.readlines()

        (
            graph,
            checksum_num_nodes,
            checksum_num_edges,
        ) = self._populate_graph_from_col_file(lines)

        self._validate_col_file_checksum(graph, checksum_num_nodes, checksum_num_edges)

        return graph

    def _populate_graph_from_col_file(self, lines):

        graph = self.empty_graph()

        for line in lines:
            if line.startswith("c"):
                continue

            elif line.startswith("p"):
                _, _, checksum_num_nodes, checksum_num_edges = line.strip().split()
                checksum_num_nodes = int(checksum_num_nodes)
                checksum_num_edges = int(checksum_num_edges)

            elif line.startswith("e"):
                node1, node2 = line.split()[1:]
                graph.add_nodes_from((node1, node2))
                graph.add_edge(node1, node2)

            elif line.startswith("n"):
                node = line.split()[1]
                graph.add_node(node)
            else:
                raise ValueError(f"Unknown line type: {line}")

        return graph, checksum_num_nodes, checksum_num_edges

    @staticmethod
    def empty_graph() -> Graph:
        """Return an empty graph.

        Returns
        -------
        Graph
            An empty graph.
        """
        return Graph()

    @staticmethod
    def _validate_col_file_checksum(graph, checksum_num_nodes, checksum_num_edges):

        if len(graph.nodes) != checksum_num_nodes:
            raise ColFileReadError(
                f"Number of nodes ({len(graph.nodes)}) does not match checksum ({checksum_num_nodes})"
            )

        if len(graph.edges) != checksum_num_edges:
            raise ColFileReadError(
                f"Number of edges ({len(graph.edges)}) does not match checksum ({checksum_num_edges})"
            )

    def plot(
        self, output_folder: Path, positions, file_name: str = "graph.png"
    ) -> Path:
        """Plot the graph.

        Parameters
        ----------
        output_folder : Path
            The folder to save the plot to.
        file_name : str, optional
            The name of the file to save the plot to, by default "graph.png"

        Returns
        -------
        Path
            The path to the plot.
        """

        output_path = output_folder / file_name

        nx.draw_networkx(self, positions)

        plt.axis("off")
        plt.savefig(output_path)

        return output_path


class NodeColouredGraph(Graph):
    """A graph where each node has a colour."""

    @property
    def colour_dict(self) -> dict:
        """Return a dictionary mapping nodes to their colours.

        Returns
        -------
        dict
            A dictionary mapping nodes to their colours.
        """

        return nx.get_node_attributes(self, "colour")

    def node_colour_match(self, node1: Any, node2: Any) -> bool:
        """Return true if the nodes have the same colour

        Parameters
        ----------
        node1 : Any
            The first node.
        node2 : Any
            The second node.

        Returns
        -------
        bool
            True if the nodes have the same colour.
        """
        return self.nodes[node1]["colour"] == self.nodes[node2]["colour"]

    def from_col_file(self, file_name: Path) -> NodeColouredGraph:
        graph = super().from_col_file(file_name)
        nx.set_node_attributes(graph, None, "colour")
        return NodeColouredGraph(graph)


class ColFileReadError(Exception):
    """An error occurred while reading a .col file."""
