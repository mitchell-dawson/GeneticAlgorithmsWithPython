
from __future__ import annotations

from pathlib import Path

import networkx as nx


class Graph(nx.Graph):
    """A graph, represented as a set of nodes and a set of edges.
    """

    @staticmethod
    def from_col_file(file_name: Path) -> Graph:
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

        assert file_name.suffix == '.col'

        with open(file_name, 'r') as col_file:
            lines = col_file.readlines()
            
        graph = Graph()
            
        for line in lines:
            if line.startswith('c'):
                continue
            
            elif line.startswith('p'):
                _, _, checksum_num_nodes, checksum_num_edges = line.strip().split()
                checksum_num_nodes = int(checksum_num_nodes)
                checksum_num_edges = int(checksum_num_edges)

            elif line.startswith('e'):
                node1, node2 = line.split()[1:]
                graph.add_nodes_from((node1, node2))
                graph.add_edge(node1, node2)

            elif line.startswith('n'):
                node = line.split()[1]
                graph.add_node(node)
            else:
                raise ValueError(f"Unknown line type: {line}")
        
        if len(graph.nodes) != checksum_num_nodes:
            raise ColFileReadError(
                f"Number of nodes ({len(graph.nodes)}) does not match checksum ({checksum_num_nodes})"
            )

        if len(graph.edges) != checksum_num_edges:
            raise ColFileReadError(
                f"Number of edges ({len(graph.edges)}) does not match checksum ({checksum_num_edges})"
            )

        return graph
    
    def plot(self, output_folder: Path, positions, file_name: str = "graph.png") -> Path:
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

class ColFileReadError(Exception):
    """An error occurred while reading a .col file.
    """