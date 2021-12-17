"""Defines utility method for graphs in the BaseGraph class."""
from typing import Union

from simplifier.world.graphs.interface import EDGE, NODE, GraphEdgeInterface, GraphNodeInterface
from simplifier.world.graphs.nxgraph import NetworkXGraph


class BaseGraph(NetworkXGraph):
    """Basic graph implementation for the project."""

    def has_edge(self, source: NODE, sink: NODE) -> bool:
        """Check if the given edge is part of the graph."""
        return self.get_edge(source, sink) is not None

    def replace(self, replacee: NODE, replacement: NODE):
        """Replace the given node with another, maintaining incoming edges."""
        self.add_node(replacement)
        for in_edge in self.get_in_edges(replacee):  # type: GraphEdgeInterface
            self.add_edge(in_edge.copy(sink=replacement))
        self.remove_node(replacee)

    def substitute(self, replacee: Union[NODE, EDGE], replacement: Union[NODE, EDGE]):
        """Substitute edges or nodes with each other, invoking the corresponding function."""
        if isinstance(replacee, GraphNodeInterface) and isinstance(replacement, GraphNodeInterface):
            self.substitute_node(replacee, replacement)
        elif isinstance(replacee, GraphEdgeInterface) and isinstance(replacement, GraphEdgeInterface):
            self.substitute_edge(replacee, replacement)
        else:
            raise TypeError("Replacee and Replacement do not have a coherent type for substitution.")

    def substitute_edge(self, replacee: EDGE, replacement: EDGE):
        """Substitute an edge in the graph with another."""
        self.remove_edge(replacee)
        self.add_edge(replacement)

    def substitute_node(self, replacee: NODE, replacement: NODE):
        """Substitute one node in the graph with another."""
        self.add_node(replacement)
        for in_edge in self.get_in_edges(replacee):  # type: GraphEdgeInterface
            self.add_edge(in_edge.copy(sink=replacement))
        if self.out_degree(replacee) == 1 and self.get_successors(replacee)[0] == replacement:
            self.remove_node(replacee)
            return
        for out_edge in self.get_out_edges(replacee):  # type: GraphEdgeInterface
            self.add_edge(out_edge.copy(source=replacement))
        self.remove_node(replacee)
