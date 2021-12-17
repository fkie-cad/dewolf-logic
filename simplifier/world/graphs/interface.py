"""Defines a generic graph interface suitable for multiple graph backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Optional, Tuple, TypeVar, Union


class GraphNodeInterface(ABC):
    """Basic Interface for graph nodes."""

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation."""

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Graph nodes should be equal for equal content."""

    @abstractmethod
    def __hash__(self) -> int:
        """Graph nodes should always have an unique hash."""

    @abstractmethod
    def copy(self) -> GraphNodeInterface:
        """Return a copy of the graph node."""

    def copy_tree(self) -> GraphNodeInterface:
        """Return a deep copy of the graph node."""
        return self.copy()


class GraphEdgeInterface(ABC):
    """Interface for graph edges."""

    @property
    @abstractmethod
    def source(self) -> GraphNodeInterface:
        """Return the origin of the edge."""

    @property
    @abstractmethod
    def sink(self) -> GraphNodeInterface:
        """Return the target of the edge."""

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check whether two edges are equal."""

    @abstractmethod
    def copy(self, source: Optional[NODE] = None, sink: Optional[NODE] = None) -> GraphEdgeInterface:
        """Copy the edge, returning a new object."""


NODE = TypeVar("NODE", bound=GraphNodeInterface)
EDGE = TypeVar("EDGE", bound=GraphEdgeInterface)


class GraphInterface(ABC):
    """Basic interface for all graph backends."""

    # Graph properties
    @abstractmethod
    def get_roots(self) -> Tuple[NODE, ...]:
        """Return all root nodes of the graph."""

    @abstractmethod
    def get_leaves(self) -> Tuple[NODE, ...]:
        """Return all leaf nodes of the graph."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the overall amount of nodes."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check whether the two uniquely labeled graphs are equal."""

    @abstractmethod
    def __iter__(self) -> Iterator[NODE]:
        """Iterate all nodes contained in the graph."""

    def __contains__(self, obj: Union[NODE, EDGE]):
        """Check if an node or edge is contained in the graph."""
        return obj in self.nodes or obj in self.edges

    @property
    @abstractmethod
    def edges(self) -> Tuple[EDGE, ...]:
        """Return a tuple containing all edges of the graph."""

    @property
    @abstractmethod
    def nodes(self) -> Tuple[NODE, ...]:
        """Return a tuple containing all nodes in the graph."""

    @abstractmethod
    def copy(self) -> GraphInterface:
        """Return a deep copy of the graph."""

    @abstractmethod
    def subgraph(self, nodes: Tuple[NODE, ...]) -> GraphInterface:
        """Return a shallow copy of the graph containing the given nodes."""

    # Graph manipulation

    @abstractmethod
    def add_node(self, node: NODE):
        """Add a node to the graph."""

    def add_nodes_from(self, nodes: Iterable[NODE]):
        """Add multiple nodes to the graph (legacy)."""
        for node in nodes:
            self.add_node(node)

    @abstractmethod
    def add_edge(self, edge: EDGE):
        """Add a single edge to the graph."""

    def add_edges_from(self, edges: Iterable[EDGE]):
        """Add multiple edges to the graph (legacy)."""
        for edge in edges:
            self.add_edge(edge)

    @abstractmethod
    def remove_node(self, node: NODE):
        """Remove the given node from the graph."""

    def remove_nodes_from(self, nodes: Iterable[NODE]):
        """Remove all nodes from the given iterator."""
        for node in nodes:
            self.remove_node(node)

    @abstractmethod
    def remove_edge(self, edge: EDGE):
        """Remove the given edge from the graph."""

    def remove_edges_from(self, edges: Iterable[EDGE]):
        """Remove all nodes in the given tuple from the graph."""
        for edge in edges:
            self.remove_edge(edge)

    # Graph traversal

    @abstractmethod
    def iter_depth_first(self, source: NODE) -> Iterator[NODE]:
        """Iterate all nodes in dfs fashion."""

    @abstractmethod
    def iter_breadth_first(self, source: NODE) -> Iterator[NODE]:
        """Iterate all nodes in dfs fashion."""

    @abstractmethod
    def iter_postorder(self, source: NODE = None) -> Iterator[NODE]:
        """Iterate all nodes in post order."""

    @abstractmethod
    def iter_preorder(self, source: NODE = None) -> Iterator[NODE]:
        """Iterate all nodes in pre order."""

    @abstractmethod
    def iter_topological(self) -> Iterator[NODE]:
        """Iterate all nodes in topological order. Raises an error if the graph is not acyclic."""

    # Node relations

    @abstractmethod
    def get_predecessors(self, node: NODE) -> Tuple[NODE, ...]:
        """Return a tuple of parent nodes of the given node."""

    @abstractmethod
    def get_successors(self, node: NODE) -> Tuple[NODE, ...]:
        """Return a tuple of child nodes of the given node."""

    @abstractmethod
    def get_adjacent_nodes(self, node: NODE) -> Tuple[NODE, ...]:
        """Return a tuple of all nodes directly connected to the given node."""

    # Edges

    @abstractmethod
    def get_in_edges(self, node: NODE) -> Tuple[EDGE, ...]:
        """Return a tuple of all edges targeting the given node."""

    @abstractmethod
    def get_out_edges(self, node: NODE) -> Tuple[EDGE, ...]:
        """Return a tuple of all edges starting at the given node."""

    @abstractmethod
    def get_incident_edges(self, node: NODE) -> Tuple[EDGE, ...]:
        """Get all edges either starting or ending at the given node."""

    @abstractmethod
    def get_edge(self, source: NODE, sink: NODE) -> Optional[EDGE]:
        """Get the first edge between the two given nodes, if any."""

    @abstractmethod
    def get_edges(self, source: NODE, sink: NODE) -> Tuple[EDGE, ...]:
        """Get all edges between the two nodes."""

    def in_degree(self, node: NODE) -> int:
        """Return the amount of edges pointing to the given node."""
        return len(self.get_predecessors(node))

    def out_degree(self, node: NODE) -> int:
        """Return the amount of edges starting at the given node."""
        return len(tuple(self.get_successors(node)))

    # Paths

    @abstractmethod
    def has_path(self, source: NODE, sink: NODE) -> bool:
        """Check whether there is a valid path between the given nodes."""

    @abstractmethod
    def get_paths(self, source: NODE, sink: NODE) -> Iterator[Tuple[NODE, ...]]:
        """Iterate all paths between the given nodes by iterating the nodes along the path."""


class BasicNode(GraphNodeInterface):
    """Basic node implementation for testing purposes."""

    def __init__(self, value: Any = None):
        """Initialize an node based on the given Value."""
        self._value = value

    def __str__(self) -> str:
        """Return a string representation of the node."""
        return str(self._value)

    def __repr__(self) -> str:
        """Return a representation for debug purposes."""
        return f"Node({str(self)})"

    def __eq__(self, other) -> bool:
        """Check equality based on the string representation."""
        return isinstance(other, BasicNode) and self._value == other._value

    def __hash__(self) -> int:
        """Return an unique hash value for the node."""
        return hash(self._value)

    def copy(self) -> BasicNode:
        """Return a new object with the same value."""
        return BasicNode(self._value)


class BasicEdge(GraphEdgeInterface):
    """A basic edge implementation for various purposes."""

    def __init__(self, source: GraphNodeInterface, sink: GraphNodeInterface):
        """Init an edge just based on source and sink."""
        self._source: GraphNodeInterface = source
        self._sink: GraphNodeInterface = sink

    @property
    def source(self) -> GraphNodeInterface:
        """Return the source of the edge."""
        return self._source

    @property
    def sink(self) -> GraphNodeInterface:
        """Return the sink of the edge."""
        return self._sink

    def __eq__(self, other) -> bool:
        """Check whether two edges are equal."""
        return other is not None and self.__dict__ == other.__dict__

    def copy(self, source: GraphNodeInterface = None, sink: GraphNodeInterface = None) -> GraphEdgeInterface:
        """Copy the edge, returning a new object."""
        return BasicEdge(source if source is not None else self._source, sink if sink is not None else self._sink)
