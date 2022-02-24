"""Module defining the edges utilized in the graph of a World instance."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from simplifier.world.graphs.interface import NODE, GraphEdgeInterface
from simplifier.world.nodes import BaseVariable, Operation, WorldObject


class WorldRelation(GraphEdgeInterface, ABC):
    """Interface class for edges contained the the graph of a World instance."""

    def __init__(self, source: WorldObject, sink: WorldObject):
        """Create an edge by setting a sink and a source."""
        self._source = source
        self._sink = sink

    @property
    def source(self) -> WorldObject:
        """Return the start (origin) of the edge."""
        return self._source

    @property
    def sink(self) -> WorldObject:
        """Return the end (target) of the edge."""
        return self._sink

    def __eq__(self, other) -> bool:
        """Check for equality based on source and sink."""
        return isinstance(other, WorldRelation) and (self._source, self._sink) == (
            other._source,
            other._sink,
        )

    def __hash__(self) -> int:
        """Generate a hash for the edge based on source and sink."""
        return hash(self._source) ^ hash(self._sink)

    def __repr__(self) -> str:
        """Return a string representation for debug purposes."""
        return f"{self.__class__.__name__}({self.source} -> {self.sink})"

    @property
    @abstractmethod
    def index(self) -> int:
        """Return the index of the edge."""


class DefinitionEdge(WorldRelation):
    """Class representing an edge between a variable and its definition."""

    _source: BaseVariable

    @property
    def index(self) -> int:
        """Return the index of the edge."""
        return 0

    @property
    def variable(self) -> BaseVariable:
        """Return the variable being defined by the edge."""
        return self._source

    @property
    def definition(self) -> WorldObject:
        """Return the WorldObject defining the variable."""
        return self._sink

    def copy(self, source: Optional[NODE] = None, sink: Optional[NODE] = None) -> DefinitionEdge:
        """Generate a copy of the edge with updated fields."""
        assert not source or isinstance(source, BaseVariable)
        assert not sink or isinstance(sink, WorldObject)
        return DefinitionEdge(
            self.variable if not source else source,  # type: ignore
            self.definition if not sink else sink,  # type: ignore
        )


class OperandEdge(WorldRelation):
    """Class representing an edge between an operation and its operand."""

    _source: Operation

    def __init__(self, operation: Operation, operand: WorldObject, index: int = 0):
        """Initialize an edge between the given operation and its operand."""
        super().__init__(operation, operand)
        self._index = index

    @property
    def operation(self) -> Operation:
        """Return the source (start) of the edge."""
        return self._source

    @property
    def operand(self) -> WorldObject:
        """Return the sink (end) of the edge."""
        return self._sink

    @property
    def index(self) -> int:
        """Return the index of the edge."""
        return self._index

    def __eq__(self, other) -> bool:
        """Check whether two edges are equal."""
        return isinstance(other, OperandEdge) and super(OperandEdge, self).__eq__(other) and self._index == other._index

    def __hash__(self) -> int:
        """Generate a hash for the edge considering its index."""
        return super(OperandEdge, self).__hash__() ^ hash(self._index)

    def copy(self, source: Optional[NODE] = None, sink: Optional[NODE] = None, index: int = None) -> OperandEdge:
        """Generate a copy of the edge."""
        assert not source or isinstance(source, Operation)
        assert not sink or isinstance(sink, WorldObject)
        return OperandEdge(
            self.operation if not source else source,  # type: ignore
            self.operand if not sink else sink,  # type: ignore
            self.index if index is None else index,
        )
