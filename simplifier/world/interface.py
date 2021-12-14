"""Module implementing the main logic container class."""
import logging
from typing import Dict, Iterable, Iterator, List, Optional, TypeVar

from simplifier.operations.interface import CommutativeOperation
from simplifier.world.coloring import GraphColoringGenerator
from simplifier.world.edges import DefinitionEdge, OperandEdge, WorldRelation
from simplifier.world.graphs import Graph
from simplifier.world.nodes import BaseVariable, Operation, Variable, WorldObject


def _contains_defining_variable(*nodes: WorldObject) -> bool:
    """Check whether nodes contains at least one defining variable. If at least one node in nodes is a defining variable, we return true."""
    return any(isinstance(node, Variable) and node.world.get_definition(node) for node in nodes)


class WorldInterface:
    """
    Class managing terms valid under closed world assumption.

    More formally, this class is a container for terms and clauses considered
    and allows for simplification and SAT statements based on contradictions.

    Each variable in the world is represented by an unique node, whereas operations
    may be contained multiple times regardless of their operands via an unique id.

    There are two types of edges: Variable-Definition and Operation-Operand.

    Constants should never have child nodes.
    """

    def __init__(self):
        """
        Generate a new (empty) world.

        Under CWD no other term can be derived from CWD unless negated.
        """
        self._graph = Graph()
        self._variables: Dict[str, BaseVariable] = dict()

    def free_world_condition(self, condition: WorldObject):
        """
        Copy operations such that each has only in-degree 1.

        If a condition is given, do it only for this condition, otherwise for the hole world.
        """
        in_degree_larger_one_operations = set()
        condition_nodes = set()
        for node in self.iter_postorder(condition):
            if isinstance(node, Operation) and self._graph.in_degree(node) > 1:
                in_degree_larger_one_operations.add(node)
            condition_nodes.add(node)

        while in_degree_larger_one_operations:
            operation = in_degree_larger_one_operations.pop()
            operand_edges = self._graph.get_out_edges(operation)
            relations = {relation for relation in self._graph.get_in_edges(operation)[1:] if relation.source in condition_nodes}
            condition_nodes.remove(operation)
            for in_relation in relations:
                copy_node = operation.copy()
                condition_nodes.add(copy_node)
                for relation in operand_edges:
                    self._graph.add_edge(relation.copy(source=copy_node))
                    if self._graph.in_degree(relation.sink) > 1:
                        in_degree_larger_one_operations.add(relation.sink)
                self._graph.substitute_edge(in_relation, in_relation.copy(sink=copy_node))

    def free_world_conditions(self):
        """
        Copy operations such that each has only in-degree 1.

        If a condition is given, do it only for this condition, otherwise for the hole world.
        """
        in_degree_larger_one_operations = {
            node for node in self.iter_postorder() if isinstance(node, Operation) and self._graph.in_degree(node) > 1
        }
        while in_degree_larger_one_operations:
            operation = in_degree_larger_one_operations.pop()
            operand_edges = self._graph.get_out_edges(operation)
            for in_relation in self._graph.get_in_edges(operation)[1:]:
                copy_node = operation.copy()
                for relation in operand_edges:
                    self._graph.add_edge(relation.copy(source=copy_node))
                    if self._graph.in_degree(relation.sink) > 1:
                        in_degree_larger_one_operations.add(relation.sink)
                self._graph.substitute_edge(in_relation, in_relation.copy(sink=copy_node))

    def __len__(self) -> int:
        """Return the number of WorldObjects."""
        return len(self._graph)

    def simplify(self) -> None:
        """Try to simplify the graph utilizing the methods of the contained operations."""
        for node in list(self._graph.iter_postorder()):
            if not isinstance(node, Operation):
                continue
            node.simplify()

    def validate(self) -> bool:
        """
        Check whether all assertions in the world can be valid at the same time.

        TODO: Planed implementation for a later milestone.
        """
        logging.warning("WorldInterface.validate is still unimplemented")
        return True

    def terms(self) -> List[WorldObject]:
        """Return all root objects of the world graph."""
        return self._graph.get_roots()

    # Utility methods

    def replace(self, replacee: WorldObject, replacement: WorldObject):
        """Replace one world object with another, maintaining incoming edges."""
        logging.debug("Replace %r with %r", replacee, replacement)
        self._graph.replace(replacee, replacement)

    def substitute(self, replacee: WorldObject, replacement: WorldObject):
        """Substitute the given object with another object, maintaining all relations."""
        logging.debug("Substitute %r with %r", replacee, replacement)
        self._graph.substitute_node(replacee, replacement)

    def define(self, variable: BaseVariable, definition: WorldObject) -> WorldObject:
        """Define the value of the given variable."""
        if self._graph.get_successors(variable):
            raise ValueError(f"The variable {variable} can only be defined once.")
        self._graph.add_edge(DefinitionEdge(variable, definition))
        return definition

    def get_operands(self, operation: Operation) -> List[Operation]:
        """Return the operands of the given operation."""
        return [op_edge.operand for op_edge in sorted(self._graph.get_out_edges(operation), key=lambda edge: edge.index)]

    def get_definition(self, variable: BaseVariable) -> Optional[WorldObject]:
        """Return the definition of the given variable, if any."""
        successors: List[WorldObject] = self._graph.get_successors(variable)
        if successors:
            return successors[0]
        return None

    def add_operand(self, operation: Operation, operand: WorldObject, index: int = None):
        """Add an operand to the given operation."""
        if index is None:
            if edges := self._graph.get_out_edges(operation):
                index = max(edge.index for edge in edges) + 1
            else:
                index = 0
        self._graph.add_edge(OperandEdge(operation, operand, index))

    def remove_operand(self, operation: Operation, operand: WorldObject) -> bool:
        """Remove the first occurrence of the given operand from the given operation."""
        edges = sorted(self._graph.get_edges(operation, operand), key=lambda op: op.index)
        if edges:
            self._graph.remove_edge(edges[0])
            return True
        return False

    def add_operation_on_edge(self, edge: WorldRelation, operation: Operation):
        """Add the given operation on the given edge, by splitting it."""
        new_operation = self._add_operation(operation, [edge.sink])
        self._graph.add_edge(edge.copy(sink=new_operation))
        self._graph.remove_edge(edge)

    def parent_operation(self, node: WorldObject) -> List[Operation]:
        """Return the parent operations of a given World Object."""
        return self._graph.get_predecessors(node)

    def get_relation(self, node1: WorldObject, node2: WorldObject) -> List[WorldRelation]:
        """Return the relation between two nodes."""
        return self._graph.get_edges(node1, node2)

    def cleanup(self):
        """Remove all orphaned nodes in the graph. This are all nodes not reachable from a defining variable."""
        removable_nodes = {node for node in self.terms() if not isinstance(node, Variable)}
        while removable_nodes:
            current_node = removable_nodes.pop()
            for child in self._graph.get_successors(current_node):
                if not isinstance(child, Variable) and len(self._graph.get_predecessors(child)) == 1:
                    removable_nodes.add(child)
            self._graph.remove_node(current_node)

    def iter_postorder(self, source: WorldObject = None) -> Iterator[WorldObject]:
        """Iterate all nodes in post order starting at the given source."""
        yield from self._graph.iter_postorder(source)

    @classmethod
    def compare(cls, a: WorldObject, b: WorldObject) -> bool:
        """
        Compare whatever two WorldObjects describe the same AST.

        We treat variables that define a formula differently, i.e.,
        - var x = (w + z), var z = 1 implies x == (w + 1)
        - var x = (w & z), var y = (z & w) implies x == y
        -> In these cases x != (w+1) and x != y so we have to handle this in the sanity check a != b.
        """
        if id(a) == id(b):
            return True
        if a != b and not _contains_defining_variable(a, b):
            return False
        graph_coloring_generator = GraphColoringGenerator()
        graph_coloring_generator.color_subtree_with_head(a)
        graph_coloring_generator.color_subtree_with_head(b)
        return graph_coloring_generator.color_of_node(a) == graph_coloring_generator.color_of_node(b)

    # Private methods

    OperationType = TypeVar("OperationType", bound=Operation)

    def _add_operation(self, operation: OperationType, operands: Iterable[WorldObject]) -> OperationType:
        """Add the given operation to the world."""
        self._graph.add_node(operation)
        for index, operand in enumerate(operands):
            self._graph.add_edge(OperandEdge(operation, operand, index=index))
        return operation
