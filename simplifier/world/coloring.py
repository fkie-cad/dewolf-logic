"""Module for coloring the nodes of a world."""
from typing import Dict, NewType, Optional, Tuple

from simplifier.operations.boolean import BaseGreater, BaseGreaterEqual, Equal, Relation, Unequal
from simplifier.operations.interface import CommutativeOperation, Operation
from simplifier.world.nodes import Variable, WorldObject

Color = NewType("Color", int)


class GraphColoringGenerator:
    """Class that handles the coloring of one or multiple graphs representing a world."""

    def __init__(self):
        """Generate a new type of class GraphColoringGenerator."""
        self._color_of_node: Dict[WorldObject, Color] = dict()

    def color_of_node(self, node: WorldObject) -> Optional[Color]:
        """Return color of the given node."""
        return self._color_of_node.get(node)

    def color_subtree_with_head(self, head: WorldObject) -> None:
        """Color the head node and returns its color."""
        for node in head.world.iter_postorder(head):
            self._color_of_node[node] = self._compute_color_of(node)

    def _get_operand_classes(self, node: WorldObject) -> Tuple[Color, ...]:
        """Return tuple of classes of the operands."""
        if isinstance(node, (CommutativeOperation, Equal, Unequal)):
            return tuple(sorted(self._color_of_node[node] for node in node.operands))
        elif isinstance(node, (BaseGreater, BaseGreaterEqual)):
            return tuple(self._color_of_node[node] for node in reversed(node.operands))
        if isinstance(node, Operation):
            return tuple(self._color_of_node[node] for node in node.operands)
        return ()

    def _compute_color_of(self, node: WorldObject) -> Color:
        """Compute color of the node."""
        if isinstance(node, Variable) and (defining_term := node.world.get_definition(node)):
            return self._color_of_node[defining_term]
        operands: Tuple[Color, ...] = self._get_operand_classes(node)
        if isinstance(node, (BaseGreater, BaseGreaterEqual)):
            return Color(hash(f"{node.SYMBOL.replace('>', '<')}, {operands}"))
        if isinstance(node, Operation):
            return Color(hash(f"{node.SYMBOL}, {operands}"))
        return Color(hash(f"{node}, {operands}"))
