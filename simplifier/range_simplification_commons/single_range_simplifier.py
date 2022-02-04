"""Module implementing single range simplification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

from simplifier.operations import Equal
from simplifier.operations.boolean import BaseGreater, BaseGreaterEqual, BaseLesser, BaseLesserEqual, Relation, SignedRelation
from simplifier.range_simplification_commons.utils import eval_constant, get_min_max_number_for_size
from simplifier.world.nodes import Constant, WorldObject


@dataclass
class RichOperand:
    """
    Class in charge of handling additional information about an operand.

    Here the size bound and whether it belongs to a signed or unsigned operation.
    """

    operand: WorldObject
    size_bound: Constant
    signed: bool

    @property
    def constant_bound(self) -> Constant:
        """Return the bound for the operand, i.e, the operand itself if it is a constant and the size bound otherwise."""
        if isinstance(self.operand, Constant):
            return self.operand
        return self.size_bound

    @property
    def bound_value(self) -> Union[int, float]:
        """Return the integer value of the bound."""
        return eval_constant(self.constant_bound, self.signed)

    @property
    def operand_equals_size_bound(self) -> bool:
        """Check whether the operand is equal to the size bound."""
        return self.operand == self.size_bound


class SingleRangeSimplifier:
    """Class in charge of simplifying a single relation."""

    def __init__(self, relation: Relation):
        """
        Initialize a new single range simplifier that simplifies the given relation.

        Here, we assume that all relations are binary relations.
        """
        assert len(operands := relation.operands) == 2, f"SingleRangeSimplifier is limited to binary operations ({len(operands)} operands)."
        self._relation: Relation = relation
        self._world = relation.world

        left_size_bound, right_size_bound = self._get_size_bounds()
        self._left_operand: RichOperand = RichOperand(operands[0], left_size_bound, isinstance(relation, SignedRelation))
        self._right_operand: RichOperand = RichOperand(operands[1], right_size_bound, isinstance(relation, SignedRelation))

    def simplify(self) -> None:
        """
        Simplify a single binary relation.

        - We only simplify when at least on operand is a constant.
        - Depending on the bounds of the relation and the size-bounds we try to simplify the relation.
        """
        if not any(isinstance(operand, Constant) for operand in self._relation.operands):
            return
        if isinstance(self._relation, (BaseGreater, BaseLesser)):
            if self._relation_is_unfulfillable():
                self._evaluate_relation_to_false()
            elif self._are_consecutive_numbers():
                self._evaluate_relation_to_single_possible_value()
        elif isinstance(self._relation, (BaseLesserEqual, BaseGreaterEqual)):
            if self._left_operand.operand_equals_size_bound or self._right_operand.operand_equals_size_bound:
                self._evaluate_relation_to_true()
            elif self._left_operand.constant_bound == self._right_operand.constant_bound:
                self._world.substitute(self._relation, Equal(self._world))

    def _evaluate_relation_to_single_possible_value(self) -> None:
        """Replace the relation var ~ const by var == size_bound_of_var."""
        expression_operand = self._left_operand if not isinstance(self._left_operand.operand, Constant) else self._right_operand
        self._world.replace(self._relation, self._world.bool_equal(expression_operand.operand, expression_operand.size_bound))

    def _evaluate_relation_to_true(self) -> None:
        """Replace the relation by True."""
        self._world.replace(self._relation, self._world.constant(1, 1))

    def _evaluate_relation_to_false(self) -> None:
        """Replace the relation by false."""
        self._world.replace(self._relation, self._world.constant(0, 1))

    def _relation_is_unfulfillable(self) -> bool:
        """Check whether the given relation is unfulfillable considering the given bounds."""
        return self._relation.eval([self._left_operand.constant_bound, self._right_operand.constant_bound]).unsigned == 0

    def _are_consecutive_numbers(self) -> bool:
        """Check whether the relation bounds are consecutive."""
        return abs(self._left_operand.bound_value - self._right_operand.bound_value) == 1

    def _get_min_max_value(self) -> Tuple[Constant, Constant]:
        """Return the minimum and maximum possible value for the operands in the relation depending on their size."""
        size = max(operand.size for operand in self._relation.operands)
        min_number, max_number = get_min_max_number_for_size(size, isinstance(self._relation, SignedRelation))
        min_value = Constant(self._world, min_number, size)
        max_value = Constant(self._world, max_number, size)
        return min_value, max_value

    def _get_size_bounds(self) -> Tuple[Constant, Constant]:
        """Return the upper and lower bound for the operation according to its size."""
        min_value, max_value = self._get_min_max_value()
        if isinstance(self._relation, (BaseGreater, BaseGreaterEqual)):
            return max_value, min_value
        return min_value, max_value
