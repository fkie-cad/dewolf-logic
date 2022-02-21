"""Module implementing range simplification."""
from __future__ import annotations

from typing import Dict, List

from simplifier.operations import BitwiseAnd, BitwiseOr
from simplifier.operations.boolean import Relation
from simplifier.range_simplification_commons.expression_values import BoundRelation, ExpressionValues
from simplifier.range_simplification_commons.single_range_simplifier import SingleRangeSimplifier
from simplifier.world.nodes import Constant, Operation, TmpVariable, WorldObject
from simplifier.world.world import World


class BitwiseAndRangeSimplifier:
    """Class in charge of simplifying a conjunction of ranges."""

    def __init__(self, operation: Operation):
        """Initialize a new and-range-simplifier that simplifies the given operation."""
        assert isinstance(operation, BitwiseAnd), f"Simplification is only for And-operations, but we have {operation}."
        self._and_operation: BitwiseAnd = operation
        self._world = operation.world

    def simplify(self) -> None:
        """Simplify the ranges of the and-operation."""
        expression_value_for: Dict[WorldObject, ExpressionValues] = self._get_constant_constrains()
        for variable, expression_value in expression_value_for.items():
            if expression_value.is_unfulfillable():
                self._world.replace(self._and_operation, self._world.constant(0, 1))
            elif len(expression_value.must_values) == 1:
                self._world.add_operand(self._and_operation, self._world.bool_equal(variable, expression_value.must_value))
            else:
                self._add_unequal_constrains(expression_value, variable)
                self._add_bound_constrains(expression_value)

    def _add_unequal_constrains(self, expression_value: ExpressionValues, variable: WorldObject) -> None:
        """Given all unequal constrains, add range constrains for consecutive forbidden-values and != constrains for all remaining."""
        if not expression_value.forbidden_values:
            return
        sorted_forbidden_values = sorted(expression_value.forbidden_values, key=lambda const: const.unsigned)
        current_row = [sorted_forbidden_values[0]]
        for constant in sorted_forbidden_values[1:]:
            if current_row[-1].unsigned + 1 == constant.unsigned:
                current_row.append(constant)
            else:
                self._add_constraints_for(current_row, variable, expression_value)
                current_row = [constant]
        self._add_constraints_for(current_row, variable, expression_value)

    def _add_constraints_for(self, current_row: List[Constant], variable: WorldObject, expression_value: ExpressionValues) -> None:
        """Add constrains for the given list of forbidden-constants for the given variable."""
        if current_row[0] == expression_value.size_lower_bound.unsigned:
            self._world.add_operand(self._and_operation, self._world.unsigned_lt(current_row[-1], variable))
        elif current_row[-1] == expression_value.size_upper_bound.unsigned:
            self._world.add_operand(self._and_operation, self._world.unsigned_lt(variable, current_row[0]))
        elif current_row[-1] == expression_value.size_upper_bound.signed:
            self._world.add_operand(self._and_operation, self._world.signed_lt(variable, current_row[0]))
        elif len(current_row) <= 2:
            for const in current_row:
                self._world.add_operand(self._and_operation, self._world.bool_unequal(variable, const))
        else:
            lower_bound = self._world.unsigned_le(current_row[0], variable)
            upper_bound = self._world.unsigned_le(variable, current_row[-1])
            self._world.add_operand(self._and_operation, self._world.bitwise_negate(self._world.bitwise_and(lower_bound, upper_bound)))

    def _add_bound_constrains(self, expression_value: ExpressionValues):
        """Add bound-constrains for the given bounds of the given expression-values object."""
        variable = expression_value.expression
        if expression_value.upper_bound.signed:
            self._world.add_operand(self._and_operation, self._world.signed_le(variable, expression_value.upper_bound.signed))
        if expression_value.upper_bound.unsigned:
            self._world.add_operand(self._and_operation, self._world.unsigned_le(variable, expression_value.upper_bound.unsigned))
        if expression_value.lower_bound.signed:
            self._world.add_operand(self._and_operation, self._world.signed_le(expression_value.lower_bound.signed, variable))
        if expression_value.lower_bound.unsigned:
            self._world.add_operand(self._and_operation, self._world.unsigned_le(expression_value.lower_bound.unsigned, variable))

    def _get_constant_constrains(self) -> Dict[WorldObject, ExpressionValues]:
        """Create a dictionary that maps to each expression the set of constants it must fulfill."""
        expression_values_for: Dict[WorldObject, ExpressionValues] = dict()
        for operand in self._and_operation.operands:
            if bound_relation := BoundRelation.generate_from(operand):
                self._world.remove_operand(self._and_operation, operand)
                if bound_relation.expression not in expression_values_for:
                    expression_values_for[bound_relation.expression] = ExpressionValues(bound_relation.expression)
                expression_values_for[bound_relation.expression].update_with(bound_relation)
        for expression_value in expression_values_for.values():
            expression_value.simplify()
        return expression_values_for


#########################################################################
class BitwiseOrRangeSimplifier:
    """
    Class in charge of simplifying a conjunction of ranges.

    Note: We use the BitwiseAndRangeSimplifier for this since c1 | c2 | ... | cn = ~( ~c1 & ~c2 & ... & ~cn).
    """

    def __init__(self, operation: Operation):
        """Initialize a new or-range-simplifier that simplifies the given operation."""
        assert isinstance(operation, BitwiseOr), f"Simplification is only for Or-operations, but we have {operation}."
        self._or_operation: BitwiseOr = operation
        self._world = operation.world

    def simplify(self):
        """Simplify Or of ranges by negating the formula to an And condition and applying the BitwiseAndRangeSimplifier."""
        new_and_operation = self._create_and_operation_for_simplification()
        simplified_formula = self._simplify_and_operation(new_and_operation)

        if isinstance(simplified_formula, Constant):
            self._world.replace(self._or_operation, Constant(self._world, 1, 1))
        else:
            self._world.add_operand(self._or_operation, simplified_formula.negate())

    def _simplify_and_operation(self, new_and_operation: BitwiseAnd) -> WorldObject:
        """Simplify the given And-operation using the range-simplifier for and-operations."""
        self._world.define(defining_var := self._world.new_variable(1, tmp=True), new_and_operation)
        BitwiseAndRangeSimplifier(new_and_operation).simplify()
        simplified_formula = self._world.get_definition(defining_var)
        return simplified_formula

    def _create_and_operation_for_simplification(self) -> BitwiseAnd:
        """
        Create the And-condition we simplify with the BitwiseAndRangeSimplifier.

        - Add the negation of each range to a new and-operation.
        """
        new_and_operation = BitwiseAnd(self._world)
        for operand in self._or_operation.operands:
            if isinstance(operand, Relation):
                self._or_operation.remove_operand(operand)
                self._world.add_operand(new_and_operation, operand.negate())
        return new_and_operation


#########################################################################
class RangeSimplifier:
    """Class in charge of simplifying ranges."""

    def __init__(self, operation: WorldObject):
        """Initialize a new object of the range simplifier to simplify the given world-object."""
        self._world: World = operation.world
        self._defining_variable: TmpVariable = self._world.new_variable(1, tmp=True)  # type: ignore
        self._world.define(self._defining_variable, operation)

    @classmethod
    def simplify(cls, operation: WorldObject):
        """
        Simplify the ranges in the given operation.

        1. simplify the operation.
        2. split each relation to a binary relation
        3. simplify bitwise-and-operations
        4. simplify bitwise-or-relation
        """
        range_simplifier = cls(operation)
        if not range_simplifier._simplify_operation():
            operation.world.cleanup()
            return

        range_simplifier._split_non_binary_relations()
        range_simplifier._simplify_single_ranges()

        if not range_simplifier._simplify_operation():
            operation.world.cleanup()
            return

        if isinstance(range_simplifier.get_operation, BitwiseAnd):
            BitwiseAndRangeSimplifier(range_simplifier.get_operation).simplify()
        elif isinstance(range_simplifier.get_operation, BitwiseOr):
            BitwiseOrRangeSimplifier(range_simplifier.get_operation).simplify()

        range_simplifier._simplify_operation()

        operation.world.cleanup()

    @property
    def get_operation(self) -> Operation:
        """Return the operation we currently simplify."""
        operation = self._world.get_definition(self._defining_variable)
        assert isinstance(operation, Operation), "The operation can not be simplified"
        return operation

    @property
    def can_be_simplified(self) -> bool:
        """Check whether the operand is still simplifiable."""
        return isinstance(self._world.get_definition(self._defining_variable), Operation)

    def _simplify_operation(self) -> bool:
        """Simplifies the operation if possible and check whether it can still be simplified after the simplification."""
        if self.can_be_simplified:
            self.get_operation.simplify()
        return self.can_be_simplified

    def _split_non_binary_relations(self):
        """Split non-binary relations into binary relations."""
        for relation in [op for op in self._world.iter_postorder(self._defining_variable) if isinstance(op, Relation)]:
            relation.split()
        self.get_operation.simplify()

    def _simplify_single_ranges(self):
        """If the operation we want to simplify is a relation, we simplify this relation. Otherwise, we simplify all operand relations."""
        if isinstance(operation := self.get_operation, Relation):
            relation_operands: List[Relation] = [operation]  # type: ignore
        else:
            relation_operands: List[Relation] = [operand for operand in operation.operands if isinstance(operand, Relation)]
        for operand in relation_operands:
            SingleRangeSimplifier(operand).simplify()
