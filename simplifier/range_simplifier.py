"""Module implementing range simplification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

from simplifier.operations import BitwiseAnd, BitwiseOr
from simplifier.operations.boolean import (
    BaseGreater,
    BaseGreaterEqual,
    BaseLesser,
    BaseLesserEqual,
    Equal,
    NonEqualRelation,
    Relation,
    SignedRelation,
    Unequal,
)
from simplifier.world.nodes import Constant, Operation, TmpVariable, Variable, WorldObject
from simplifier.world.world import World


def eval_constant(constant: Constant, is_signed: bool) -> Union[float, int]:
    """Evaluate the value of the given constant depending on the is_signed parameter."""
    return constant.signed if is_signed else constant.unsigned


def smaller(first_constant: Constant, second_constant: Constant, is_signed: bool) -> bool:
    """Evaluate whether the first constant is smaller than the second constant."""
    return eval_constant(first_constant, is_signed) < eval_constant(second_constant, is_signed)


def modify_constant_by(const: Constant, modification: int) -> Constant:
    """Modify the given constant by the given integer and return this modified constant."""
    return Constant(const.world, const.unsigned + modification, const.size)


def get_min_max_number_for_size(size: int, is_signed: bool) -> Tuple[int, int]:
    """Return the minimum and maximum possible value of the given size depending on is_signed."""
    max_number = (1 << (size - int(is_signed))) - 1
    min_number = -max_number - 1 if is_signed else 0
    return min_number, max_number


###################################################
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


##########################################################


@dataclass
class ConstantBound:
    """Dataclass maintaining the bound of an world-object."""

    signed: Optional[Constant] = None
    unsigned: Optional[Constant] = None

    @classmethod
    def get_size_bounds_for(cls, expression: WorldObject) -> Tuple[ConstantBound, ConstantBound]:
        """Return the lower and upper bound for the given expression depending on its size."""
        bit_size = expression.size
        signed_min, signed_max = get_min_max_number_for_size(bit_size, True)
        unsigned_min, unsigned_max = get_min_max_number_for_size(bit_size, False)
        world = expression.world
        lower_bound = cls(Constant(world, signed_min, bit_size), Constant(world, unsigned_min, bit_size))
        upper_bound = cls(Constant(world, signed_max, bit_size), Constant(world, unsigned_max, bit_size))
        return lower_bound, upper_bound

    def less_than(self, constant: Constant) -> bool:
        """Check whether the constant bound itself is lesser that the given constant."""
        return (self.signed is not None and smaller(self.signed, constant, True)) or (
            self.unsigned is not None and smaller(self.unsigned, constant, False)
        )

    def greater_than(self, constant: Constant) -> bool:
        """Check whether the constant bound itself is larger than the given constant."""
        return (self.signed is not None and smaller(constant, self.signed, True)) or (
            self.unsigned is not None and smaller(constant, self.unsigned, False)
        )

    def update_upper_bound(self, new_constant: Constant, is_signed: bool) -> None:
        """Update the constant bound itself, when interpreting it as an upper bound, assuming the given constant is also an upper bound."""
        if is_signed and (self.signed is None or smaller(new_constant, self.signed, True)):
            self.signed = new_constant
        elif not is_signed and (self.unsigned is None or smaller(new_constant, self.unsigned, False)):
            self.unsigned = new_constant

    def update_lower_bound(self, new_constant: Constant, is_signed: bool) -> None:
        """Update the constant bound itself, when interpreting it as a lower bound, assuming the given constant is also a lower bound."""
        if is_signed and (self.signed is None or smaller(self.signed, new_constant, True)):
            self.signed = new_constant
        elif not is_signed and (self.unsigned is None or smaller(self.unsigned, new_constant, False)):
            self.unsigned = new_constant

    def modify_if_contained_in(self, constants: Set[Constant], param: int) -> bool:
        """Modify the bound the the given parameter, when the constant is contained in the given set of constants."""
        modify = False
        if self.signed in constants:
            self.signed = modify_constant_by(self.signed, param)
            modify = True
        if self.unsigned in constants:
            self.unsigned = modify_constant_by(self.unsigned, param)
            modify = True
        return modify

    def contained_in(self, constants: Set[Constant]) -> bool:
        """Check whether the constant bound itself is contained in the set of constants."""
        return self.signed in constants or self.unsigned in constants


class ExpressionValues:
    """Class in charge of handling possible values of an expression."""

    def __init__(self, expression: WorldObject):
        """
        Initialize a new expression values handler for the given world object.

         - self.expression: The expression whose possible values we consider.
        - self.must_values: A set of constants that must be equal to the given expression.
        - self.forbidden_values: A set of constants the expression can not have.
        - self.size_lower_bound: The lower bound of the expression value depending on its size.
        - self.size_upper_bound: The upper bound of the expression value depending on its size.
        - self.lower_bound: The lower bound of the expression value depending on range constraints an d forbidden values.
        - self.upper_bound: The upper bound of the expression value depending on range constraints an d forbidden values.
        """
        self.expression: WorldObject = expression
        self.must_values: Set[Constant] = set()
        self.forbidden_values: Set[Constant] = set()
        self.size_lower_bound, self.size_upper_bound = ConstantBound.get_size_bounds_for(expression)
        self.lower_bound: ConstantBound = ConstantBound()
        self.upper_bound: ConstantBound = ConstantBound()

    @property
    def must_value(self) -> Optional[Constant]:
        """Return the first must-value, if one exists; otherwise return None."""
        for element in self.must_values:
            return element
        return None

    def update_with(self, bound_relation: BoundRelation) -> None:
        """Update the expression values regarding the given relation."""
        relation = bound_relation.relation
        if isinstance(relation, Equal):
            self.must_values.add(bound_relation.constant)
        elif isinstance(relation, Unequal):
            self.forbidden_values.add(bound_relation.constant)
        else:  # should be BaseGreaterEqual,  BaseLesserEqual, BaseGreater and BaseLesser
            self._update_bounds(bound_relation)

    def is_unfulfillable(self) -> bool:
        """Return false when not all constrains of the expression values can be fulfilled simultaneously."""
        if len(self.must_values) > 1 or self.must_values & self.forbidden_values:
            return True
        if self.must_value:
            if self.upper_bound.less_than(self.must_value):
                return True
            if self.lower_bound.greater_than(self.must_value):
                return True
        if self.upper_bound_smaller_lower_bound():
            return True
        return False

    def _update_bounds(self, bound_relation: BoundRelation) -> None:
        """Update the upper and lower bound according to the given relation."""
        if isinstance(const := bound_relation.greater_operand, Constant):
            if isinstance(bound_relation.relation, NonEqualRelation):
                const = modify_constant_by(const, -1)  # type: ignore
            self.upper_bound.update_upper_bound(const, bound_relation.is_signed)
        elif isinstance(const := bound_relation.smaller_operand, Constant):
            if isinstance(bound_relation.relation, NonEqualRelation):
                const = modify_constant_by(const, 1)  # type: ignore
            self.lower_bound.update_lower_bound(const, bound_relation.is_signed)

    def upper_bound_smaller_lower_bound(self) -> bool:
        """Check whether the upper bound is smaller than the lower bound."""
        if self.lower_bound.signed and self.upper_bound.signed and smaller(self.upper_bound.signed, self.lower_bound.signed, True):
            return True
        if self.lower_bound.unsigned and self.upper_bound.unsigned and smaller(self.upper_bound.unsigned, self.lower_bound.unsigned, False):
            return True
        return False

    def simplify(self):
        """Simplify the given possible value sets."""
        self._add_must_value_if_bounds_equal_size_bounds()
        self._refine_bounds_using_forbidden_values()
        self._remove_redundant_forbidden_values()
        self._add_must_value_if_bounds_are_equal()
        self._add_must_value_if_bounds_equal_size_bounds()

    def _refine_bounds_using_forbidden_values(self) -> None:
        """Refine the upper and lower bounds considering the forbidden values."""
        while self.upper_bound.contained_in(self.forbidden_values) or self.lower_bound.contained_in(self.forbidden_values):
            self.upper_bound.modify_if_contained_in(self.forbidden_values, -1)
            self.lower_bound.modify_if_contained_in(self.forbidden_values, 1)

    def _remove_redundant_forbidden_values(self) -> None:
        """Remove redundant forbidden values, i.e., values that are already forbidden due to the upper and lower bounds."""
        for forbidden_constant in list(self.forbidden_values):
            if self.lower_bound.greater_than(forbidden_constant) or self.upper_bound.less_than(forbidden_constant):
                self.forbidden_values.remove(forbidden_constant)

    def _add_must_value_if_bounds_are_equal(self) -> None:
        """If the lower and upper bound are equal, then the expression must have this value, so we add it in this case."""
        self._add_must_value_if_values_are_equal(self.lower_bound, self.upper_bound)

    def _add_must_value_if_bounds_equal_size_bounds(self) -> None:
        """Add must value if lower bound equals upper-size-bound and if upper_bound equals lower_size-bound."""
        self._add_must_value_if_values_are_equal(self.lower_bound, self.size_upper_bound)
        self._add_must_value_if_values_are_equal(self.upper_bound, self.size_lower_bound)

    def _add_must_value_if_values_are_equal(self, first_const_bound: ConstantBound, second_const_bound: ConstantBound) -> None:
        """Add must value if the given bounds are equal."""
        if first_const_bound.signed and first_const_bound.signed == second_const_bound.signed:
            self.must_values.add(first_const_bound.signed)
        if first_const_bound.unsigned and first_const_bound.unsigned == second_const_bound.unsigned:
            self.must_values.add(first_const_bound.unsigned)


class BoundRelation:
    """Class in charge of handling relations and saving additional information."""

    def __init__(self, relation: Relation):
        """Initialize a new bound-relation."""
        assert self._is_valid_bound_relation(relation), f"{relation} is not a binary relation with exactly one constant operand!"
        self._relation: Relation = relation
        self._operands = relation.operands
        self._constant: Constant = relation.constants[0]
        self._expression: WorldObject = self.left_operand if self.right_operand == self._constant else self.right_operand

    @classmethod
    def generate_from(cls, relation: WorldObject) -> Optional[BoundRelation]:
        """Return a new bound relation given the input relation."""
        if not cls._is_valid_bound_relation(relation):
            return None
        return cls(relation)  # type: ignore

    @staticmethod
    def _is_valid_bound_relation(relation: WorldObject) -> bool:
        """Check that it is a binary relation where exactly one operand is a constant."""
        if not isinstance(relation, Relation) or len(relation.operands) != 2:
            return False
        return len(relation.constants) == 1

    @property
    def relation(self) -> Relation:
        """Return the relation."""
        return self._relation

    @property
    def constant(self) -> Constant:
        """Return the constant operand of the relation."""
        return self._constant

    @property
    def expression(self) -> WorldObject:
        """Return the non-constant operand of the relation."""
        return self._expression

    @property
    def left_operand(self) -> WorldObject:
        """Return the left-operand of the relation."""
        return self._operands[0]

    @property
    def right_operand(self) -> WorldObject:
        """Return the right-operand of the relation."""
        return self._operands[1]

    @property
    def smaller_operand(self) -> Optional[WorldObject]:
        """Return the smaller operand of the relation or None if the relation is of type == or !=."""
        if isinstance(self._relation, (BaseGreater, BaseGreaterEqual)):
            return self.right_operand
        if isinstance(self._relation, (BaseLesserEqual, BaseLesser)):
            return self.left_operand
        return None

    @property
    def greater_operand(self) -> Optional[WorldObject]:
        """Return the greater operand of the relation or None if the relation is of type == or !=."""
        if isinstance(self._relation, (BaseGreater, BaseGreaterEqual)):
            return self.left_operand
        if isinstance(self._relation, (BaseLesserEqual, BaseLesser)):
            return self.right_operand
        return None

    @property
    def is_signed(self) -> bool:
        """Return whether the relation is signed."""
        return isinstance(self._relation, SignedRelation)


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

    SIMPLIFIABLE_OPERANDS = (Relation, BitwiseAnd, BitwiseOr)

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
        return isinstance(self._world.get_definition(self._defining_variable), self.SIMPLIFIABLE_OPERANDS)

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
