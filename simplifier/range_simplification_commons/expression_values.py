"""Module implementing bounds and possible values handling for range simplification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

from simplifier.operations import Equal, Unequal
from simplifier.operations.boolean import (
    BaseGreater,
    BaseGreaterEqual,
    BaseLesser,
    BaseLesserEqual,
    NonEqualRelation,
    Relation,
    SignedRelation,
)
from simplifier.range_simplification_commons.utils import eval_constant, get_min_max_number_for_size, modify_constant_by, smaller
from simplifier.world.nodes import Constant, WorldObject


@dataclass
class ConstantBound:
    """Dataclass maintaining the bound of a world-object."""

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
        - self.lower_bound: The lower bound of the expression value depending on range constraints and forbidden values.
        - self.upper_bound: The upper bound of the expression value depending on range constraints and forbidden values.
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
        if self._upper_bound_smaller_lower_bound():
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

    def _upper_bound_smaller_lower_bound(self) -> bool:
        """Check whether the upper bound is smaller than the lower bound."""
        if self.lower_bound.signed and self.upper_bound.signed and smaller(self.upper_bound.signed, self.lower_bound.signed, True):
            return True
        if self.lower_bound.unsigned and self.upper_bound.unsigned and smaller(self.upper_bound.unsigned, self.lower_bound.unsigned, False):
            return True
        return False

    def simplify(self):
        """Simplify the given possible value sets."""
        self._remove_redundant_bounds()
        self._add_must_value_if_bounds_equal_size_bounds()
        self._refine_bounds_using_forbidden_values()
        self._remove_redundant_forbidden_values()
        self._add_must_value_if_bounds_are_equal()
        self._add_must_value_if_bounds_equal_size_bounds()

    def _refine_bounds_using_forbidden_values(self) -> None:
        """Refine the upper and lower bounds considering the forbidden values."""
        while self.upper_bound.contained_in(self.forbidden_values) or self.lower_bound.contained_in(self.forbidden_values):
            if self.upper_bound.modify_if_contained_in(self.forbidden_values, -1):
                self._add_must_value_if_values_are_equal(self.upper_bound, self.size_lower_bound)
            if self.lower_bound.modify_if_contained_in(self.forbidden_values, 1):
                self._add_must_value_if_values_are_equal(self.lower_bound, self.size_upper_bound)

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

    def _remove_redundant_bounds(self) -> None:
        """Remove redundant bounds if signed and unsigned exist."""
        if self.upper_bound.signed and self.upper_bound.unsigned:
            self._combine_signed_unsigned_upper_bounds()
        if self.lower_bound.signed and self.lower_bound.unsigned:
            self._combine_singed_unsigned_lower_bounds()
        if self.lower_bound.signed and self.upper_bound.unsigned:
            self._combine_singed_lower_unsigned_upper_bounds()
        if self.upper_bound.signed and self.lower_bound.unsigned:
            self._combine_singed_upper_unsigned_lower_bounds()

    def _combine_signed_unsigned_upper_bounds(self):
        """
        Combine signed and unsigned values of upper bounds, i.e. x u<= c_u and x s<= c_s.

        MS = maximum size of a singed value, i.e., for size 4 it is 8
        1. c_u <= MS and c_s < 0: (x u<= c_u and x s<= c_s) can be simplified to (0 s<= x <= c_s) which is false
        2. c_u <= MS and c_s >= 0: (x u<= c_u and x s<= c_s) can be simplified to (x u<= min(c_s, c_u))
        3. c_u > MS and c_s < 0: (x u<= c_u and x s<= c_s) can be simplified to (x s<= min(c_s, c_u))
        """
        upper_size_bound_value_signed = eval_constant(self.size_upper_bound.signed, is_signed=True)
        signed_upper_bound_value = eval_constant(self.upper_bound.signed, is_signed=True)
        unsigned_upper_bound_value = eval_constant(self.upper_bound.unsigned, is_signed=False)
        if unsigned_upper_bound_value <= upper_size_bound_value_signed and signed_upper_bound_value < 0:
            self.lower_bound.signed = Constant(self.expression.world, 0, self.expression.size)
            self.upper_bound.unsigned = None
        elif unsigned_upper_bound_value <= upper_size_bound_value_signed and 0 <= signed_upper_bound_value:
            if unsigned_upper_bound_value > signed_upper_bound_value:
                self.upper_bound.unsigned = self.upper_bound.signed
            self.upper_bound.signed = None
        elif unsigned_upper_bound_value > upper_size_bound_value_signed and 0 > signed_upper_bound_value:
            if eval_constant(self.upper_bound.unsigned, is_signed=True) <= signed_upper_bound_value:
                self.upper_bound.signed = self.upper_bound.unsigned
            self.upper_bound.unsigned = None

    def _combine_singed_unsigned_lower_bounds(self):
        """
        Combine signed and unsigned values of lower bounds, i.e. x u>= c_u and x s>= c_s.

        MS = maximum size of a singed value, i.e., for size 4 it is 8
        1. c_u > MS and c_s >= 0: (x u>= c_u and x s>= c_s) can be simplified to (c_s s<= x < 0) which is false
        2. c_u > MS and c_s < 0: (x u>= c_u and x s>= c_s) can be simplified to (max(c_s, c_u) u<= x)
        3. c_u <= MS and c_s >= 0: (x u>= c_u and x s>= c_s) can be simplified to (max(c_s, c_u) s<= x)
        """
        upper_size_bound_value_signed = eval_constant(self.size_upper_bound.signed, is_signed=True)
        signed_lower_bound_value = eval_constant(self.lower_bound.signed, is_signed=True)
        unsigned_lower_bound_value = eval_constant(self.lower_bound.unsigned, is_signed=False)
        if signed_lower_bound_value >= 0 and unsigned_lower_bound_value > upper_size_bound_value_signed:
            self.upper_bound.signed = Constant(self.expression.world, -1, self.expression.size)
            self.lower_bound.unsigned = None
        elif unsigned_lower_bound_value <= upper_size_bound_value_signed and signed_lower_bound_value >= 0:
            if signed_lower_bound_value < unsigned_lower_bound_value:
                self.lower_bound.signed = self.lower_bound.unsigned
            self.lower_bound.unsigned = None
        elif unsigned_lower_bound_value > upper_size_bound_value_signed and signed_lower_bound_value < 0:
            if eval_constant(self.lower_bound.signed, is_signed=False) >= unsigned_lower_bound_value:
                self.lower_bound.unsigned = self.lower_bound.signed
            self.lower_bound.signed = None

    def _combine_singed_lower_unsigned_upper_bounds(self):
        """
        Combine signed lower and unsigned upper bound values, i.e. x u<= c_u and x s>= c_s.

        MS = maximum size of a singed value, i.e., for size 4 it is 8
        1. c_u <= MS and c_s < 0: (x u<= c_u and x s>= c_s) can be simplified to (x u<= c_u)
        2. c_u <= MS and c_s >= 0: (x u<= c_u and x s>= c_s) can be simplified to (c_s u<= x u<= c_u)
        3. c_u > MS and c_s >= 0: (x u<= c_u and x s>= c_s) can be simplified to (x s>= c_s)
        """
        upper_size_bound_value_signed = eval_constant(self.size_upper_bound.signed, is_signed=True)
        signed_lower_bound_value = eval_constant(self.lower_bound.signed, is_signed=True)
        unsigned_upper_bound_value = eval_constant(self.upper_bound.unsigned, is_signed=False)
        if unsigned_upper_bound_value <= upper_size_bound_value_signed and signed_lower_bound_value < 0:
            self.lower_bound.signed = None
        elif unsigned_upper_bound_value <= upper_size_bound_value_signed and signed_lower_bound_value >= 0:
            if signed_lower_bound_value > 0:
                self.lower_bound.unsigned = self.lower_bound.signed
            self.lower_bound.signed = None
        elif unsigned_upper_bound_value > upper_size_bound_value_signed and signed_lower_bound_value >= 0:
            self.upper_bound.unsigned = None

    def _combine_singed_upper_unsigned_lower_bounds(self):
        """
        Combine signed upper and unsigned lower bound values, i.e. x u>= c_u and x s<= c_s.

        MS = maximum size of a singed value, i.e., for size 4 it is 8
        1. c_u <= MS and c_s < 0: (x u>= c_u and x s<= c_s) can be simplified to (x s<= c_s)
        2. c_u > MS and c_s < 0: (x u>= c_u and x s<= c_s) can be simplified to (c_u s<= x s<= c_s)
        3. c_u > MS and c_s >= 0: (x u>= c_u and x s<= c_s) can be simplified to (x u>= c_u)
        """
        upper_size_bound_value_signed = eval_constant(self.size_upper_bound.signed, is_signed=True)
        signed_upper_bound_value = eval_constant(self.upper_bound.signed, is_signed=True)
        unsigned_lower_bound_value = eval_constant(self.lower_bound.unsigned, is_signed=False)
        if unsigned_lower_bound_value <= upper_size_bound_value_signed and signed_upper_bound_value < 0:
            self.lower_bound.unsigned = None
        elif unsigned_lower_bound_value > upper_size_bound_value_signed and signed_upper_bound_value < 0:
            if eval_constant(self.lower_bound.unsigned, is_signed=True) != -upper_size_bound_value_signed - 1:
                self.lower_bound.signed = self.lower_bound.unsigned
            self.lower_bound.unsigned = None
        elif unsigned_lower_bound_value > upper_size_bound_value_signed and signed_upper_bound_value >= 0:
            self.upper_bound.signed = None


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
