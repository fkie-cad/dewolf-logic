"""Module for normal form-visitor."""
from abc import ABC, abstractmethod
from typing import Union

from simplifier.operations.bitwise import BitwiseAnd, BitwiseNegate, BitwiseOr
from simplifier.visitor.visitor import WorldObjectVisitor
from simplifier.world.nodes import BaseVariable, Constant, WorldObject


class NormalFormVisitor(WorldObjectVisitor, ABC):
    """Visitor class for normal form transformation."""

    def __init__(self, expression_to_simplify: WorldObject, *args, **kwargs):
        """Initialize normal form transformation and transform given world-object to normal form."""
        self.visit(expression_to_simplify)

    def visit_world_object(self, world_object: WorldObject):
        """By default, we do nothing for normal form transformation."""
        pass

    def visit_variable(self, variable: BaseVariable):
        """If variable is a defining variable then replace the term that is defined by the variable in a term in normal form."""
        if term := variable.world.get_definition(variable):
            self.visit(term)

    def visit_constant(self, constant: Constant):
        """Constant is in normal form."""
        pass

    @abstractmethod
    def visit_bitwise_and(self, and_operation: BitwiseAnd):
        """Transform given And-operation to normal form."""

    @abstractmethod
    def visit_bitwise_or(self, or_operation: BitwiseOr):
        """Transform given Or-operation to normal form."""

    def visit_bitwise_negate(self, neg_operation: BitwiseNegate):
        """Transform given Negate-operation to normal form."""
        if resolve_negation := neg_operation.dissolve_negation():
            self.visit(resolve_negation)

    def _recursive_call_visitor(self, operation: Union[BitwiseAnd, BitwiseOr]):
        """
        Call visitor on each operand recursively.

        For CNF-form, we do this if the operation is an AND-operation, because then we only have to bring the operands into CNF form.
        For DNF-form, we do this if the operation is an OR-operation, because then we only have to bring the operands into DNF form.
        """
        if simplified := operation.simplify(keep_form=True):
            self.visit(simplified)
            return
        for child in operation.children:
            self.visit(child)
        operation.simplify(keep_form=True)

    def _multiplying_out_a_term(self, operation: Union[BitwiseAnd, BitwiseOr]):
        """
        Bring the operation to normal form by multiplying out one operand.

        For CNF-form, we have an OR-operation and search for an operand that is a conjunction to obtain an equivalent AND-formula.
        For DNF-form, we have an AND-operation and search for an operand that is a disjunction to obtain an equivalent OR-formula.
        """
        if simplified := operation.simplify(keep_form=True):
            self.visit(simplified)
            return
        junction = operation.get_junction()
        if junction:
            new_condition = operation.get_equivalent_condition_of_negated_type(junction)
            if simplified_cond := new_condition.simplify(keep_form=True):
                self.visit(simplified_cond)
                return
            for cond in new_condition.operands:
                self.visit(cond)
            new_condition.simplify(keep_form=True)

    visit_unsigned_add = visit_world_object  # type: ignore
    visit_signed_add = visit_world_object  # type: ignore
    visit_unsigned_sub = visit_world_object  # type: ignore
    visit_signed_sub = visit_world_object  # type: ignore
    visit_unsigned_mod = visit_world_object  # type: ignore
    visit_signed_mod = visit_world_object  # type: ignore
    visit_unsigned_mul = visit_world_object  # type: ignore
    visit_signed_mul = visit_world_object  # type: ignore
    visit_unsigned_div = visit_world_object  # type: ignore
    visit_signed_div = visit_world_object  # type: ignore

    visit_bitwise_xor = visit_world_object  # type: ignore
    visit_shift_left = visit_world_object  # type: ignore
    visit_shift_right = visit_world_object  # type: ignore
    visit_rotate_left = visit_world_object  # type: ignore
    visit_rotate_right = visit_world_object  # type: ignore

    visit_bool_negate = visit_bitwise_negate  # type: ignore
    visit_bool_equal = visit_world_object  # type: ignore
    visit_bool_unequal = visit_world_object  # type: ignore

    visit_signed_gt = visit_world_object  # type: ignore
    visit_signed_ge = visit_world_object  # type: ignore
    visit_signed_lt = visit_world_object  # type: ignore
    visit_signed_le = visit_world_object  # type: ignore
    visit_unsigned_gt = visit_world_object  # type: ignore
    visit_unsigned_ge = visit_world_object  # type: ignore
    visit_unsigned_lt = visit_world_object  # type: ignore
    visit_unsigned_le = visit_world_object  # type: ignore


class ToCnfVisitor(NormalFormVisitor):
    """Visitor class for CNF transformation."""

    def visit_bitwise_and(self, and_operation: BitwiseAnd):
        """Transform given And-operation to CNF."""
        self._recursive_call_visitor(and_operation)

    def visit_bitwise_or(self, or_operation: BitwiseOr):
        """Transform given Or-operation to CNF."""
        self._multiplying_out_a_term(or_operation)


class ToDnfVisitor(NormalFormVisitor):
    """Visitor class for DNF transformation."""

    def visit_bitwise_and(self, and_operation: BitwiseAnd):
        """Transform given And-operation to DNF."""
        self._multiplying_out_a_term(and_operation)

    def visit_bitwise_or(self, or_operation: BitwiseOr):
        """Transform given Or-operation to DNF."""
        self._recursive_call_visitor(or_operation)
