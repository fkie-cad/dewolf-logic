"""Module for CNF-visitor."""
from simplifier.operations.bitwise import BitwiseAnd, BitwiseNegate, BitwiseOr
from simplifier.visitor.visitor import WorldObjectVisitor
from simplifier.world.nodes import BaseVariable, Constant, WorldObject


class ToCnfVisitor(WorldObjectVisitor):
    """Visitor class for CNF transformation."""

    def __init__(self, expression_to_simplify: WorldObject, *args, **kwargs):
        """Initialize CNF transformation and transform given world-object to CNF form."""
        self.visit(expression_to_simplify)

    def visit_world_object(self, world_object: WorldObject):
        """By default we do nothing for CNF transformation."""
        pass

    def visit_variable(self, variable: BaseVariable):
        """If variable is a defining variable then replace the term that is defined by the variable in a term in CNF form."""
        if term := variable.world.get_definition(variable):
            self.visit(term)

    def visit_constant(self, constant: Constant):
        """Constant is in CNF form."""
        pass

    def visit_bitwise_and(self, and_operation: BitwiseAnd):
        """Transform given And-operation to CNF."""
        if simplified := and_operation.simplify(keep_form=True):
            self.visit(simplified)
            return
        for child in and_operation.children:
            self.visit(child)
        and_operation.simplify(keep_form=True)

    def visit_bitwise_or(self, or_operation: BitwiseOr):
        """Transform given Or-operation to CNF."""
        if simplified := or_operation.simplify(keep_form=True):
            self.visit(simplified)
            return
        conjunction = or_operation.get_conjunction()
        if conjunction:
            new_condition = or_operation.get_equivalent_and_condition(conjunction)
            if simplified_cond := new_condition.simplify(keep_form=True):
                self.visit(simplified_cond)
                return
            for cond in new_condition.operands:
                self.visit(cond)
            new_condition.simplify(keep_form=True)

    def visit_bitwise_negate(self, neg_operation: BitwiseNegate):
        """Transform given Negate-operation to CNF."""
        if resolve_negation := neg_operation.dissolve_negation():
            self.visit(resolve_negation)

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
