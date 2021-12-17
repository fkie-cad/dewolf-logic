"""Module for visitor ABCs."""
from abc import ABCMeta, abstractmethod
from typing import Generic

from simplifier.common import T
from simplifier.operations.arithmetic import *
from simplifier.operations.bitwise import *
from simplifier.operations.boolean import *
from simplifier.world.nodes import BaseVariable, Constant, Variable, WorldObject


class WorldObjectVisitor(Generic[T], metaclass=ABCMeta):
    """Visitor class for WorldObjects."""

    def visit(self, world_object: WorldObject) -> T:
        """Visit an AST node, dispatching to the correct handler."""
        return world_object.accept(self)

    @abstractmethod
    def visit_world_object(self, world_object: WorldObject):
        """Visit World Object, default case."""

    @abstractmethod
    def visit_variable(self, variable: BaseVariable):
        """Visit Variable."""

    @abstractmethod
    def visit_constant(self, constant: Constant):
        """Visit Constant."""

    @abstractmethod
    def visit_unsigned_add(self, unsigned_add_operation: UnsignedAddition):
        """Visit an unsigned addition."""

    @abstractmethod
    def visit_signed_add(self, signed_add_operation: SignedAddition):
        """Visit an signed addition."""

    @abstractmethod
    def visit_unsigned_sub(self, unsigned_sub_operation: UnsignedSubtraction):
        """Visit an unsigned subtraction."""

    @abstractmethod
    def visit_signed_sub(self, signed_sub_operation: SignedSubtraction):
        """Visit an signed subtraction."""

    @abstractmethod
    def visit_unsigned_mod(self, unsigned_mod_operation: UnsignedModulo):
        """Visit an unsigned modulo."""

    @abstractmethod
    def visit_signed_mod(self, signed_mod_operation: SignedModulo):
        """Visit an signed modulo."""

    @abstractmethod
    def visit_unsigned_mul(self, unsigned_mul_operation: UnsignedMultiplication):
        """Visit an unsigned multiplication."""

    @abstractmethod
    def visit_signed_mul(self, signed_mul_operation: SignedMultiplication):
        """Visit an signed multiplication."""

    @abstractmethod
    def visit_unsigned_div(self, unsigned_div_operation: UnsignedDivision):
        """Visit an unsigned division."""

    @abstractmethod
    def visit_signed_div(self, signed_div_operation: SignedDivision):
        """Visit an signed division."""

    @abstractmethod
    def visit_bitwise_and(self, and_operation: BitwiseAnd):
        """Visit BitwiseAnd."""

    @abstractmethod
    def visit_bitwise_or(self, or_operation: BitwiseOr):
        """Visit BitwiseOr."""

    @abstractmethod
    def visit_bitwise_xor(self, xor_operation: BitwiseXor):
        """Visit BitwiseXor."""

    @abstractmethod
    def visit_shift_left(self, xor_operation: ShiftLeft):
        """Visit ShiftLeft."""

    @abstractmethod
    def visit_shift_right(self, xor_operation: ShiftRight):
        """Visit ShiftRight."""

    @abstractmethod
    def visit_rotate_left(self, rot_operation: RotateLeft):
        """Visit RotateLeft."""

    @abstractmethod
    def visit_rotate_right(self, rot_operation: RotateRight):
        """Visit RotateRight."""

    @abstractmethod
    def visit_bitwise_negate(self, neg_operation: BitwiseNegate):
        """Visit BitwiseNegate."""

    @abstractmethod
    def visit_bool_negate(self, not_operation: Not):
        """Visit Not."""

    @abstractmethod
    def visit_bool_equal(self, relation: Equal):
        """Visit ==."""

    @abstractmethod
    def visit_bool_unequal(self, relation: Unequal):
        """Visit !=."""

    @abstractmethod
    def visit_signed_gt(self, relation: SignedGreater):
        """Visit SignedGreater."""

    @abstractmethod
    def visit_signed_ge(self, relation: SignedGreaterEqual):
        """Visit SignedGreaterEqual."""

    @abstractmethod
    def visit_signed_lt(self, relation: SignedLesser):
        """Visit SignedLesser."""

    @abstractmethod
    def visit_signed_le(self, relation: SignedLesserEqual):
        """Visit SignedLesserEqual."""

    @abstractmethod
    def visit_unsigned_gt(self, relation: UnsignedGreater):
        """Visit UnsignedGreater."""

    @abstractmethod
    def visit_unsigned_ge(self, relation: UnsignedGreaterEqual):
        """Visit UnsignedGreaterEqual."""

    @abstractmethod
    def visit_unsigned_lt(self, relation: UnsignedLesser):
        """Visit UnsignedLesser."""

    @abstractmethod
    def visit_unsigned_le(self, relation: UnsignedLesserEqual):
        """Visit UnsignedLesserEqual."""
