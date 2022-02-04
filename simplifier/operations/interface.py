"""Module implementing the basic Operation interface."""
from __future__ import annotations

import logging
from abc import ABC
from typing import Optional

from simplifier.world.nodes import Operation, WorldObject


class UnaryOperation(Operation, ABC):
    """Base class for operations with exactly one operand."""

    @property
    def operand(self) -> WorldObject:
        """Return the single operand of the operation."""
        return self.operands[0]

    def add_operand(self, operand: WorldObject):
        """Add an operand to the operation, if it does not have an operand yet."""
        if self.operands:
            logging.error("Tried to add operand %r to %r", operand, self)
            raise ValueError("UnaryOperation already has an operand defined.")
        super().add_operand(operand)

    def fold(self, keep_form: bool = False):
        """Unary operations can not be folded."""
        pass


class OrderedOperation(Operation, ABC):
    """Base class for operations with ordered operands."""

    pass


class AssociativeOperation(Operation, ABC):
    """
    Base class for associative operations.

    e.g. (a + b) + c = a + (b + c)
    """

    def _promote_single_operand(self) -> WorldObject:
        """
        Check if there is only a single operand left after folding constants and variables.

        If that is the case, promote the operand.
        e.g. &(2) -> 2
        """
        operands = self.operands
        if len(operands) != 1:
            return self
        self.remove_operand(operands[0])
        self.world.replace(self, operands[0])
        return operands[0]

    def simplify(self, keep_form: bool = False) -> Optional[WorldObject]:
        """Simplify the operation by promoting single operands as well."""
        simplified_operation = super().simplify(keep_form)
        if simplified_operation is not None and simplified_operation != self:
            return simplified_operation
        operands = self.operands
        operation = self._promote_single_operand()
        if len(operands) == 1:
            return operation
        return None


class CommutativeOperation(Operation, ABC):
    """
    Base class of commutative operations.

    e.g. 1 + 2 = 2 + 1
    """

    def _fold_constants(self):
        """
        Fold all constants in the operand list.

        e.g. a & 1 & b & 3 = a & b & 1
        e.g. a & b & 0 = 0
        """
        constants = self.constants
        if len(constants) > 1:
            new_constant = self.eval(self.constants)
            self.add_operand(new_constant)
            for constant in constants:
                self.remove_operand(constant)

    def _promote_subexpression(self):
        """
        Promote nested suboperations of the same type.

        e.g. 1 + (a + b) + x = 1 + a + b + x
        """
        for operand in self.operands:
            if isinstance(operand, Operation) and operand.__class__ == self.__class__:
                for suboperand in operand.operands:
                    self.add_operand(suboperand)
                self.remove_operand(operand)

    def _promote_single_operand(self) -> WorldObject:
        """
        Check if there is only a single operand left after folding constants and variables.

        If that is the case, promote the operand.
        e.g. &(2) -> 2
        """
        operands = self.operands
        if len(operands) != 1:
            return self
        self.remove_operand(operands[0])
        self.world.replace(self, operands[0])
        return operands[0]

    def simplify(self, keep_form: bool = False) -> Optional[WorldObject]:
        """
        Simplify the given operation.

        keep_form=True means that after the simplification the formula is still in CNF reps. DNF form it had this form before.
        If the operand changes during the simplification, then we return the new operand.
        """
        self._promote_subexpression()
        self._fold_constants()
        return super().simplify(keep_form)
