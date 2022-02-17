"""Module implementing boolean operations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union

from simplifier.common import T
from simplifier.operations import BitwiseAnd
from simplifier.operations.interface import OrderedOperation, UnaryOperation
from simplifier.util.decorators import dirty
from simplifier.world.nodes import BitVector, Constant, Operation, WorldObject

if TYPE_CHECKING:
    from simplifier.visitor import WorldObjectVisitor


class BooleanOperation(Operation, ABC):
    """Base class for all Operands which can only return TRUE or FALSE."""

    @property
    def size(self):
        """Return the size of the operation. Since it is a boolean operation, the size is always 1."""
        return 1


class Not(BooleanOperation, UnaryOperation):
    """Class representing the negation of another boolean operation."""

    SYMBOL = "!"

    def eval(self, operands: List[Constant]) -> Constant:
        """Eval True to False and False to True."""
        assert len(operands) == 1
        return self.world.constant(1 if operands[0].unsigned == 0 else 0, 1)

    @dirty
    def factorize(self) -> Optional[Operation]:
        """Resolve the negation, if possible."""
        if self.operand.__class__ in BOOLEAN_NEGATIONS:
            return BOOLEAN_NEGATIONS[self.operand.__class__](self.world).replace_operands(self.operand.operands)  # type: ignore
        if isinstance(self.operand, Not):
            return self.operand.operand  # type: ignore
        return None

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean Not."""
        return visitor.visit_bool_negate(self)


class Relation(BooleanOperation, OrderedOperation, ABC):
    """Base class for comparison operations."""

    @staticmethod
    @abstractmethod
    def _operation(a: Union[float, int], b: Union[float, int]) -> bool:
        """Return the base python relation of the operation."""

    @staticmethod
    @abstractmethod
    def _interpret(self: Constant) -> Union[float, int]:
        """
        Return the function to be invoked on constants to return the actual value.

        e.g. either Constant.signed or Constant.unsigned
        """

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the value of the relation based on a list of constant operands."""
        for a, b in ((operands[i], operands[i + 1]) for i in range(len(operands) - 1)):
            if not self._operation(self._interpret(a), self._interpret(b)):
                return self.world.constant(0, 1)
        return self.world.constant(1, 1)

    def _failfast(self):
        """Check whether the Relation can be evaluated to False based on constant operands."""
        return self.eval(self.constants).unsigned == 0

    @dirty
    def simplify(self, keep_form: bool = False) -> Optional[WorldObject]:
        """Overwrite simplify to call collapse first."""
        if collapsed := self.collapse():
            return collapsed
        self.fold(keep_form=keep_form)
        if not keep_form and (new_operation := self.factorize()):
            self.world.replace(self, new_operation)
            return new_operation
        return self

    def collapse(self) -> Optional[BitVector]:
        """Collapse the relation to False if one variable occurs multiple times."""
        if self._failfast():
            replacement = self.world.constant(0, 1)
            self.world.replace(self, replacement)
            return replacement
        return super().collapse()

    def factorize(self) -> Optional[Operation]:
        """Try to reformulate this operation in a simpler form, possibly altering the operation."""
        pass

    def fold(self, keep_form: bool = False):
        """Fold the constants - omitting the first constant of each pair of constants."""
        operands = self.operands
        if len(self.operands) == len(self.constants):
            return
        for i in range(len(operands)):
            if isinstance(operands[i], Constant):
                neighbors = {name: operands[j] for name, j in (("left", i - 1), ("right", i + 1)) if 0 <= j < len(operands)}
                if "right" in neighbors and operands[i] == neighbors["right"]:
                    self.remove_operand(operands[i])
                elif all(isinstance(neighbor, Constant) for neighbor in neighbors.values()):
                    self.remove_operand(operands[i])

    def split(self) -> Union[Relation, BitwiseAnd]:
        """
        Split the given relation into Binary-Relations by taking pairs of consecutive operands.

        More princely, we split 1 < a < 3 < b into 1 < a & a < 3 & 3 < b and a == b == 1 into a == b & b == 1
        """
        if len(self.operands) <= 2:
            return self
        new_operation = self.world.bitwise_and(*(self._get_splitted_operands()))
        self.world.replace(self, new_operation)
        return new_operation

    @abstractmethod
    def _get_splitted_operands(self) -> Iterator[Relation]:
        """Split the given relation in pairs of relations."""

    def negate(self, level: Optional[int] = None, predecessors: Optional[List[Operation]] = None) -> WorldObject:
        """Negate the Relation."""
        if level == 0:
            return super().negate(0)
        self.world.substitute(self, new_op := BOOLEAN_NEGATIONS[self.__class__](self.world))
        return new_op


class UnsignedRelation(Relation, ABC):
    """Base class for unsigned relations."""

    @staticmethod
    def _interpret(constant: Constant) -> Union[float, int]:
        """Return the unsigned value of the given constant."""
        return constant.unsigned


class SignedRelation(Relation, ABC):
    """Base class for signed relations."""

    @staticmethod
    def _interpret(constant: Constant) -> Union[float, int]:
        """Return the unsigned value of the given constant."""
        return constant.signed


class NonEqualRelation(Relation, ABC):
    """Base class for unequal relations such as != < >."""

    def collapse(self) -> Optional[BitVector]:
        """Collapse the relation to False if one variable occurs multiple times."""
        variables = self.variables
        if len(variables) != len(set(variables)):
            value = self.world.constant(0, 1)
            self.world.replace(self, value)
            return value
        return super().collapse()


class EqualRelation(Relation, ABC):
    """Base class for semi-equal relations such as == <= >=."""

    def collapse(self) -> Optional[BitVector]:
        """Collapse the relation to True if all operands are the same variable."""
        variables = self.variables
        if len(self.operands) == len(variables) and len(set(variables)) == 1:
            value = self.world.constant(1, 1)
            self.world.replace(self, value)
            return value
        return super().collapse()


class Equal(EqualRelation, UnsignedRelation):
    """Class implementing the equivalence relation."""

    SYMBOL = "=="

    @staticmethod
    def _operation(a: Union[float, int], b: Union[float, int]) -> bool:
        """Evaluate the operation given two operands."""
        return a == b

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean Equal."""
        return visitor.visit_bool_equal(self)

    def fold(self, keep_form: bool = False):
        """Fold the constants."""
        constants = self.constants
        if len(constants) > 1 and len(set((self._interpret(constant) for constant in constants))) == 1:
            for constant in constants[1:]:
                self.remove_operand(constant)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.bool_equal(op1, op2)


class Unequal(NonEqualRelation, UnsignedRelation):
    """Class implementing the not-equivalent relation."""

    SYMBOL = "!="

    @staticmethod
    def _operation(a: Union[float, int], b: Union[float, int]) -> bool:
        """Evaluate the operation given two operands."""
        return a != b

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean Unequal."""
        return visitor.visit_bool_unequal(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.bool_unequal(op1, op2)


class BaseGreater(NonEqualRelation, ABC):
    """Base class for > relations."""

    @staticmethod
    def _operation(a: Union[float, int], b: Union[float, int]) -> bool:
        """Evaluate the operation given two operands."""
        return a > b


class UnsignedGreater(BaseGreater, UnsignedRelation):
    """Class implementing the unsigned > relation."""

    SYMBOL = "u>"

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean Unequal."""
        return visitor.visit_unsigned_gt(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.unsigned_gt(op1, op2)


class SignedGreater(BaseGreater, SignedRelation):
    """Class implementing the signed > relation."""

    SYMBOL = "s>"

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean SignedGreater."""
        return visitor.visit_signed_gt(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.signed_gt(op1, op2)


class BaseGreaterEqual(EqualRelation, ABC):
    """Base class for >= relations."""

    @staticmethod
    def _operation(a: Union[float, int], b: Union[float, int]) -> bool:
        """Evaluate the operation given two operands."""
        return a >= b


class UnsignedGreaterEqual(BaseGreaterEqual, UnsignedRelation):
    """Class implementing the unsigned >= relation."""

    SYMBOL = "u>="

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean UnsignedGreaterEqual."""
        return visitor.visit_unsigned_ge(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.unsigned_ge(op1, op2)


class SignedGreaterEqual(BaseGreaterEqual, SignedRelation):
    """Class implementing the signed >= relation."""

    SYMBOL = "s>="

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean SignedGreaterEqual."""
        return visitor.visit_signed_ge(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.signed_ge(op1, op2)


class BaseLesser(NonEqualRelation, ABC):
    """Base class for < relations."""

    @staticmethod
    def _operation(a: Union[float, int], b: Union[float, int]) -> bool:
        """Evaluate the operation given two operands."""
        return a < b


class UnsignedLesser(BaseLesser, UnsignedRelation):
    """Class implementing the unsigned < relation."""

    SYMBOL = "u<"

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean UnsignedLesser."""
        return visitor.visit_unsigned_lt(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.unsigned_lt(op1, op2)


class SignedLesser(BaseLesser, SignedRelation):
    """Class implementing the signed < relation."""

    SYMBOL = "s<"

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean SignedLesser."""
        return visitor.visit_signed_lt(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.signed_lt(op1, op2)


class BaseLesserEqual(EqualRelation, ABC):
    """Base class for <= relations."""

    @staticmethod
    def _operation(a: Union[float, int], b: Union[float, int]) -> bool:
        """Evaluate the operation given two operands."""
        return a <= b


class UnsignedLesserEqual(BaseLesserEqual, UnsignedRelation):
    """Class implementing the unsigned <= relation."""

    SYMBOL = "u<="

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean UnsignedLesserEqual."""
        return visitor.visit_unsigned_le(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.unsigned_le(op1, op2)


class SignedLesserEqual(BaseLesserEqual, SignedRelation):
    """Class implementing the signed <= relation."""

    SYMBOL = "s<="

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for boolean SignedLesserEqual."""
        return visitor.visit_signed_le(self)

    def _get_splitted_operands(self) -> Iterator[Relation]:
        operands = self.operands
        for op1, op2 in zip(operands[:-1], operands[1:]):
            yield self.world.signed_le(op1, op2)


# Dict to lookup negations of boolean operations.
BOOLEAN_NEGATIONS: Dict[Type, Type] = {
    Equal: Unequal,
    Unequal: Equal,
    SignedGreater: SignedLesserEqual,
    SignedLesserEqual: SignedGreater,
    SignedLesser: SignedGreaterEqual,
    SignedGreaterEqual: SignedLesser,
    UnsignedGreater: UnsignedLesserEqual,
    UnsignedLesserEqual: UnsignedGreater,
    UnsignedLesser: UnsignedGreaterEqual,
    UnsignedGreaterEqual: UnsignedLesser,
}
