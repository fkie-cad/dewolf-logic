"""Module implementing the arithmetic operations."""
from __future__ import annotations

import functools
import operator
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, TypeVar, Union

from simplifier.common import T
from simplifier.operations.interface import AssociativeOperation, CommutativeOperation, OrderedOperation
from simplifier.util.decorators import dirty
from simplifier.world.nodes import Constant, Operation, Variable, WorldObject

if TYPE_CHECKING:
    from simplifier.visitor import WorldObjectVisitor


class ArithmeticOperation(Operation, ABC):
    """Base Operation class for arithmetic operations."""

    @property
    @abstractmethod
    def SIGNED(self) -> bool:
        """Return a boolean describing whether this operation interprets sign bits."""

    @dirty
    def fold(self, keep_form: bool = False):
        """
        Fold all constant operands utilizing the eval method.

        e.g. 2*x + 4 + 3 + 8*x = 2*x + 7 + 8*x
        """
        if len(self.constants) > 0:
            new_constant = self.eval(self.constants)
            for constant in self.constants:
                self.remove_operand(constant)
            self.add_operand(new_constant)


class BaseDivision(ArithmeticOperation, OrderedOperation, ABC):
    """Base interface for signed and unsigned Division."""

    SYMBOL = "/"

    @staticmethod
    def div(iterable: Iterable[Union[int, float]]) -> Union[int, float]:
        """Perform division over all elements of the iterable."""
        for item in iterable:
            if type(item) == float:
                return functools.reduce(operator.truediv, iterable)
        return functools.reduce(operator.floordiv, iterable)

    @dirty
    def factorize(self) -> Optional[Operation]:
        """Nothing to factorize."""
        pass


class SignedDivision(BaseDivision):
    """Class representing signed division operations."""

    SIGNED = True

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the signed division based on the given constant values."""
        return Constant(
            self._world, BaseDivision.div([constant.signed for constant in operands]), max([constant.size for constant in operands])
        )

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for SignedDivision."""
        return visitor.visit_signed_div(self)


class UnsignedDivision(BaseDivision):
    """Class representing unsigned division operations."""

    SIGNED = False

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the unsigned division based on the given constant values."""
        return Constant(
            self._world, BaseDivision.div([constant.unsigned for constant in operands]), max([constant.size for constant in operands])
        )

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for SignedDivision."""
        return visitor.visit_unsigned_div(self)


class BaseModulo(ArithmeticOperation, OrderedOperation, ABC):
    """Base interface for signed and unsigned Modulo."""

    SYMBOL = "%"

    @staticmethod
    def c_mod(a: int, b: int) -> int:
        """Perform C99 standard a modulo b."""
        m = abs(a) % abs(b)
        m = -(m) if a < 0 else m
        return m

    @staticmethod
    def mod(iterable: Iterable[int]) -> int:
        """Perform modulo over all elements of the iterable."""
        return functools.reduce(BaseModulo.c_mod, iterable)

    @dirty
    def factorize(self) -> Optional[Operation]:
        """Nothing to factorize."""
        pass


class SignedModulo(BaseModulo):
    """Class representing signed modulo operations."""

    SIGNED = True

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the signed modulo based on the given constant values."""
        return Constant(self._world, BaseModulo.mod([constant.signed for constant in operands]), max([constant.size for constant in operands]))  # type: ignore

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for SignedModulo."""
        return visitor.visit_signed_mod(self)


class UnsignedModulo(BaseModulo):
    """Class representing unsigned modulo operations."""

    SIGNED = False

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the unsigned modulo based on the given constant values."""
        return Constant(self._world, BaseModulo.mod([constant.unsigned for constant in operands]), max([constant.size for constant in operands]))  # type: ignore

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for SignedModulo."""
        return visitor.visit_unsigned_mod(self)


class BaseMultiplication(ArithmeticOperation, CommutativeOperation, AssociativeOperation, ABC):
    """Base interface for signed and unsigned Multiplications."""

    SYMBOL = "*"

    @staticmethod
    def prod(iterable: Iterable[Union[float, int]]) -> Union[float, int]:
        """Perform multiplication over all elements of the iterable."""
        return functools.reduce(operator.mul, iterable, 1)

    @dirty
    def factorize(self) -> Optional[Operation]:
        """Nothing to factorize."""
        pass


class SignedMultiplication(BaseMultiplication):
    """Class representing signed multiplication operations."""

    SIGNED = True

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the signed multiplication based on the given constant values."""
        return Constant(
            self._world, BaseMultiplication.prod([constant.signed for constant in operands]), max([constant.size for constant in operands])
        )

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for SignedMultiplication."""
        return visitor.visit_signed_mul(self)


class UnsignedMultiplication(BaseMultiplication):
    """Class representing unsigned multiplication operations."""

    SIGNED = False

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the unsigned multiplication based on the given constant values."""
        return Constant(
            self._world,
            BaseMultiplication.prod([constant.unsigned for constant in operands]),
            max([constant.size for constant in operands]),
        )

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for UnsignedMultiplication."""
        return visitor.visit_unsigned_mul(self)


M = TypeVar("M", bound=BaseMultiplication)


class BaseAddSub(ArithmeticOperation, ABC):
    """Base interface for common operations between Addition and Subtraction."""

    @dirty
    def factorize(self) -> Optional[Operation]:
        """Factorize out common factors utilizing multiplication."""
        if (result := self._factorize(SignedMultiplication)) is None:
            result = self._factorize(UnsignedMultiplication)
        elif (result2 := self._factorize(UnsignedMultiplication)) is not None:
            result = result2

        if result is not None and len(result.operands) == 1:
            result = result.operands[0]  # type: ignore
        return result

    def _factorize(self, mul_type: Type[M]) -> Optional[BaseAddition]:
        """
        Factorize out common factors utilizing M multiplication where M is unsigned or signed.

        e.g. 3*x + 2*x + 4 = x*(3+2) + 4
        """
        if (result := self._find_common_factor(mul_type)) is None:
            return None

        common_factor, involved_operands, uninvolved_operands = result

        # Find the remaining terms after a factorization
        # e.g. [2*x, 3*x, 4*x] -> [2, 3, 4]
        remaining_terms = []
        for operand in involved_operands:  # type: Operation
            operand = operand.copy_tree()
            operand.remove_operand(common_factor)
            remaining_terms.append(operand if len(operand.operands) > 1 else operand.operands[0])

        # Build factorized subexpr tree
        terms_addition = self.copy()
        terms_addition.replace_operands(remaining_terms)
        factorized_subexpr = involved_operands[0].copy()
        factorized_subexpr.add_operand(common_factor)
        factorized_subexpr.add_operand(terms_addition)

        new_expr = self.copy()
        new_expr.add_operand(factorized_subexpr)
        for uninvolved_operand in uninvolved_operands:
            new_expr.add_operand(uninvolved_operand)
        return new_expr  # type: ignore

    def _find_common_factor(self, mul_type: Type[M]) -> Optional[Tuple[WorldObject, List[BaseMultiplication], List[WorldObject]]]:
        """
        Identify a common factor between the operands to be factorized.

        e.g. 2*x + 8 + 3 + 6*a -> (2, (2*x, 8, 6*a))
        e.g. 2*x + 3*x + 6 -> (x, (2*x, 3*x))
        """
        # Filter what we want to simplify next step.
        factors: List[WorldObject] = []
        factorizing_operands: List[WorldObject] = []
        non_factorizing_operands: List[WorldObject] = []
        for op in self.operands:
            if isinstance(op, Variable):
                factors.append(op)
                factorizing_operands.append(op)
            elif isinstance(op, mul_type):
                mul_op: BaseMultiplication = op
                factors.extend(mul_op.operands)
                factorizing_operands.append(mul_op)
            else:
                non_factorizing_operands.append(op)

        factorizations = self._find_factorizations(factors, factorizing_operands, mul_type)

        # Sort by factor with most operands containing it
        sorted_results = sorted(factorizations, key=lambda triple: len(triple[1]), reverse=True)

        if len(sorted_results) > 0:
            triple = sorted_results[0]
            triple[2].extend(non_factorizing_operands)
            return triple

        return None

    def _find_factorizations(
        self, factors: List[WorldObject], factorizing_exprs: List[WorldObject], mul_type: Type[M]
    ) -> List[Tuple[WorldObject, List[BaseMultiplication], List[WorldObject]]]:
        """
        Find all factorizations given a list of factors and the list of expressions to factorize from.

        e.g. Given [2, x] and [2*x, 2*w, x], returns [(2, [2*x, 2*w], [x]), (x, [2*x, x], [2*w])]
        """
        factorizations = []
        for factor in factors:
            involved_operands: List[BaseMultiplication] = []
            uninvolved_operands: List[WorldObject] = []
            for op in factorizing_exprs:
                if isinstance(op, Variable) and op.world.compare(factor, op):
                    mul_one = mul_type(self._world)
                    mul_one.add_operand(op)
                    mul_one.add_operand(Constant(self._world, 1, op.size))
                    involved_operands.append(mul_one)
                elif isinstance(op, mul_type):
                    mul_op: BaseMultiplication = op
                    involved = False
                    for term in mul_op.operands:
                        if mul_op.world.compare(factor, term):
                            involved = True
                            involved_operands.append(mul_op)
                            break
                    if not involved:
                        uninvolved_operands.append(mul_op)
                else:
                    uninvolved_operands.append(op)

            # Factorization out of only one operand not useful.
            if len(involved_operands) > 1:
                factorizations.append((factor, involved_operands, uninvolved_operands))

        return factorizations


class BaseAddition(BaseAddSub, CommutativeOperation, AssociativeOperation):
    """Base interface for signed and unsigned Additions."""

    SYMBOL = "+"


class SignedAddition(BaseAddition):
    """Class representing signed addition operations."""

    SIGNED = True

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the signed addition based on the given constant values."""
        return Constant(self._world, sum([constant.signed for constant in operands]), max([constant.size for constant in operands]))

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for SignedAddition."""
        return visitor.visit_signed_add(self)


class UnsignedAddition(BaseAddition):
    """Class representing unsigned addition operations."""

    SIGNED = False

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the unsigned addition based on the given constant values."""
        return Constant(self._world, sum([constant.unsigned for constant in operands]), max([constant.size for constant in operands]))

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for UnsignedAddition."""
        return visitor.visit_unsigned_add(self)


class BaseSubtraction(BaseAddSub, OrderedOperation):
    """Base interface for signed and unsigned Subtractions."""

    SYMBOL = "-"

    @staticmethod
    def sub(iterable: Iterable[Union[float, int]]) -> Union[float, int]:
        """Perform subtraction over all elements of the iterable."""
        return functools.reduce(operator.sub, iterable)


class SignedSubtraction(BaseSubtraction):
    """Class representing signed Subtraction operations."""

    SIGNED = True

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the signed addition based on the given constant values."""
        return Constant(
            self._world, BaseSubtraction.sub([constant.signed for constant in operands]), max([constant.size for constant in operands])
        )

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for SignedSubtraction."""
        return visitor.visit_signed_sub(self)


class UnsignedSubtraction(BaseSubtraction):
    """Class representing unsigned Subtraction operations."""

    SIGNED = False

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the unsigned addition based on the given constant values."""
        return Constant(
            self._world, BaseSubtraction.sub([constant.unsigned for constant in operands]), max([constant.size for constant in operands])
        )

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for UnsignedSubtraction."""
        return visitor.visit_unsigned_sub(self)
