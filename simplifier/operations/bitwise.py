"""Module implementing logic operations such as AND and OR."""
from __future__ import annotations

from abc import ABC
from functools import reduce
from itertools import combinations, permutations, product
from operator import and_, lshift, neg, or_, rshift, xor
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

from simplifier.common import T
from simplifier.operations.interface import AssociativeOperation, CommutativeOperation, OrderedOperation, UnaryOperation
from simplifier.util.decorators import dirty
from simplifier.world.nodes import BitVector, Constant, Operation, Variable, WorldObject

if TYPE_CHECKING:
    from simplifier.visitor import WorldObjectVisitor


class BitwiseOperation(Operation, ABC):
    """Common base class for bitwise operations."""

    def _split_terms(self) -> Tuple[List[WorldObject], List[WorldObject]]:
        """Split the terms in a list of negated and non-negated terms."""
        negated_terms: List[WorldObject] = []
        non_negated_terms: List[WorldObject] = []
        for term in self.operands:
            if isinstance(term, BitwiseNegate):
                negated_terms.append(term.operand)
            else:
                non_negated_terms.append(term)
        return non_negated_terms, negated_terms

    def _get_duplicate_terms(self) -> Iterator[WorldObject]:
        """Yield all duplicate terms in the operands."""
        operands = self.operands
        for index, term in enumerate(operands):
            yield from [x for x in operands[index + 1 :] if x == term and self.world.compare(x, term)]

    def _has_collision(self) -> bool:
        """
        Check if there are any colliding terms in this operation.

        e.g. a & !a
        """
        operands, negated_operands = self._split_terms()
        for negated_operand in negated_operands:
            possible_collisions = [operand for operand in operands if operand == negated_operand]
            for possible_collision in possible_collisions:
                if self.world.compare(negated_operand, possible_collision):
                    return True
        return False

    def _get_suboperands(self, operation_type: Type[Operation]) -> List[Set[WorldObject]]:
        """
        Return a list of sets containing all subterms of the operation considering the given type.

        e.g. 1 | a | a&b | !x, BitwiseAnd = [{1}, {a}, {a, b}, {}]
        """
        suboperands: List[Set] = []
        if len(operands := self.operands) < 2:
            return []
        for operand in operands:
            if isinstance(operand, (Constant, Variable)):
                suboperands.append({operand})
            else:
                suboperands.append(set(operand.operands) if isinstance(operand, operation_type) else set())
        return suboperands

    def _get_common_suboperands(self, operation_type: Type[Operation]) -> Set[WorldObject]:
        """Return a set of all terms which are common to all operands, if any."""
        suboperands = self._get_suboperands(operation_type)
        common_suboperands = set()
        if suboperands:
            for common in suboperands[0]:
                # Must manually iterate as Operation.__hash__ is unique, but we want to find duplicate objects
                if all([any([self.world.compare(common, op) for op in op_set]) for op_set in suboperands[1:]]):
                    common_suboperands.add(common)
        return common_suboperands

    def _get_negated_operands(self, level: Optional[int]):
        """
        Return the negation of each operand.

        If an operand is a BitVector, we do not want to negate the BitVector for all formulas, only the this one!
        """
        for op in self.operands:
            if isinstance(op, Operation):
                yield op.negate(level)
            else:
                yield op.negate(predecessors=[self])

    def _common_negation_for_or_and(self, new_op: Union[BitwiseAnd, BitwiseOr], level: Optional[int] = None) -> WorldObject:
        """Negates BitwiseOr and BitwiseAnd conditions (helper function)."""
        if level == 0:
            return super().negate(0)
        new_level = level - 1 if level else None
        for neg_op in self._get_negated_operands(new_level):
            self.world.add_operand(new_op, neg_op)
        self.world.replace(self, new_op)
        return new_op

    def _remove_duplicated_terms(self):
        """Remove all duplicated terms in the operands."""
        for duplicate in self._get_duplicate_terms():
            self.remove_operand(duplicate)

    def _simple_folding(self):
        """Apply all the simple folding strategies, i.e., removing duplicates, collisions and simplifying zeros and max-constants."""
        pass

    def is_operand(self, operand: WorldObject) -> bool:
        """Check whether the given operand is an operand of the operation."""
        for op in self.operands:
            if self.world.compare(operand, op):
                return True
        return False

    def replace_term_by(self, replacement_term, value: Constant) -> bool:
        """Return whether we replaced the given term by the given value somewhere in the formula."""
        changes = False
        for operand in self.operands:
            if self.world.compare(operand, replacement_term):
                self.world.replace_operand(self, operand, value)
                self._simple_folding()
                changes = True
                break
        for operand in self.operands:
            if isinstance(operand, BitwiseOperation):
                changes = operand.replace_term_by(replacement_term, value) | changes
            elif isinstance(operand, BitwiseNegate):
                neg_operand = operand.operand
                if self.world.compare(neg_operand, replacement_term):
                    self.world.replace_operand(self, operand, operand.eval([value]))
                    changes = True
                elif isinstance(neg_operand, BitwiseOperation):
                    changes = neg_operand.replace_term_by(replacement_term, value) | changes

        self._simple_folding()
        if len(self.operands) == 1 and isinstance(self, (CommutativeOperation, AssociativeOperation)):
            self._promote_single_operand()
            return True
        return changes


class BitwiseAnd(BitwiseOperation, CommutativeOperation, AssociativeOperation):
    """Class representing a bitwise and operation."""

    SYMBOL = "&"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        new_value = reduce(and_, [constant.unsigned for constant in operands])
        return Constant(self.world, new_value, max([constant.size for constant in operands]))

    @dirty
    def fold(self, keep_form: bool = False):
        """
        Fold all terms in the operation.

        a & !a = 0
        a & a = a
        a & 0 = 0
        a@8 & 11111111 = a
        """
        self._simple_folding()

        # perform associative folding here.
        has_folded = True
        while has_folded:
            self._promote_subexpression()
            has_folded = False
            for operand1, operand2 in permutations(self.operands, 2):
                if self._associative_fold(operand1, operand2):
                    has_folded = True
                    self._simple_folding()
                    break

        self._promote_subexpression()
        if self._has_collision():
            self.replace_operands([Constant(self.world, 0, max([operand.size for operand in self.operands]))])

    def _simple_folding(self):
        """Apply all the simple folding strategies, i.e., removing duplicates, collisions and simplifying zeros and max-constants."""
        self._remove_duplicated_terms()
        # eval to 0 if one operand is zero or there is a collision (e.g. a AND !a)
        if any([constant for constant in self.constants if constant.unsigned == 0]) or self._has_collision():
            self.replace_operands([Constant(self.world, 0, max([operand.size for operand in self.operands]))])
        # remove all constants of maximum value (e.g. x:8 and 11111111:8 = x:8)
        for constant in self.constants:
            if constant.unsigned == constant.maximum and len(self.operands) > 1:
                self.remove_operand(constant)

    def _associative_fold(self, term1: WorldObject, term2: WorldObject):
        """
        Associative folding.

        If one operand is also contained in the other operand, replace it by all 1s because it must be all 1s to fulfill the formula.
            e.g. a & (a | b) = a and (a | b) & (a | b | c) = (a | b)
            e.g. a & (!a | b) = (a & b)
        """
        if not isinstance(term2, BitwiseOperation) or (isinstance(term1, Operation) and term1.variable_count() > term2.variable_count()):
            return False
        if term2.replace_term_by(term1, Constant.create_maximum_value(self.world, term1.size)):
            return True
        if isinstance(term1, BitwiseNegate) and term2.replace_term_by(term1.operand, Constant(self.world, 0, term1.size)):
            return True

        if isinstance(term1, BitwiseOr) and isinstance(term2, BitwiseOr):
            for operand_1 in term1.operands:
                if not term2.is_operand(operand_1):
                    return False
            # term1 is sub-term of term 2
            self.remove_operand(term2)
            return True
        return False

    @dirty
    def factorize(self) -> Optional[BitwiseOr]:
        """
        Eliminate common subterms.

        e.g. (x | a) & ( x | b) => x | (a & b)
        """
        if not (common_suboperands := self._get_common_suboperands(BitwiseOr)):
            return None
        and_operation = self.world.bitwise_and()
        for operand in self.children:
            if isinstance(operand, BitwiseOr):
                for common_suboperand in common_suboperands:
                    for suboperand in operand.operands:
                        # We need to remove `suboperand`, not `common_suboperand` as they
                        # are distinct, may have different ASTs, and currently have
                        # different __hash__es
                        if self.world.compare(common_suboperand, suboperand):
                            operand.remove_operand(suboperand)
                            break
                and_operation.add_operand(operand)
                operand.simplify()
        or_operation = self.world.bitwise_or(*common_suboperands, and_operation)
        and_operation.simplify()
        return or_operation

    def _replace_operands(self, op1: WorldObject, op2: BitwiseOperation, operation: BitwiseOperation):
        """Replace op2 with operation or a new variable that doesn't contain op1."""
        if len(op2.operands) == 2:
            # special case: we only need the variable, throw away BitwiseOperation.
            new = op2.operands[0] if self.world.compare(op2.operands[1], op1) else op2.operands[1]
            self.replace_operand(op2, new)
        else:
            for op in op2.operands:
                if not self.world.compare(op, op1):
                    operation.add_operand(op)
            self.replace_operand(op2, operation)

    def negate(self, level: Optional[int] = None, predecessors: Optional[List[Operation]] = None) -> WorldObject:
        """Negate the given condition."""
        return self._common_negation_for_or_and(BitwiseOr(self.world), level)

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for BitwiseAnd."""
        return visitor.visit_bitwise_and(self)


class BitwiseOr(BitwiseOperation, CommutativeOperation, AssociativeOperation):
    """Class representing a bitwise or operation."""

    SYMBOL = "|"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        new_value = reduce(or_, [constant.unsigned for constant in operands])
        return Constant(self.world, new_value, max([constant.size for constant in operands]))

    @dirty
    def fold(self, keep_form: bool = False):
        """
        Apply folding to all terms in the operation.

        e.g. a | !a = 0xffff...
        e.g. a | a = a
        e.g. a | 0 = a
        e.g. a | 0xffff... = 0xffff...
        """
        self._simple_folding()

        # associative & variable folding: may change the form of this operation
        has_folded = True
        while has_folded:
            self._promote_subexpression()
            has_folded = False
            for operand1, operand2 in permutations(self.operands, 2):
                if self._associative_fold(operand1, operand2):
                    has_folded = True
                    self._simple_folding()
                    break

        self._promote_subexpression()
        if self._has_collision():
            self.replace_operands([Constant.create_maximum_value(self.world, max([operand.size for operand in self.operands]))])

    def _simple_folding(self):
        """Apply all the simple folding strategies, i.e., removing duplicates, collisions and simplifying zeros and max-constants."""
        self._remove_duplicated_terms()
        # evaluate to 0xffff.. if a term and its direct negation are part of the operation
        maximum = Constant.create_maximum_value(self.world, max([operand.size for operand in self.operands]))
        if self._has_collision() or maximum in self.constants:
            self.replace_operands([maximum])
        for constant in self.constants:
            # remove any zero constant, unless its the last operand
            if constant.unsigned == 0 and len(self.operands) > 1:
                self.remove_operand(constant)

    @dirty
    def factorize(self) -> Optional[BitwiseAnd]:
        """
        Eliminate common subterms.

        e.g. (x & a) | (x & b) => x & (a | b)
        """
        if not (common_suboperands := self._get_common_suboperands(BitwiseAnd)):
            return None
        or_operation = self.world.bitwise_or()
        for operand in self.operands:
            if isinstance(operand, BitwiseAnd):
                for common_suboperand in common_suboperands:
                    for suboperand in operand.operands:
                        # We need to remove `suboperand`, not `common_suboperand` as they
                        # are distinct, may have different ASTs, and currently have
                        # different __hash__es
                        if self.world.compare(common_suboperand, suboperand):
                            operand.remove_operand(suboperand)
                            break
                or_operation.add_operand(operand)
                operand.simplify()
        and_operation = self.world.bitwise_and(*common_suboperands, or_operation)
        or_operation.simplify()
        return and_operation

    def negate(self, level: Optional[int] = None, predecessors: Optional[List[Operation]] = None) -> WorldObject:
        """Negate the given condition."""
        return self._common_negation_for_or_and(BitwiseAnd(self.world), level)

    def get_equivalent_and_condition(self, conjunction: BitwiseAnd) -> BitwiseAnd:
        """
        Create the new And-condition that is equivalent to self and replace self by it.

        - conjunction must be a term of the BitwiseOr formula self.

        self = (a1 & a2 & .. & ak) | l1 | l2 | ... | lh, where conjunction = (a1 & a2 & .. & ak), remaining_conditions = [l1, l2, ..., lh]
        and each li is a condition
        -> new_condition = (a1 | l1 | ... |lh) & (a2 | l1 | ... |lh) & ... & (ak | l1 | ... |lh)
        """
        conditions = [
            self.world.bitwise_or(arg, *(cond.copy_tree() for cond in self.operands if hash(cond) != hash(conjunction)))
            for arg in conjunction.operands
        ]
        new_condition = self.world.bitwise_and(*conditions)
        for cond in new_condition.operands:
            cond.simplify(keep_form=True)
        self.world.replace(self, new_condition)
        return new_condition

    def get_conjunction(self) -> Optional[BitwiseAnd]:
        """If the Or-formula has a literal that is an conjunction, then return it, otherwise return None."""
        for child in self.operands:
            if isinstance(child, BitwiseNegate):
                if resolved_neg := child.dissolve_negation():
                    child = resolved_neg
            if isinstance(child, BitwiseAnd):
                return child
        return None

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for BitwiseOr."""
        return visitor.visit_bitwise_or(self)

    def _associative_fold(self, term1: WorldObject, term2: WorldObject) -> bool:
        """Apply associative folding.

        Eg.
            a | (!a & b) = a | b
            (!a & b) | a = a | b
            !a | (a & b) = !a | b
            (a&c) | (!(a&c) & b) = (a&c) | b
        eg. a | (a & b) = a and (a & b) | (a & b & c) = (a & b)
        """
        if not isinstance(term2, BitwiseOperation) or (isinstance(term1, Operation) and term1.variable_count() > term2.variable_count()):
            return False
        if term2.replace_term_by(term1, Constant(self.world, 0, term1.size)):
            return True
        if isinstance(term1, BitwiseNegate) and term2.replace_term_by(term1.operand, Constant.create_maximum_value(self.world, term1.size)):
            return True

        if isinstance(term1, BitwiseAnd) and isinstance(term2, BitwiseAnd):
            for operand_1 in term1.operands:
                if not term2.is_operand(operand_1):
                    return False
            # term1 is sub-term of term 2
            self.remove_operand(term2)
            return True
        return False


class BitwiseXor(BitwiseOperation, CommutativeOperation, AssociativeOperation):
    """Class implementing a bitwise xor operation."""

    SYMBOL = "^"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        new_value = reduce(xor, [constant.unsigned for constant in operands])
        return Constant(self.world, new_value, max([constant.size for constant in operands]))

    def _get_duplicate_classes(self) -> Iterator[Tuple[WorldObject, List[WorldObject]]]:
        """Yield all duplicate classes of the operand."""
        operands: List[Tuple[int, WorldObject]] = [(idx, operand) for idx, operand in enumerate(self.operands)]
        sorted_operands: Set[int] = set()
        for index, term in operands:
            if index in sorted_operands:
                continue
            duplicate_class = []
            for idx, op in operands[index + 1 :]:
                if idx not in sorted_operands and self.world.compare(op, term):
                    sorted_operands.add(idx)
                    duplicate_class.append(op)
            yield term, duplicate_class

    @dirty
    def fold(self, keep_form: bool = False):
        """
        Fold all terms in the operation.

        e.g. x ^ 2 ^ x = 2
        e.g. x ^ y ^ 0 = x ^ y
        e.g. x ^ y ^ 1 = !(x ^ y)
        e.g. 1 ^ 1 = 0, 0 ^ 0 = 0, 0 ^ 1 = 1
        """
        self._simple_folding()

    def _simple_folding(self):
        """Apply all the simple folding strategies, i.e., removing duplicates, simplify 0 and max-value constants."""
        # remove duplicates
        operation_size = self.size
        for operand, duplicate_class in self._get_duplicate_classes():
            for op in duplicate_class:
                self.remove_operand(op)
            if (len(duplicate_class) + 1) % 2 == 0:
                self.remove_operand(operand)
        if len(self.operands) == 0:
            self.world.add_operand(self, Constant(self.world, 0, operation_size))
            return
        max_const = Constant.create_maximum_value(self.world, operation_size)
        for const_operand in (op for op in self.operands if isinstance(op, Constant)):
            if const_operand.unsigned == 0:
                self.remove_operand(const_operand)
            if const_operand == max_const:
                self.remove_operand(const_operand)
                self.negate()

    @dirty
    def factorize(self) -> None:
        """Factorize the xor operation."""
        pass

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for BitwiseXor."""
        return visitor.visit_bitwise_xor(self)


class BitwiseNegate(UnaryOperation):
    """Class representing a bitwise negation in a world object."""

    SYMBOL = "~"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        if len(operands) != 1:
            raise ValueError(f"An UnaryOperation should have exactly one operand.")
        return Constant(self.world, -operands[0].signed - 1, operands[0].size)

    @dirty
    def fold(self, keep_form: bool = False):
        """Fold all terms in the operation."""
        operand = self.operand
        if isinstance(operand, BitwiseNegate):
            self.remove_operand(operand)
            self.world.substitute(self, operand.operand)
            operand.remove_operand(operand.operand)

    @dirty
    def factorize(self) -> None:
        """Factorize the bitwise negation."""
        pass

    def negate(self, level: Optional[int] = None, predecessors: Optional[List[Operation]] = None) -> WorldObject:
        """Negate the given condition."""
        negated_operand = self.operand
        self.world.substitute(self, self.operand)
        return negated_operand

    def dissolve_negation(self) -> Optional[WorldObject]:
        """
        Dissolves the negation of the BitwiseNegate operation, if possible.

        - More precisely, if the one operand of the BitwiseNegate is also an operation, then the we negate the operands, i.e., the AND, OR,
        or NEG and remove the outer negation.
        - If the operand is not an operation, then we can replace self by an equivalent term that is not a BitwiseNegate.
        - In contrast to negate, self if equivalent to the return value.
        """
        if isinstance(self.operand, Operation):
            self.operand.negate()
            return self.negate()
        return None

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for BitwiseNegate."""
        return visitor.visit_bitwise_negate(self)


class ShiftOperation(BitwiseOperation, OrderedOperation):
    """Class representing the base for all shift and rotate operations."""

    @dirty
    def fold(self, keep_form: bool = False):
        """Fold all terms in the operation."""
        pass

    @dirty
    def factorize(self) -> None:
        """Factorize the bitwise negation."""
        pass


class ShiftLeft(ShiftOperation):
    """Class representing a bitwise shift left in a world object."""

    SYMBOL = "<<"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        new_value = reduce(lshift, [constant.unsigned for constant in operands])
        return Constant(self.world, new_value, max([constant.size for constant in operands]))

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for ShiftLeft."""
        return visitor.visit_shift_left(self)


class ShiftRight(ShiftOperation):
    """Class representing a bitwise shift right in a world object."""

    SYMBOL = ">>"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        new_value = reduce(rshift, [constant.unsigned for constant in operands])
        return Constant(self.world, new_value, max([constant.size for constant in operands]))

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for ShiftRight."""
        return visitor.visit_shift_right(self)


class RotateLeft(ShiftOperation):
    """Class representing a left rotate in a world object."""

    SYMBOL = "rotl"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        pass

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for RotateLeft."""
        return visitor.visit_rotate_left(self)


class RotateRight(ShiftOperation):
    """Class representing a right rotate in a world object."""

    SYMBOL = "rotr"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        pass

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for RotateRight."""
        return visitor.visit_rotate_right(self)
