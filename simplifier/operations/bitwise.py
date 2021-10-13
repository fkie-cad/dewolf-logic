"""Module implementing logic operations such as AND and OR."""
from __future__ import annotations

from abc import ABC
from functools import reduce
from itertools import combinations, product
from operator import and_, lshift, neg, or_, rshift, xor
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

from simplifier.common import T
from simplifier.operations.interface import AssociativeOperation, CommutativeOperation, OrderedOperation, UnaryOperation
from simplifier.util.decorators import dirty
from simplifier.world.nodes import Constant, Operation, Variable, WorldObject

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
        # remove duplicate terms from the operation
        for duplicate in self._get_duplicate_terms():
            self.remove_operand(duplicate)
        # eval to 0 if one operand is zero or there is a collision (e.g. a AND !a)
        if any([constant for constant in self.constants if constant.unsigned == 0]) or self._has_collision():
            self.replace_operands([Constant(self.world, 0, max([operand.size for operand in self.operands]))])
        # remove all constants of maximum value (e.g. x:8 and 11111111:8 = x:8)
        for constant in self.constants:
            if constant.unsigned == constant.maximum and len(self.operands) > 1:
                self.remove_operand(constant)

        # perform associative folding here.
        while True:
            self._promote_subexpression()
            changes = False
            for operand1, operand2 in combinations(self.operands, 2):
                if (
                    self._variable_fold(operand1, operand2)
                    or self._variable_fold(operand2, operand1)
                    or self._associative_fold(operand1, operand2)
                    or self._associative_fold(operand2, operand1)
                ):
                    changes = True
                    break
            if changes is False:
                break
        self._promote_subexpression()
        for duplicate in self._get_duplicate_terms():
            self.remove_operand(duplicate)
        if self._has_collision():
            self.replace_operands([Constant(self.world, 0, max([operand.size for operand in self.operands]))])

    def _variable_fold(self, term1, term2):
        """
        Fold a variable.

        eg. a & (a | b) = a and (a | b) & (a | b | c) = (a | b)
        """
        if not isinstance(term1, (BitwiseOr, BitwiseXor)):
            return False
        for operand in term1.operands:
            if self.world.compare(operand, term2):
                if isinstance(term1, BitwiseOr):
                    self.remove_operand(term1)
                else:  # BitwiseXor
                    term1.remove_operand(operand)
                    term1.negate(predecessors=[self])
                    term1._promote_single_operand()
                return True
            # I would like to remove this part after we simplify as long as nothing changes!
            if isinstance(operand, BitwiseAnd) and any(self.world.compare(term2, op) for op in operand.operands):
                for op in operand.operands:
                    if self.world.compare(term2, op):
                        operand.remove_operand(op)
                        operand._promote_single_operand()
                        return True

        if isinstance(term1, BitwiseOr) and isinstance(term2, BitwiseOr):
            for operand_2 in term2.operands:
                for operand_1 in term1.operands:
                    if self.world.compare(operand_1, operand_2):
                        break
                else:
                    break
            else:
                self.remove_operand(term1)
                return True
        return False

    def _associative_fold(self, term1: WorldObject, term2: WorldObject) -> bool:
        """Apply associative folding.

        Eg. a & (!a | b) = (a & b)
        """
        if not isinstance(term1, (BitwiseOr, BitwiseXor)) or (
            isinstance(term2, Operation) and term1.variable_count() < term2.variable_count()
        ):
            return False
        negated_term_2 = term2.operand if isinstance(term2, BitwiseNegate) else self.world.bitwise_negate(term2)
        for operand in term1.operands:
            if self.world.compare(operand, negated_term_2):
                term1.remove_operand(operand)
                term1._promote_single_operand()
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
        # remove duplicate terms from the operation
        for duplicate in self._get_duplicate_terms():
            self.remove_operand(duplicate)
        # evaluate to 0xffff.. if a term and its direct negation are part of the operation
        maximum = Constant.create_maximum_value(self.world, max([operand.size for operand in self.operands]))
        if self._has_collision() or maximum in self.constants:
            self.replace_operands([maximum])
        for constant in self.constants:
            # remove any zero constant, unless its the last operand
            if constant.unsigned == 0 and len(self.operands) > 1:
                self.remove_operand(constant)
        if keep_form:
            return
        # associative & variable folding: may change the form of this operation
        while True:
            self._promote_subexpression()
            changes = False
            for operand1, operand2 in combinations(self.operands, 2):
                if (
                    self._variable_fold(operand1, operand2)
                    or self._variable_fold(operand2, operand1)
                    or self._associative_fold(operand1, operand2)
                    or self._associative_fold(operand2, operand1)
                ):
                    changes = True
                    break
            if changes is False:
                break
        if self._has_collision():
            self.replace_operands([maximum])

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
        """
        if not isinstance(term1, (BitwiseAnd, BitwiseXor)) or (
            isinstance(term2, Operation) and term1.variable_count() < term2.variable_count()
        ):
            return False
        negated_term_2 = term2.operand if isinstance(term2, BitwiseNegate) else self.world.bitwise_negate(term2)
        negated_term_2_resolved = negated_term_2.copy_tree().dissolve_negation() if isinstance(negated_term_2, BitwiseNegate) else None
        for operand in term1.operands:
            if self.world.compare(operand, negated_term_2) or self.world.compare(operand, negated_term_2_resolved):
                if isinstance(term1, BitwiseAnd):
                    term1.remove_operand(operand)
                    term1._promote_single_operand()
                else:  # BitwiseXor
                    term1.remove_operand(operand)
                    term1.negate(predecessors=[self])
                    term1._promote_single_operand()
                return True

        if isinstance(term1, BitwiseAnd) and isinstance(term2, BitwiseAnd):
            unique_operands_term_1 = []
            common_terms = set()
            for operand_1 in term1.operands:
                for operand_2 in term2.operands:
                    if self.world.compare(operand_1, operand_2):
                        common_terms.add(operand_2)
                        break
                else:
                    unique_operands_term_1.append(operand_1)
            if common_terms:
                unique_operands_term_2 = [op for op in term2.operands if op not in common_terms]
                unique_term_2 = (
                    self.world.bitwise_and(unique_operands_term_2) if len(unique_operands_term_2) > 1 else unique_operands_term_2[0]
                )
                unique_term_1 = (
                    self.world.bitwise_and(unique_operands_term_1) if len(unique_operands_term_1) > 1 else unique_operands_term_1[0]
                )
                negated_term_2 = (
                    unique_term_2.operand if isinstance(unique_term_2, BitwiseNegate) else self.world.bitwise_negate(unique_term_2)
                )
                negated_term_2_resolved = (
                    negated_term_2.copy_tree().dissolve_negation() if isinstance(negated_term_2, BitwiseNegate) else None
                )
                if self.world.compare(unique_term_1, negated_term_2) or self.world.compare(unique_term_1, negated_term_2_resolved):
                    self.world.remove_operand(self, term1)
                    self.world.remove_operand(self, term2)
                    new_operand = self.world.bitwise_and(common_terms) if len(common_terms) > 1 else common_terms.pop()
                    self.world.add_operand(self, new_operand)
                    return True

        return False

    def _variable_fold(self, term1: WorldObject, term2: WorldObject) -> bool:
        """
        Fold a variable.

        eg. a | (a & b) = a and (a & b) | (a & b & c) = (a & b)
        """
        if not isinstance(term1, (BitwiseAnd, BitwiseXor)):
            return False
        for operand in term1.operands:
            if self.world.compare(operand, term2):
                if isinstance(term1, BitwiseAnd):
                    self.remove_operand(term1)
                else:  # BitwiseXor
                    term1.remove_operand(operand)
                    term1._promote_single_operand()
                return True

        if isinstance(term1, BitwiseAnd) and isinstance(term2, BitwiseAnd):
            for operand_2 in term2.operands:
                for operand_1 in term1.operands:
                    if self.world.compare(operand_1, operand_2):
                        break
                else:
                    break
            else:
                self.remove_operand(term1)
                return True

        return False


class BitwiseXor(BitwiseOperation, CommutativeOperation, AssociativeOperation):
    """Class implementing a bitwise xor operation."""

    SYMBOL = "^"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        new_value = reduce(xor, [constant.unsigned for constant in operands])
        return Constant(self.world, new_value, max([constant.size for constant in operands]))

    @dirty
    def fold(self, keep_form: bool = False):
        """
        Fold all terms in the operation.

        e.g. x ^ 2 ^ x = 2
        """
        removal_candidates: Dict[WorldObject, int] = {}
        for duplicate in self._get_duplicate_terms():
            found = False
            for candidate in removal_candidates.keys():
                if self.world.compare(candidate, duplicate):
                    removal_candidates[candidate] += 1
                    found = True
            if not found:
                removal_candidates[duplicate] = 1
        for (duplicate, count) in removal_candidates.items():
            for _ in range(count // 2):
                self.remove_operand(duplicate)
                self.remove_operand(duplicate)

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
