"""Module implementing logic operations such as AND and OR."""
from __future__ import annotations

from abc import ABC, abstractmethod
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

    def _remove_duplicated_terms(self):
        """Remove all duplicated terms in the operands."""
        for duplicate in self._get_duplicate_terms():
            self.remove_operand(duplicate)

    def _get_common_and_unique_operands(
        self, term1: BitwiseOperation, term2: BitwiseOperation
    ) -> Tuple[Set[WorldObject], Set[WorldObject], Set[WorldObject]]:
        """Return the list of common operands of the given operations as well as sets with their respective unique operands."""
        operands_1 = set(term1.operands)
        operands_2 = set(term2.operands)
        common_operands = set()
        for operand_1, operand_2 in product(term1.operands, term2.operands):
            if self.world.compare(operand_1, operand_2):
                common_operands.add(operand_1)
                operands_1.remove(operand_1)
                operands_2.remove(operand_2)
        return common_operands, operands_1, operands_2

    def _simple_folding(self):
        """
        Do basic constant folding on this operation that is not called recursively.

        Helper function for BitwiseAnd/Or/Xor/Neg-folding.
        Only implemented if such a folding strategy exists, otherwise we do nothing.
        """
        pass

    def replace_term_by(self, replacement_term: WorldObject, value: Constant) -> bool:
        """Return whether we replaced the given term by the given value somewhere in the formula."""
        changes = self._replace_only_operands_by_term(replacement_term, value)
        changes = self._replace_recursively_in_operands_by_term(replacement_term, value) | changes
        self._simple_folding()
        changes = self._try_to_promote_single_operand() | changes
        return changes

    def _replace_only_operands_by_term(self, replacement_term: WorldObject, value: Constant) -> bool:
        """Replace each operand that is equivalent to the replacement term by the given value. Return True if an operand is replaced."""
        for operand in self.operands:
            if self.world.compare(operand, replacement_term):
                self.world.replace_operand(self, operand, value)
                self._simple_folding()
                return True
        return False

    def _replace_recursively_in_operands_by_term(self, replacement_term: WorldObject, value: Constant) -> bool:
        """
        Replace in each operand every sub-operand that is equivalent to the replacement term by the given value.

        Return True if any operand is replaced.
        """
        changes = False
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
        return changes

    def _try_to_promote_single_operand(self) -> bool:
        """Promote single operand if possible and return whether it was successfully."""
        if len(self.operands) == 1 and isinstance(self, (CommutativeOperation, AssociativeOperation)):
            self._promote_single_operand()
            return True
        return False


class CommonBitwiseAndOr(BitwiseOperation, CommutativeOperation, AssociativeOperation, ABC):
    """Base class for BitwiseAnd and BitwiseOr."""

    def negate(self, level: Optional[int] = None, predecessors: Optional[List[Operation]] = None) -> WorldObject:
        """Negate the given condition."""
        new_op = self._negated_class(self.world)
        if level == 0:
            return super().negate(0)
        new_level = level - 1 if level else None
        for neg_op in self._get_negated_operands(new_level):
            self.world.add_operand(new_op, neg_op)
        self.world.replace(self, new_op)
        return new_op

    def get_equivalent_condition_of_negated_type(self, operation: Union[BitwiseAnd, BitwiseOr]) -> Union[BitwiseAnd, BitwiseOr]:
        """
        Create the new condition that is equivalent to self and replace self by it.

        - if self is Or-operation and operation is a conjunction
            self = A | l1 | l2 | ... | lh, where A = (a1 & a2 & ... & ak),
            remaining_conditions = [l1, l2, ..., lh] and each li is a condition
            -> new_condition = (a1 | l1 | ... |lh) & (a2 | l1 | ... |lh) & ... & (ak | l1 | ... |lh)
        -if self is And-operation and operation is a disjunction
            self = A & l1 & l2 & ... & lh, where A = (a1 | a2 | ... | ak),
            remaining_conditions = [l1, l2, ..., lh] and each li is a condition
            -> new_condition = (a1 & l1 & ... & lh) | (a2 & l1 & ... & lh) | ... | (ak & l1 & ... & lh)
        """
        conditions = []
        for literal in operation.operands:
            new_operands = [literal, *(cond.copy_tree() for cond in self.operands if hash(cond) != hash(operation))]
            conditions.append(self.__class__(self.world).replace_operands(new_operands))
        new_condition = self._negated_class(self.world).replace_operands(conditions)
        for cond in new_condition.operands:
            cond.simplify(keep_form=True)
        self.world.replace(self, new_condition)
        return new_condition

    def get_junction(self):
        """If the formula has a literal that is a junction of negated type, then return it, otherwise return None."""
        for child in self.operands:
            if isinstance(child, BitwiseNegate):
                if resolved_neg := child.dissolve_negation():
                    child = resolved_neg
            if isinstance(child, self._negated_class):
                return child
        return None

    @property
    @abstractmethod
    def _negated_class(self) -> Union[Type[BitwiseAnd], Type[BitwiseOr]]:
        """Return BitwiseAnd if the operation is an BitwiseOr and BitwiseOr if the operation is an BitwiseAnd."""

    def _common_fold(self, const_for_collision: Constant):
        """Fold all terms in the given operation."""
        self._simple_folding()
        self._associative_folding()

        self._promote_subexpression()
        if self._has_collision():
            self.replace_operands([const_for_collision])

    def _associative_folding(self):
        """Apply associative folding on all pairs of operands of the given operation as long as no pair can be simplified."""
        while True:
            self._promote_subexpression()
            for operand1, operand2 in permutations(self.operands, 2):
                if self._associative_fold_of_pair(operand1, operand2):
                    self._simple_folding()
                    break
            else:
                return

    @abstractmethod
    def _associative_fold_of_pair(self, term1: WorldObject, term2: WorldObject) -> bool:
        """Associative folding for BitwiseAnd reps. BitwiseOr of the given operands."""

    def _common_associative_fold_of_pair(
        self, sub_term: WorldObject, term: WorldObject, vanishing_const: Constant, dominating_const: Constant
    ) -> bool:
        """
        Associative folding strategy of the given two operands of an BitwiseAnd resp. BitwiseOr operation.

        We return True when we replace sub_term in term by any constant.

        The vanishing constant is the constant that we remove if it is an operand.
        The dominating constant is the constant that makes all other operands obsolete,
            i.e., we can remove all other operands if it is an operand

        1. We replace sub_term in term by the vanishing_constant,
            i.e., self: Or sub_term = a and term_2 = a & b -> term_2 = False & b;
                  self: And sub_term = a and term_2 = a | b -> term_2 = True | b
        2. We replace !sub_term in term by the dominating_constant if sub_term is a bitwise negate,
            i.e., self: Or sub_term = !a and term_2 = a & b -> term_2 = True & b;
                  self: And sub_term = !a and term_2 = a | b -> term_2 = False | b
        3. We split sub_term and term in 3 sets, common-operands, and unique-operands of sub_term and term.
            - If each operand from sub_term is also an operand of term, then remove term.
            - If the one unique part is the negation of the other unique part, then remove term and sub_term and add the term consisting
              of the common operands.
        """
        neg_class: Union[Type[BitwiseAnd], Type[BitwiseOr]] = self._negated_class
        if not isinstance(term, BitwiseOperation) or not self._can_be_contained_in(sub_term, term):
            return False
        if term.replace_term_by(sub_term, vanishing_const):
            return True

        if (negated_sub_term := self._get_negated_term(sub_term)) is None:
            return False
        if term.replace_term_by(negated_sub_term, dominating_const):
            return True

        if isinstance(sub_term, neg_class) and isinstance(term, neg_class):
            common_operands, unique_operands_sub_term, unique_operands_term = self._get_common_and_unique_operands(sub_term, term)  # type: ignore
            if len(unique_operands_sub_term) == 0:
                self.remove_operand(term)
                return True
            junction_unique_sub_term = self._get_junction_of_operands(unique_operands_sub_term)
            junction_unique_term = self._get_junction_of_operands(unique_operands_term)
            if self._are_complementary(junction_unique_sub_term, junction_unique_term):
                new_operand = self._get_junction_of_operands(common_operands)
                self._replace_operands([sub_term, term], new_operand)
                return True
        return False

    def _replace_operands(self, operands: List[WorldObject], new_operand: WorldObject):
        """Replace the operands by the new operand."""
        for operand in operands:
            self.remove_operand(operand)
        self.add_operand(new_operand)

    def _get_junction_of_operands(self, unique_operands: Set[WorldObject]):
        """Return junction of the given list of operands."""
        if len(unique_operands) == 1:
            return unique_operands.pop()
        return self._negated_class(self.world).copy().replace_operands(unique_operands)

    def _common_factorize(self) -> Optional[Union[BitwiseOr, BitwiseAnd]]:
        """Factorize strategy of all operands of BitwiseAnd and BitwiseOr."""
        neg_class = self._negated_class
        if not (common_suboperands := self._get_common_suboperands(neg_class)):
            return None
        unique_operation_part = self.__class__(self.world)
        assert all(
            isinstance(operand, neg_class) for operand in self.children
        ), f"Since {self} has common suboperands, every operand must be of type {neg_class} or a BitVector."
        for operand in self.children:
            self._remove_common_operands(operand, common_suboperands)
            unique_operation_part.add_operand(operand)
            operand.simplify()
        new_operands = common_suboperands | {unique_operation_part}
        factorized_operation: Union[BitwiseAnd, BitwiseOr] = neg_class(self.world).replace_operands(new_operands)  # type: ignore
        unique_operation_part.simplify()
        return factorized_operation

    def _remove_common_operands(self, operand: Operation, common_suboperands: Set[WorldObject]):
        for common_suboperand in common_suboperands:
            for suboperand in operand.operands:
                if self.world.compare(common_suboperand, suboperand):
                    operand.remove_operand(suboperand)
                    break

    @staticmethod
    def _can_be_contained_in(sub_term: WorldObject, term: BitwiseOperation) -> bool:
        """Check whether sub_term can be contained in term."""
        return not isinstance(sub_term, Operation) or sub_term.variable_count() <= term.variable_count()

    def _get_negated_term(self, term: WorldObject) -> Optional[WorldObject]:
        """Get negation of term when it is a BitwiseNegate, BitwiseAnd or BitwiseOr."""
        if isinstance(term, BitwiseNegate):
            return term.operand
        if isinstance(term, BitwiseAnd):
            operands = [op.operand if isinstance(op, BitwiseNegate) else self.world.bitwise_negate(op) for op in term.operands]
            return self.world.bitwise_or(*operands)
        if isinstance(term, BitwiseOr):
            operands = [op.operand if isinstance(op, BitwiseNegate) else self.world.bitwise_negate(op) for op in term.operands]
            return self.world.bitwise_and(*operands)

        return None

    def _are_complementary(self, term1: WorldObject, term2: WorldObject) -> bool:
        """Check whether the given terms are complementary to each other."""
        return (isinstance(term1, BitwiseNegate) and self.world.compare(term1.operand, term2)) or (
            isinstance(term2, BitwiseNegate) and self.world.compare(term1, term2.operand)
        )


class BitwiseAnd(CommonBitwiseAndOr):
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
        self._common_fold(const_for_collision=Constant(self.world, 0, max([operand.size for operand in self.operands])))

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

    def _associative_fold_of_pair(self, sub_term: WorldObject, term: WorldObject):
        """
        Associative folding.

        If the operand sub_term is also contained in the other operand term, replace sub_term by all 1s because it must be all 1s to
        fulfill the formula.
            e.g. a & (a | b) = a and (a | b) & (a | b | c) = (a | b)
            e.g. a & (!a | b) = (a & b)
        Similar, if the negation of the sub_term is also contained in the other operand term, replace !sub_term by 0 because it must be 0
        to fulfill the formula.
            e.g. !a & (a | b) = !a & b
        """
        return self._common_associative_fold_of_pair(
            sub_term,
            term,
            vanishing_const=Constant.create_maximum_value(self.world, sub_term.size),
            dominating_const=Constant(self.world, 0, sub_term.size),
        )

    @dirty
    def factorize(self) -> Optional[BitwiseOr]:
        """
        Eliminate common subterms.

        e.g. (x | a) & ( x | b) => x | (a & b)
        """
        return self._common_factorize()  # type: ignore

    @property
    def _negated_class(self) -> Type[BitwiseOr]:
        """Return BitwiseOr, which corresponds to ~BitwiseAnd."""
        return BitwiseOr

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for BitwiseAnd."""
        return visitor.visit_bitwise_and(self)


class BitwiseOr(CommonBitwiseAndOr):
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
        self._common_fold(const_for_collision=Constant.create_maximum_value(self.world, max([operand.size for operand in self.operands])))

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

    def _associative_fold_of_pair(self, sub_term: WorldObject, term: WorldObject) -> bool:
        """Apply associative folding.

        If the operand sub_term is also contained in the other operand term, replace sub_term by 0 because term must only be fulfilled
        to fulfill the formula when sub_term is 0.
            e.g. (a&c) | (!(a&c) & b) = (a&c) | b
            e.g. a | (!a & b) = (a | b)
            e.g. a | (a & b) = a and (a & b) | (a & b & c) = (a & b)
        Similar, if the negation of the sub_term is also contained in the other operand term, replace !sub_term by all 1s because term must
        only be fulfilled to to fulfill the formula when sub_term 0s all 1s.
            e.g. !a | (a & b) = !a | b
        """
        return self._common_associative_fold_of_pair(
            sub_term,
            term,
            vanishing_const=Constant(self.world, 0, sub_term.size),
            dominating_const=Constant.create_maximum_value(self.world, sub_term.size),
        )

    @dirty
    def factorize(self) -> Optional[BitwiseAnd]:
        """
        Eliminate common subterms.

        e.g. (x & a) | (x & b) => x & (a | b)
        """
        return self._common_factorize()  # type: ignore

    @property
    def _negated_class(self) -> Type[BitwiseAnd]:
        """Return BitwiseAns, which corresponds to ~BitwiseOr."""
        return BitwiseAnd

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for BitwiseOr."""
        return visitor.visit_bitwise_or(self)


class BitwiseXor(BitwiseOperation, CommutativeOperation, AssociativeOperation):
    """Class implementing a bitwise xor operation."""

    SYMBOL = "^"

    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the operation with the given constant operands."""
        new_value = reduce(xor, [constant.unsigned for constant in operands])
        return Constant(self.world, new_value, max([constant.size for constant in operands]))

    def _get_duplicate_classes(self) -> Iterator[Tuple[WorldObject, List[WorldObject]]]:
        """
        Yield all duplicate classes of the operand.

        i.e. given the formula x ^ x ^ y ^ z ^ z ^ z it returns (x : [x]), (y: [0]), (z : [z,z])
        """
        operands: Set[Tuple[int, WorldObject]] = {(idx, operand) for idx, operand in enumerate(self.operands)}
        while operands:
            _, operand = operands.pop()
            duplicates_of_operand = set()
            for idx, unsorted_operand in operands:
                if self.world.compare(operand, unsorted_operand):
                    duplicates_of_operand.add((idx, unsorted_operand))
            operands -= duplicates_of_operand
            yield operand, [duplicate for _, duplicate in duplicates_of_operand]

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
        self._remove_duplicates()

        if len(self.operands) == 0:
            self.world.add_operand(self, Constant(self.world, 0, operation_size))
            return

        self._remove_redundant_constants(operation_size)

    def _remove_duplicates(self):
        """Remove duplicated operands, i.e., remove always pairs of equivalent operands."""
        for operand, duplicates_of_operand in self._get_duplicate_classes():
            for duplicate in duplicates_of_operand:
                self.remove_operand(duplicate)
            if (len(duplicates_of_operand) + 1) % 2 == 0:
                self.remove_operand(operand)

    def _remove_redundant_constants(self, operation_size: int):
        """Remove every 0 constant operand and if an operand is all 1s, remove it and negate the operation."""
        for const_operand in (op for op in self.operands if isinstance(op, Constant)):
            if const_operand.unsigned == 0:
                self.remove_operand(const_operand)
            if const_operand == Constant.create_maximum_value(self.world, operation_size):
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

    def _simple_fold(self):
        """Apply the simple folding strategy that removes double negation."""
        operand = self.operand
        if isinstance(operand, BitwiseNegate):
            self.remove_operand(operand)
            self.world.substitute(self, operand.operand)
            operand.remove_operand(operand.operand)

    @dirty
    def fold(self, keep_form: bool = False):
        """Fold all terms in the operation."""
        self._simple_fold()

    @dirty
    def factorize(self) -> None:
        """Factorize the bitwise negation."""
        pass

    def negate(self, level: Optional[int] = None, predecessors: Optional[List[Operation]] = None) -> WorldObject:
        """Negate the given condition."""
        negated_operand = self.operand
        if predecessors is None:
            self.world.substitute(self, self.operand)
        else:
            for pred in predecessors:
                self.world.replace_operand(pred, self, negated_operand)
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
