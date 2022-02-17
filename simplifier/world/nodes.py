"""Module defining the nodes types of World instances."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Iterable, List, Optional, TypeVar, Union

from simplifier.common import T
from simplifier.util.decorators import clean, dirty
from simplifier.world.graphs.interface import GraphNodeInterface

if TYPE_CHECKING:
    from simplifier.visitor import WorldObjectVisitor
    from simplifier.world.interface import WorldInterface
WorldType = TypeVar("WorldType", bound="WorldInterface")


class WorldObject(GraphNodeInterface, Generic[WorldType], ABC):
    """Common interface for all nodes contained in the graph of a World instance."""

    def __init__(self, world: WorldType):
        """Create an new operation with a reference to the world it is contained in."""
        self._world = world

    @property
    def world(self) -> WorldType:
        """Return the World this operation is defined in."""
        return self._world

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the size of the vector in bits."""

    def negate(self, level: Optional[int] = None, predecessors: Optional[List[Operation]] = None) -> WorldObject:
        """
        Negate the world object, by default this calls the add_negate function.

        - level tells us how often we call the negate recursively until we only add a negation in front.
          None means we do it as long as possible
        - predecessor tells us whether we do not negate the object overall. If it is None we negate it overall otherwise only on the edges
          between the predecessors and self.
        """
        if predecessors is None:
            predecessors = self.world.parent_operation(self)
        negated_operation = getattr(self.world, "bitwise_negate")(self)
        for pred in predecessors:
            self.world.remove_operand(pred, self)
            self.world.add_operand(pred, negated_operation)
        return negated_operation

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for WorldObjects. By default we do nothing."""
        return visitor.visit_world_object(self)

    def __str__(self) -> str:
        """Return a string representation of the current operation."""
        from simplifier.visitor import PrintVisitor

        return PrintVisitor().visit(self)

    def __repr__(self) -> str:
        """Return a string representation for debug purposes."""
        return str(self)


class BitVector(WorldObject, Generic[WorldType], ABC):
    """Class representing a bitvector of a given length."""

    def __init__(self, world: WorldType, size: int):
        """Initialize an BitVector based on the given size in bits."""
        super().__init__(world)
        self._size = size

    @property
    def size(self) -> int:
        """Return the size of the vector in bits."""
        return self._size

    @property
    def maximum(self) -> int:
        """Return the maximum value which can be represented."""
        return 2**self._size - 1

    def __eq__(self, other) -> bool:
        """Compare two BitVectors based on their representation."""
        return repr(self) == repr(other)

    def __hash__(self) -> int:
        """Hash an Bitvector based on its representation."""
        return hash(repr(self))


class BaseVariable(BitVector, ABC):
    """Class representing a Variable in a World instance."""

    def __init__(self, world: WorldType, name: str, size: int):
        """Initialize a new variable given an unique name and its size in bits."""
        super().__init__(world, size)
        self._name = name

    @property
    def name(self) -> str:
        """Return the unique name of the variable."""
        return self._name

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for Variable."""
        return visitor.visit_variable(self)

    def simplify(self):
        """Simplify the term defined by the variable."""
        from simplifier.operations import BitwiseNegate

        for node in list(self.world.iter_postorder(self)):
            if isinstance(node, BitwiseNegate):
                node.dissolve_negation()
        from simplifier.range_simplifier import RangeSimplifier

        for node in list(self.world.iter_postorder(self)):
            if isinstance(node, Operation):
                RangeSimplifier.simplify(node)
                from simplifier.operations.interface import AssociativeOperation, CommutativeOperation

                if isinstance(node, (AssociativeOperation, CommutativeOperation)):
                    node._promote_single_operand()
        for node in list(self.world.iter_postorder(self)):
            if isinstance(node, Operation):
                node.simplify()


class Variable(BaseVariable):
    """Class representing a Variable in a World instance."""

    def __repr__(self) -> str:
        """Return a debug representation of the variable."""
        return f'Variable("{self._name}", {self._size})'

    def copy(self) -> GraphNodeInterface:
        """Return a new instance of the same variable."""
        return Variable(self._world, self._name, self._size)


class TmpVariable(BaseVariable):
    """Class representing a Variable in a World instance."""

    def __repr__(self) -> str:
        """Return a debug representation of the variable."""
        return f'TmpVariable("{self._name}", {self._size})'

    def copy(self) -> GraphNodeInterface:
        """Return a new instance of the same variable."""
        return TmpVariable(self._world, self._name, self._size)


class Constant(BitVector):
    """Class representing a constant in a World instance."""

    def __init__(self, world: WorldType, value: Union[float, int], size: int):
        """Initialize an constant with the given value and size in bits."""
        super().__init__(world, size)
        value = value if value >= 0 or type(value) == float else value + (1 << self.size)
        # store a negative value as two's complement, but keep the original form if it is a floating point number
        self._value = value % (1 << self.size) if type(value) != float else value

    @classmethod
    def create_maximum_value(cls, world: WorldType, size: int) -> Constant:
        """Return the highest value constant representable with the given amount of bits."""
        return cls(world, (1 << size) - 1, size)

    @property
    def signed(self) -> Union[float, int]:
        """Return the (signed) value of the constant."""
        if type(self._value) == float:
            return self._value
        return self._value - 2 * ((1 << self.size - 1) & self._value)  # type: ignore

    @property
    def unsigned(self) -> Union[float, int]:
        """Return the unsigned value of the constant."""
        return self._value

    def __repr__(self) -> str:
        """Return a debug representation of the constant."""
        return f"Constant({self._value}, {self._size})"

    def copy(self) -> GraphNodeInterface:
        """Generate a copy of the constant."""
        return Constant(self._world, self._value, self._size)

    def accept(self, visitor: WorldObjectVisitor[T]) -> T:
        """Invoke the appropriate visitor for Variable."""
        return visitor.visit_constant(self)


class Operation(WorldObject, ABC):
    """Class representing the interface for Operations."""

    def __init__(self, world: WorldType):
        """Init a new operation with the dirty flag set."""
        super().__init__(world)
        self._dirty: bool = True

    @property
    @abstractmethod
    def SYMBOL(self) -> str:
        """Return the symbol of the operation, e.g. '+', '-', ..."""

    @property
    def operands(self) -> List[WorldObject]:
        """Return a list of operands for the operation."""
        return self._world.get_operands(self)

    @property
    def constants(self) -> List[Constant]:
        """Return all utilized constants."""
        return [op for op in self.operands if isinstance(op, Constant)]

    @property
    def variables(self) -> List[Variable]:
        """Return all utilized variables at the top level."""
        return [op for op in self.operands if isinstance(op, Variable)]

    def variable_count(self) -> int:
        """Return the number of vars used in this operation, with repeats."""
        count = 0
        for op in self.operands:
            if isinstance(op, Variable) or isinstance(op, Constant):
                count += 1
            elif isinstance(op, Operation):
                count += op.variable_count()
            else:
                assert False, "unhandled operand type"
        return count

    @property
    def children(self) -> List[Operation]:
        """Return all nested operations."""
        return [op for op in self.operands if isinstance(op, Operation)]

    @abstractmethod
    def eval(self, operands: List[Constant]) -> Constant:
        """Evaluate the value of the operation applied in the given constants."""

    @abstractmethod
    def fold(self, keep_form: bool = False):
        """Do constant folding on this operation."""

    @abstractmethod
    def factorize(self) -> Optional[Operation]:
        """Factor out common subterms of the operation and yield a new operation, if any."""

    def collapse(self) -> Optional[BitVector]:
        """Evaluate the Operation, if all Operands are constants."""
        value = None
        if self.operands and all((isinstance(operand, Constant) for operand in self.operands)):
            value = self.eval(self.constants)
            self.world.replace(self, value)
        return value

    @dirty
    def simplify(self, keep_form: bool = False) -> Optional[WorldObject]:
        """
        Simplify the current operation by applying constant folding and factorization.

        This function may manipulate its Child nodes and outgoing edges,
        but should maintain incoming edges to the region.
        - keep_form=True means that after the simplification the formula is still in CNF reps. DNF form it had this form before.
        - If the operand changes during the simplification, then we return the new operand.
        """
        self.fold(keep_form=keep_form)
        if not keep_form and (new_operation := self.factorize()):
            self.world.replace(self, new_operation)
            return new_operation
        return self.collapse()

    @dirty
    def add_operand(self, operand: WorldObject):
        """Add another operand to the operation."""
        self.world.add_operand(self, operand)

    @dirty
    def remove_operand(self, operand: WorldObject):
        """
        Remove the given operand from the operation.

        May has to be called several times if y variable is referenced multiple times.
        e.g. x + x
        """
        self.world.remove_operand(self, operand)

    @dirty
    def replace_operand(self, original: WorldObject, new: WorldObject):
        """Replace the original operand with the new operand."""
        self.world.replace(original, new)

    @dirty
    def replace_operands(self, operands: Iterable[WorldObject]) -> Operation:
        """Replace all current operands with the list of new operands."""
        for operand in self.operands:
            self.remove_operand(operand)
        for operand in operands:
            self.add_operand(operand)
        return self

    def copy(self) -> Operation:
        """Generate a copy of the current Operation."""
        return self.__class__(self._world)

    def copy_tree(self) -> Operation:
        """Generate a copy of the whole operation."""
        copy_operation = self.copy()
        for operand in self.operands:
            self.world.add_operand(copy_operation, operand.copy_tree())
        return copy_operation

    def __eq__(self, other: object) -> bool:
        """Check whether one operation is the same as the other without respect to its operands."""
        return self.__class__ == other.__class__

    def __hash__(self) -> int:
        """Each operation should have an unique hash."""
        return id(self)

    @property
    def size(self):
        """Return the size of the result of the operation by evaluating the sizes of the operands."""
        return max([operand.size for operand in self.operands])
