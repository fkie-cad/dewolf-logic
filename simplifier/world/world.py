"""Module extending the base world interface with builder functions for usability."""
from __future__ import annotations

import logging

from lark import Token, Transformer, v_args  # type: ignore

from simplifier.operations.arithmetic import *
from simplifier.operations.bitwise import *
from simplifier.operations.boolean import *
from simplifier.parser import grammar
from simplifier.world.interface import WorldInterface
from simplifier.world.nodes import BaseVariable, Constant, TmpVariable, Variable, WorldObject


class World(WorldInterface):
    """
    Expands the world interface with helper functions.

    Construction functions creating new objects in the world
    These functions return their root nodes to allow bottom-up construction

    w = World()
    w.bitwise_and(w.variable('x', 8), w.bitwise_or(w.constant(1, 8), w.variable('y', 8))
    """

    def __init__(self, *args, **kwargs):
        """Initialise a new World.

        The transformer TreeInWorld is specific to the legal operations on World.
        """
        self._parser = grammar(TreeInWorld(self))
        self.variable_name_counter = 0
        super().__init__(*args, **kwargs)

    def variable(self, name: str, size: Optional[int] = None) -> Variable:
        """Declare a variable with an unique name and a given size in bits."""
        if variable := self._variables.get(name):
            if not isinstance(variable, Variable):
                raise ValueError(f"Variable {variable} is a temporary variable and can therefore not be a variable.")
            if size is None or variable.size == size:
                return variable
            raise ValueError(f"Tried to add `{name}` with size `{size}`, but already have {variable}")
        elif size is None:
            raise ValueError(f"Undefined variable {name}, please specify a size")
        else:
            variable = Variable(self, name, size)
            self._variables[name] = variable
            self._graph.add_node(variable)
            return variable

    def tmp_variable(self, name: str, size: int) -> TmpVariable:
        """Declare a variable with an unique name and a given size in bits."""
        if self._variables.get(name):
            raise ValueError(f"Variable with name {name} already exists, so we can not add it as a temporary variable.")
        variable = TmpVariable(self, name, size)
        self._variables[name] = variable
        self._graph.add_node(variable)
        return variable

    def constant(self, value: Union[int, float], size: int) -> Constant:
        """Declare a constant in the current world with a given size in bits."""
        constant = Constant(self, value, size)
        self._graph.add_node(constant)
        return constant

    def new_variable(self, size: int, tmp: bool = False) -> BaseVariable:
        """Generate a new variable of given size. Depending on the tmp flag it is a temporary variable."""
        self.variable_name_counter += 1
        if tmp:
            return self.tmp_variable(f"tmp_var_{self.variable_name_counter}", size)
        return self.variable(f"var_{self.variable_name_counter}", size)

    # arithmetic operations

    def signed_mul(self, *operands: WorldObject) -> SignedMultiplication:
        """Generate a signed multiplication operation."""
        return self._add_operation(SignedMultiplication(self), operands)

    def unsigned_mul(self, *operands: WorldObject) -> UnsignedMultiplication:
        """Generate a unsigned multiplication operation."""
        return self._add_operation(UnsignedMultiplication(self), operands)

    def signed_div(self, *operands: WorldObject) -> SignedDivision:
        """Generate a signed division operation."""
        return self._add_operation(SignedDivision(self), operands)

    def unsigned_div(self, *operands: WorldObject) -> UnsignedDivision:
        """Generate a unsigned division operation."""
        return self._add_operation(UnsignedDivision(self), operands)

    def signed_add(self, *operands: WorldObject) -> SignedAddition:
        """Generate a signed addition operation."""
        return self._add_operation(SignedAddition(self), operands)

    def unsigned_add(self, *operands: WorldObject) -> UnsignedAddition:
        """Generate a unsigned addition operation."""
        return self._add_operation(UnsignedAddition(self), operands)

    def signed_sub(self, *operands: WorldObject) -> SignedSubtraction:
        """Generate a signed subtraction operation."""
        return self._add_operation(SignedSubtraction(self), operands)

    def unsigned_sub(self, *operands: WorldObject) -> UnsignedSubtraction:
        """Generate a unsigned subtraction operation."""
        return self._add_operation(UnsignedSubtraction(self), operands)

    def signed_mod(self, *operands: WorldObject) -> SignedModulo:
        """Generate a signed modulo operation."""
        return self._add_operation(SignedModulo(self), operands)

    def unsigned_mod(self, *operands: WorldObject) -> UnsignedModulo:
        """Generate a unsigned modulo operation."""
        return self._add_operation(UnsignedModulo(self), operands)

    # bitwise operations

    def bitwise_and(self, *operands: WorldObject) -> BitwiseAnd:
        """Generate a bitwise and operation."""
        return self._add_operation(BitwiseAnd(self), operands)

    def bitwise_or(self, *operands: WorldObject) -> BitwiseOr:
        """Generate a bitwise or operation."""
        return self._add_operation(BitwiseOr(self), operands)

    def bitwise_xor(self, *operands: WorldObject) -> BitwiseXor:
        """Generate a bitwise xor operation."""
        return self._add_operation(BitwiseXor(self), operands)

    def bitwise_negate(self, operand: WorldObject) -> BitwiseNegate:
        """Generate a bitwise negation operation on a single operand."""
        return self._add_operation(BitwiseNegate(self), (operand,))

    def shift_left(self, *operands: WorldObject) -> ShiftLeft:
        """Generate a bitwise shift left operation."""
        return self._add_operation(ShiftLeft(self), operands)

    def shift_right(self, *operands: WorldObject) -> ShiftRight:
        """Generate a bitwise shift right operation."""
        return self._add_operation(ShiftRight(self), operands)

    def rotate_left(self, *operands: WorldObject) -> RotateLeft:
        """Generate a bitwise rotate left operation."""
        return self._add_operation(RotateLeft(self), operands)

    def rotate_right(self, *operands: WorldObject) -> RotateRight:
        """Generate a bitwise rotate right operation."""
        return self._add_operation(RotateRight(self), operands)

    # boolean operations

    def bool_negate(self, operand: WorldObject) -> Not:
        """Generate a signed > operation."""
        return self._add_operation(Not(self), (operand,))

    def bool_equal(self, *operands: WorldObject) -> Equal:
        """Generate a == operation."""
        return self._add_operation(Equal(self), operands)

    def bool_unequal(self, *operands: WorldObject) -> Unequal:
        """Generate a != operation."""
        return self._add_operation(Unequal(self), operands)

    def signed_gt(self, *operands: WorldObject) -> SignedGreater:
        """Generate a signed > operation."""
        return self._add_operation(SignedGreater(self), operands)

    def signed_ge(self, *operands: WorldObject) -> SignedGreaterEqual:
        """Generate a signed >= operation."""
        return self._add_operation(SignedGreaterEqual(self), operands)

    def signed_lt(self, *operands: WorldObject) -> SignedLesser:
        """Generate a signed < operation."""
        return self._add_operation(SignedLesser(self), operands)

    def signed_le(self, *operands: WorldObject) -> SignedLesserEqual:
        """Generate a signed <= operation."""
        return self._add_operation(SignedLesserEqual(self), operands)

    def unsigned_gt(self, *operands: WorldObject) -> UnsignedGreater:
        """Generate a unsigned > operation."""
        return self._add_operation(UnsignedGreater(self), operands)

    def unsigned_ge(self, *operands: WorldObject) -> UnsignedGreaterEqual:
        """Generate a unsigned >= operation."""
        return self._add_operation(UnsignedGreaterEqual(self), operands)

    def unsigned_lt(self, *operands: WorldObject) -> UnsignedLesser:
        """Generate a unsigned < operation."""
        return self._add_operation(UnsignedLesser(self), operands)

    def unsigned_le(self, *operands: WorldObject) -> UnsignedLesserEqual:
        """Generate a unsigned <= operation."""
        return self._add_operation(UnsignedLesserEqual(self), operands)

    def from_string(self, string: str) -> WorldObject:
        """Apply the operations to this World."""
        result = self._parser.parse(string)
        logging.debug("Parsed %s to %r", string, result)
        return result


class TreeInWorld(Transformer):
    """Transform a parse tree within the context of a World."""

    def __init__(self, world: World, *args, **kwargs):
        """Initialise this transformer with a World."""
        self._world = world
        self.variable_name_counter = 0
        super().__init__(*args, **kwargs)

    @v_args(inline=True)
    def value(self, token: Token) -> Union[int, float]:
        """Parse an integer or float value."""
        if token.type == "SIGNED_FLOAT":
            return float(str(token))
        return int(str(token))

    @v_args(inline=True)
    def value_int(self, token: Token) -> int:
        """Parse an integer value."""
        return int(str(token))

    @v_args(inline=True)
    def ident(self, token: Token) -> str:
        """Parse an ident."""
        return str(token)

    @v_args(inline=True)
    def tmp_ident(self, token: Token) -> str:
        """Parse an tmp-ident."""
        return "(Tmp)" + str(token)

    @v_args(inline=True)
    def size(self, token: Token) -> int:
        """Parse a size."""
        return int(str(token))

    @v_args(inline=True)
    def constant(self, val: int, sz: int) -> Constant:
        """Parse a Constant."""
        return self._world.constant(val, sz)

    @v_args(inline=True)
    def constant_int(self, val: int, sz: int) -> Constant:
        """Parse a Constant."""
        return self._world.constant(val, sz)

    @v_args(inline=True)
    def variable(self, var: str, sz: Optional[int] = None) -> Variable:
        """Parse a Variable."""
        return self._world.variable(var, sz)

    @v_args(inline=True)
    def tmp_variable(self, var: str, sz: int) -> TmpVariable:
        """Parse a temporary Variable."""
        var = var[5:]
        return self._world.tmp_variable(var, sz)

    @v_args(inline=True)
    def new_variable(self, size: int, tmp: bool = False) -> BaseVariable:
        """Parse a new Variable creation."""
        return self._world.new_variable(size, tmp)

    @v_args(inline=True)
    def define(self, variable: Variable, rhs: WorldObject[World]) -> WorldObject:
        """Parse a definition statement."""
        return self._world.define(variable, rhs)

    # arithmetic operations

    @v_args(inline=True)
    def signed_add(self, *exprs: WorldObject[World]) -> SignedAddition:
        """Generate a signed addition operation."""
        return self._world.signed_add(*exprs)

    @v_args(inline=True)
    def unsigned_add(self, *exprs: WorldObject[World]) -> UnsignedAddition:
        """Generate a unsigned addition operation."""
        return self._world.unsigned_add(*exprs)

    @v_args(inline=True)
    def signed_sub(self, *exprs: WorldObject[World]) -> SignedSubtraction:
        """Generate a signed subtraction operation."""
        return self._world.signed_sub(*exprs)

    @v_args(inline=True)
    def unsigned_sub(self, *exprs: WorldObject[World]) -> UnsignedSubtraction:
        """Generate a unsigned subtraction operation."""
        return self._world.unsigned_sub(*exprs)

    @v_args(inline=True)
    def signed_mod(self, *exprs: WorldObject[World]) -> SignedModulo:
        """Generate a signed modulo operation."""
        return self._world.signed_mod(*exprs)

    @v_args(inline=True)
    def unsigned_mod(self, *exprs: WorldObject[World]) -> UnsignedModulo:
        """Generate a unsigned modulo operation."""
        return self._world.unsigned_mod(*exprs)

    @v_args(inline=True)
    def signed_mul(self, *exprs: WorldObject[World]) -> SignedMultiplication:
        """Generate a signed multiplication operation."""
        return self._world.signed_mul(*exprs)

    @v_args(inline=True)
    def unsigned_mul(self, *exprs: WorldObject[World]) -> UnsignedMultiplication:
        """Generate a unsigned multiplication operation."""
        return self._world.unsigned_mul(*exprs)

    @v_args(inline=True)
    def signed_div(self, *exprs: WorldObject[World]) -> SignedDivision:
        """Generate a signed division operation."""
        return self._world.signed_div(*exprs)

    @v_args(inline=True)
    def unsigned_div(self, *exprs: WorldObject[World]) -> UnsignedDivision:
        """Generate a unsigned division operation."""
        return self._world.unsigned_div(*exprs)

    # bitwise operations

    @v_args(inline=True)
    def bitwise_and(self, *exprs: WorldObject[World]) -> BitwiseAnd:
        """Parse a BitwiseAnd."""
        return self._world.bitwise_and(*exprs)

    @v_args(inline=True)
    def bitwise_or(self, *exprs: WorldObject[World]) -> BitwiseOr:
        """Parse a BitwiseOr."""
        return self._world.bitwise_or(*exprs)

    @v_args(inline=True)
    def bitwise_xor(self, *exprs: WorldObject[World]) -> BitwiseXor:
        """Parse a BitwiseXor."""
        return self._world.bitwise_xor(*exprs)

    @v_args(inline=True)
    def bitwise_neg(self, expr: WorldObject[World]) -> BitwiseNegate:
        """Parse a BitwiseNegate."""
        return self._world.bitwise_negate(expr)

    @v_args(inline=True)
    def shift_left(self, *exprs: WorldObject[World]) -> ShiftLeft:
        """Parse a ShiftLeft."""
        return self._world.shift_left(*exprs)

    @v_args(inline=True)
    def shift_right(self, *exprs: WorldObject[World]) -> ShiftRight:
        """Parse a ShiftRight."""
        return self._world.shift_right(*exprs)

    @v_args(inline=True)
    def rotate_left(self, *exprs: WorldObject[World]) -> RotateLeft:
        """Parse a RotateLeft."""
        return self._world.rotate_left(*exprs)

    @v_args(inline=True)
    def rotate_right(self, *exprs: WorldObject[World]) -> RotateRight:
        """Parse a RotateRight."""
        return self._world.rotate_right(*exprs)

    # boolean operations

    @v_args(inline=True)
    def bool_negate(self, expr: WorldObject[World]) -> Not:
        """Parse a signed > operation."""
        return self._world.bool_negate(expr)

    @v_args(inline=True)
    def bool_equal(self, *exprs: WorldObject[World]) -> Equal:
        """Parse an == operation."""
        return self._world.bool_equal(*exprs)

    @v_args(inline=True)
    def bool_unequal(self, *exprs: WorldObject[World]) -> Unequal:
        """Parse an != operation."""
        return self._world.bool_unequal(*exprs)

    @v_args(inline=True)
    def signed_gt(self, *exprs: WorldObject[World]) -> SignedGreater:
        """Parse a signed > operation."""
        return self._world.signed_gt(*exprs)

    @v_args(inline=True)
    def signed_ge(self, *exprs: WorldObject[World]) -> SignedGreaterEqual:
        """Parse a signed >= operation."""
        return self._world.signed_ge(*exprs)

    @v_args(inline=True)
    def signed_lt(self, *exprs: WorldObject[World]) -> SignedLesser:
        """Parse a signed < operation."""
        return self._world.signed_lt(*exprs)

    @v_args(inline=True)
    def signed_le(self, *exprs: WorldObject[World]) -> SignedLesserEqual:
        """Parse a signed <= operation."""
        return self._world.signed_le(*exprs)

    @v_args(inline=True)
    def unsigned_gt(self, *exprs: WorldObject[World]) -> UnsignedGreater:
        """Parse a unsigned > operation."""
        return self._world.unsigned_gt(*exprs)

    @v_args(inline=True)
    def unsigned_ge(self, *exprs: WorldObject[World]) -> UnsignedGreaterEqual:
        """Parse a unsigned >= operation."""
        return self._world.unsigned_ge(*exprs)

    @v_args(inline=True)
    def unsigned_lt(self, *exprs: WorldObject[World]) -> UnsignedLesser:
        """Parse a unsigned < operation."""
        return self._world.unsigned_lt(*exprs)

    @v_args(inline=True)
    def unsigned_le(self, *exprs: WorldObject[World]) -> UnsignedLesserEqual:
        """Parse a unsigned <= operation."""
        return self._world.unsigned_le(*exprs)
