"""Serialize logic expressions back to its original form (ie, same string as the one passed to world.from_string())."""
from simplifier.operations.interface import UnaryOperation
from simplifier.visitor.visitor import WorldObjectVisitor
from simplifier.world.nodes import BaseVariable, Constant, Operation, Variable, WorldObject


class SerializeVisitor(WorldObjectVisitor):
    """Visitor to serialize WorldObject back to string to be stored. This string can be used with world.from_string()."""

    def visit_world_object(self, world_object: WorldObject) -> str:
        """Raise an exception, as this should never be called."""
        raise ValueError("Tried to serialize object with no print visitor.")

    def visit_variable(self, variable: BaseVariable) -> str:
        """Return the name of the variable with the size."""
        if isinstance(variable, Variable):
            return f"{variable.name}@{variable.size}"
        return f"(Tmp){variable.name}@{variable.size}"

    def visit_constant(self, constant: Constant) -> str:
        """Return the value of the constant with the size."""
        return f"{constant._value}@{constant.size}"

    def print_unary_operation(self, op: UnaryOperation) -> str:
        """Print a unary operation."""
        if len(op.operands) > 0:
            return f"({op.SYMBOL}{self.visit(op.operand)})"
        return f"({op.SYMBOL})"

    def print_operation(self, op: Operation) -> str:
        """Print a non-unary operation."""
        if operands := op.operands:
            operands_string = " ".join([self.visit(op) for op in operands])
            return f"({op.SYMBOL} {operands_string})"
        return f"({op.SYMBOL})"

    visit_unsigned_add = print_operation  # type: ignore
    visit_signed_add = print_operation  # type: ignore
    visit_unsigned_sub = print_operation  # type: ignore
    visit_signed_sub = print_operation  # type: ignore
    visit_unsigned_mod = print_operation  # type: ignore
    visit_signed_mod = print_operation  # type: ignore
    visit_unsigned_mul = print_operation  # type: ignore
    visit_signed_mul = print_operation  # type: ignore
    visit_unsigned_div = print_operation  # type: ignore
    visit_signed_div = print_operation  # type: ignore

    visit_bitwise_and = print_operation  # type: ignore
    visit_bitwise_or = print_operation  # type: ignore
    visit_bitwise_xor = print_operation  # type: ignore
    visit_shift_left = print_operation  # type: ignore
    visit_shift_right = print_operation  # type: ignore
    visit_rotate_left = print_operation  # type: ignore
    visit_rotate_right = print_operation  # type: ignore
    visit_bitwise_negate = print_unary_operation  # type: ignore

    visit_bool_negate = print_unary_operation  # type: ignore
    visit_bool_equal = print_operation  # type: ignore
    visit_bool_unequal = print_operation  # type: ignore

    visit_signed_gt = print_operation  # type: ignore
    visit_signed_ge = print_operation  # type: ignore
    visit_signed_lt = print_operation  # type: ignore
    visit_signed_le = print_operation  # type: ignore
    visit_unsigned_gt = print_operation  # type: ignore
    visit_unsigned_ge = print_operation  # type: ignore
    visit_unsigned_lt = print_operation  # type: ignore
    visit_unsigned_le = print_operation  # type: ignore
