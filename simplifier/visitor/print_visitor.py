"""Pretty-print logic expressions in C-like syntax."""
from simplifier.operations.bitwise import RotateLeft
from simplifier.operations.interface import UnaryOperation
from simplifier.visitor.visitor import WorldObjectVisitor
from simplifier.world.nodes import BaseVariable, Constant, Operation, WorldObject


class PrintVisitor(WorldObjectVisitor):
    """Visitor to pretty-print logic expressions in C-like syntax."""

    def visit_world_object(self, world_object: WorldObject) -> str:
        """Raise an exception, as this should never be called."""
        raise ValueError("Tried to print object with no print visitor.")

    def visit_variable(self, variable: BaseVariable) -> str:
        """Return the name of the variable."""
        return variable.name

    def visit_constant(self, constant: Constant) -> str:
        """Return the value of the constant."""
        return str(constant._value)

    def print_unary_operation(self, op: UnaryOperation) -> str:
        """Print a unary operation."""
        if len(op.operands) > 0:
            return f"{op.SYMBOL}{self.visit(op.operand)}"
        return f"({op.SYMBOL})"

    def print_operation(self, op: Operation) -> str:
        """Print a non-unary operation."""
        if operands := op.operands:
            spaced_symbol = f" {op.SYMBOL} "
            return f"({spaced_symbol.join([self.visit(op) for op in operands])})"
        return f"({op.SYMBOL})"

    def print_rotate(self, op: Operation) -> str:
        """Print a left or right rotate operation."""
        assert len(op.operands) == 2
        lhs = self.visit(op.operands[0])
        rhs = self.visit(op.operands[1])
        size = max([constant.size for constant in op.operands])
        if isinstance(op, RotateLeft):
            return f"({lhs} << {rhs}) | ({lhs} >> ({size} - {rhs}))"
        else:
            f_size = "0x" + "F" * (size // 4)
            return f"({lhs} >> {rhs}) | ({lhs} << ({size} - {rhs})) & {f_size}"

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
    visit_rotate_left = print_rotate  # type: ignore
    visit_rotate_right = print_rotate  # type: ignore
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
