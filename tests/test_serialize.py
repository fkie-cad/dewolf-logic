"""Test pretty-printing and debug printing."""
import pytest

from simplifier.operations.bitwise import BitwiseNegate
from simplifier.visitor.serialize_visitor import SerializeVisitor
from simplifier.world.world import World


class TestSerializer:
    """Test C-like string printing."""

    @pytest.mark.parametrize(
        "to_parse, output",
        [
            ("2@8", "2@8"),
            ("v@4", "v@4"),
            ("(Tmp)v@4", "(Tmp)v@4"),
            ("(& a @4 a@4)", "(& a@4 a@4)"),
            ("(& a@4 b@8)", "(& a@4 b@8)"),
            ("(^ a@4  (~b@8))", "(^ a@4 (~b@8))"),
            ("(| a@4 b@4 c@4)", "(| a@4 b@4 c@4)"),
            ("(| a@4 b@4 (& 2@4 c@4))", "(| a@4 b@4 (& 2@4 c@4))"),
            ("(~b@8)", "(~b@8)"),
            ("(u< 2@8 v@8)", "(u< 2@8 v@8)"),
        ],
    )
    def test_str(self, to_parse, output):
        w = World()
        visitor = SerializeVisitor()

        e = w.from_string(to_parse)
        assert e.accept(visitor) == output

    def test_empty_operand(self):
        w = World()
        visitor = SerializeVisitor()

        bneg = BitwiseNegate(w)
        assert bneg.accept(visitor) == "(~)"
