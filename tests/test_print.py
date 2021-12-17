"""Test pretty-printing and debug printing."""
import pytest

from simplifier.operations.bitwise import BitwiseNegate
from simplifier.world.world import World


class TestPrettyPrint:
    """Test C-like string printing."""

    @pytest.mark.parametrize(
        "to_parse, pretty",
        [
            ("2@8", "2"),
            ("v@4", "v"),
            ("(Tmp)v@4", "v"),
            ("(& a @4 a@4)", "(a & a)"),
            ("(& a@4 b@8)", "(a & b)"),
            ("(^ a@4  (~b@8))", "(a ^ ~b)"),
            ("(| a@4 b@4 c@4)", "(a | b | c)"),
            ("(| a@4 b@4 (& 2@4 c@4))", "(a | b | (2 & c))"),
            ("(~b@8)", "~b"),
            ("(u< 2@8 v@8)", "(2 u< v)"),
            ("(rotl a@4 b@4)", "(a << b) | (a >> (4 - b))"),
            ("(rotr a@32 4@32)", "(a >> 4) | (a << (32 - 4)) & 0xFFFFFFFF"),
        ],
    )
    def test_str(self, to_parse, pretty):
        w = World()
        e = w.from_string(to_parse)
        assert str(e) == pretty

    def test_empty_operand(self):
        w = World()
        bneg = BitwiseNegate(w)
        assert str(bneg) == "(~)"
