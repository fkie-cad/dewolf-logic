"""Test the expression parsers."""
import pytest

from simplifier.world.nodes import TmpVariable
from simplifier.world.world import World


class TestTreeInWorld:
    @pytest.mark.parametrize(
        "string, val, sz",
        [
            ("2@8", 2, 8),
            ("7.2@32", 7.2, 32),
            ("2 @3", 2, 3),
            ("-1@4", -1, 4),
            ("-3 @ 4", -3, 4),
            ("2@1", 0, 1),
        ],
    )
    def test_constant(self, string, val, sz):
        w = World()
        c = w.constant(val, sz)
        assert World.compare(c, w.from_string(string))

    @pytest.mark.parametrize(
        "string, name, sz",
        [
            ("i@8", "i", 8),
            ("_hello@3", "_hello", 3),
            ("v1@4", "v1", 4),
            ("VAR @ 4", "VAR", 4),
        ],
    )
    def test_variable(self, string, name, sz):
        w = World()
        v = w.variable(name, sz)
        assert World.compare(v, w.from_string(string))

    def test_variable_optional_size(self):
        w = World()
        e = w.from_string("(& v@4 v)")
        v = w.from_string("v")
        assert v.size == 4

    @pytest.mark.parametrize(
        "string, name, sz",
        [
            ("(Tmp)i@8", "i", 8),
            ("(Tmp)_hello@3", "_hello", 3),
            ("(Tmp)v1@4", "v1", 4),
            ("(Tmp)VAR @ 4", "VAR", 4),
        ],
    )
    def test_tmp_variable(self, string, name, sz):
        w = World()
        v = TmpVariable(w, name, sz)
        assert World.compare(v, w.from_string(string))

    @pytest.mark.parametrize(
        "negstring, string",
        [
            ("(~2@8)", "2@8"),
            ("(~(~2@8))", "(~2@8)"),
        ],
    )
    def test_bitwise_neg(self, negstring, string):
        w = World()
        e = w.bitwise_negate(w.from_string(string))
        assert World.compare(w.from_string(negstring), e)

    @pytest.mark.parametrize(
        "andstring, ops",
        [
            ("(& 2@8 3@8)", ("2@8", "3@8")),
            ("(& (& 2@8 v@8) 3@8)", ("(& 2@8 v@8)", "3@8")),
            ("(& 2@8 (& v@8 3@8))", ("2@8", "(& v@8 3@8)")),
            ("(& 2@8 (& v@8 (~3@8)))", ("2@8", "(& v@8 (~3@8))")),
            # >2 operands
            ("(& 1@8 v@8 6@8)", ("1@8", "v@8", "6@8")),
            ("(& 1@8 (| v@8 6@8) u@4)", ("1@8", "(| v@8 6@8)", "u@4")),
            # Check that parsing doesn't simplify
            ("(& 1@8 (& v@8 6@8) u@4)", ("1@8", "(& v@8 6@8)", "u@4")),
        ],
    )
    def test_bitwise_and(self, andstring, ops):
        w = World()
        e = w.bitwise_and(*map(w.from_string, ops))
        assert World.compare(w.from_string(andstring), e)

    @pytest.mark.parametrize(
        "orstring, ops",
        [
            ("(| 2@8 3@8)", ("2@8", "3@8")),
            ("(| (| 2@8 v@8) 3@8)", ("(| 2@8 v@8)", "3@8")),
            ("(| ( ~2@8) (| v@8 3@8))", ("(~ 2@8)", "(| v@8 3@8)")),
            # >2 operands
            ("(| 1@8 v@8 6@8)", ("1@8", "v@8", "6@8")),
            # Check that parsing doesn't simplify
            ("(| 1@8 (| v@8 6@8) 1@8)", ("1@8", "(| v@8 6@8)", "1@8")),
        ],
    )
    def test_bitwise_or(self, orstring, ops):
        w = World()
        e = w.bitwise_or(*map(w.from_string, ops))
        assert World.compare(w.from_string(orstring), e)

    @pytest.mark.parametrize(
        "xorstring, ops",
        [
            ("(^ a@8 b@8)", ("a@8", "b  @ 8")),
            ("(^ a@8 (~2@8))", ("a@8", "(~2@ 8)")),
            # >2 operands
            ("(^ a@ 8 a@8 2@8)", ("a@8", "a@8", "2@8")),
        ],
    )
    def test_bitwise_xor(self, xorstring, ops):
        w = World()
        e = w.bitwise_xor(*map(w.from_string, ops))
        assert World.compare(w.from_string(xorstring), e)

    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            ("v@8", "u@8"),
            ("v@8", "(& 1@8 3@8)"),
            ("v@8", "(& x@8 y@8)"),
        ],
    )
    def test_define(self, lhs, rhs):
        w = World()
        w.from_string(f"{lhs} = {rhs}")
        assert w.get_definition(w.from_string(lhs)) == w.from_string(rhs)

    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            ("(Tmp)v@8", "u@8"),
            ("(Tmp)v@8", "(& 1@8 3@8)"),
            ("(Tmp)v@8", "(& x@8 y@8)"),
        ],
    )
    def test_define_tmp(self, lhs, rhs):
        w = World()
        w.from_string(f"{lhs} = {rhs}")
        assert w.get_definition(TmpVariable(w, "v", 8)) == w.from_string(rhs)
