"""Module testing the functionality of the logic operations."""
import pytest

from simplifier.operations.bitwise import *
from simplifier.world.nodes import WorldType
from simplifier.world.world import Constant, World


class TestBitwiseAnd:
    @pytest.mark.parametrize(
        "a, b",
        [
            ("(& x@8 y@8)", "(& y@8 x@8)"),
            ("(& x@8 y@8 z@8)", "(& z@8 x@8 y@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert w.compare(obj1, obj2)

    class TestFolding:
        def test_constant_folding(self):
            """1 & 3 = 1"""
            w = World()
            v = w.variable("v", 8)
            w.define(v, w.bitwise_and(w.constant(1, 8), w.constant(3, 8)))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.constant(1, 8))

        def test_duplicate_folding(self):
            """x & x = x"""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_and(w.variable("x", 8), w.variable("x", 8)))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.variable("x", 8))

        def test_null_folding(self):
            """x & 0 = 0"""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_and(w.variable("x", 8), w.constant(0, 8)))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.constant(0, 8))

        def test_bitmask_folding(self):
            """x@8 & b11111111 = x@8"""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_and(w.variable("x", 8), w.constant(2 ** 8 - 1, 8)))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.variable("x", 8))

        def test_variable_folding(self):
            """x & ( x | y ) = x"""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_and(w.variable("x", 8), w.bitwise_or(w.variable("x", 8), w.variable("y", 8))))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.variable("x", 8))

        def test_associative_folding_1a(self):
            """(a | b) & (a | b | c) = (a | b)"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_and(
                    w.bitwise_or(w.variable("a", 8), w.variable("b", 8)),
                    w.bitwise_or(w.variable("a", 8), w.variable("b", 8), w.variable("c", 8)),
                ),
            )
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.bitwise_or(w.variable("a", 8), w.variable("b", 8)))

        def test_associative_folding_1b(self):
            """(((a & b) | c) & ((a & x) | c | d)) = (c | ((x | d) & a & b))
            This test is to ensure that simplify() does not output ((a & b) | c)."""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_and(
                    w.bitwise_or(
                        w.bitwise_and(
                            w.variable("a", 8),
                            w.variable("b", 8),
                        ),
                        w.variable("c", 8),
                    ),
                    w.bitwise_or(
                        w.bitwise_and(
                            w.variable("a", 8),
                            w.variable("x", 8),
                        ),
                        w.variable("c", 8),
                        w.variable("d", 8),
                    ),
                ),
            )

            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.from_string("(| c@8 (& (| x@8 d@8) a@8 b@8))"))

        @pytest.mark.skip(f"Not implemented yet!")
        def test_associative_folding_2(self):
            """(a xor b) & (a xor b xor c) = (a xor b) & ~c"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_and(
                    w.bitwise_xor(w.variable("a", 8), w.variable("b", 8)),
                    w.bitwise_xor(w.variable("a", 8), w.variable("b", 8), w.variable("c", 8)),
                ),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)),
                w.bitwise_and(w.bitwise_xor(w.variable("a", 8), w.variable("b", 8)), w.bitwise_negate(w.variable("c", 8))),
            )

        def test_associative_folding_3(self):
            """(a | b) & (c | a | b) = (a | b)"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_and(
                    w.bitwise_or(w.variable("a", 8), w.variable("b", 8)),
                    w.bitwise_or(w.variable("c", 8), w.variable("a", 8), w.variable("b", 8)),
                ),
            )
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.bitwise_or(w.variable("a", 8), w.variable("b", 8)))

        def test_associative_folding_4(self):
            """a & (!a | b) = (a & b)"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_and(w.variable("a", 8), w.bitwise_or(w.bitwise_negate(w.variable("a", 8)), w.variable("b", 8))),
            )
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.bitwise_and(w.variable("a", 8), w.variable("b", 8)))

        def test_associative_folding_5(self):
            """a & (b | !a) = (a & b)"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_and(w.variable("a", 8), w.bitwise_or(w.variable("b", 8), w.bitwise_negate(w.variable("a", 8)))),
            )
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.bitwise_and(w.variable("a", 8), w.variable("b", 8)))

        def test_associative_folding_6(self):
            """(x | y) & ( x | ~y ) = x because (x | y) & ( x | ~y ) = x & (y | ~y) = x & True = x"""
            w = World()
            cond = w.from_string("(& (| x@1 y@1) (| x@1 (~y@1)) )")
            w.define(v := w.variable("v", 1), cond)
            w.simplify()
            assert World.compare(v, w.from_string("x@1"))

        def test_associative_folding_7(self):
            """(~x | ~y) & (z | (x & y)) = (~x | ~y) & z"""
            w = World()
            cond = w.from_string("(& (| (~x@1) (~y@1)) (| z@1 (& x@1 y@1)) )")
            w.define(v := w.variable("v", 1), cond)
            w.simplify()
            assert World.compare(v, w.from_string("(& (| (~ x@1) (~ y@1) ) z@1)"))

    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            ("v@4 = (& (| x@4 a@4) (| x@4 b@4))", "(| x (& a b))"),
            ("v@4 = (| (& a@4 y@4) (& y@4 b@4))", "(& y (| a b))"),
            ("v@4 = (& (| (& x@4 y@4) a@4) (| (& x y) b@4))", "(| (& x y) (& a b))"),
            ("v@4 = (& (| x@4 a@4) (| x@4 b@4) (! z@4))", "(& (| x@4 a@4) (| x@4 b@4) (! z@4))"),
            ("v@4 = (& (| x@4 a@4) (| x@4 b@4) (! (| z@4 x@4)))", "(& (| x@4 a@4) (| x@4 b@4) (! (| z@4 x@4)))"),
        ],
    )
    def test_factorize(self, lhs, rhs):
        w = World()
        w.from_string(lhs)
        w.simplify()
        assert World.compare(w.from_string("v"), w.from_string(rhs))

    def test_factorize_does_nothing(self):
        w = World()
        w.from_string("v@4 = (& (| x@4 a@4) (| x@4 b@4) (& z@4 x@4))")
        term: BitwiseAnd = w.get_definition(w.from_string("v"))  # type: ignore
        term.factorize()
        assert World.compare(w.from_string("v"), w.from_string("(& (| x@4 a@4) (| x@4 b@4) (& z@4 x@4))"))


class TestBitwiseOr:
    @pytest.mark.parametrize(
        "a, b",
        [
            ("(| x@8 y@8)", "(| y@8 x@8)"),
            ("(| x@8 y@8 z@8)", "(| z@8 x@8 y@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert w.compare(obj1, obj2)

    class TestFolding:
        def test_constant_folding(self):
            """1 | 2 = 3"""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_or(w.constant(1, 8), w.constant(2, 8)))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.constant(3, 8))

        def test_negation_folding(self):
            """x | !x = 0xfff.."""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_or(w.variable("x", 8), w.bitwise_negate(w.variable("x", 8))))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), Constant.create_maximum_value(w, 8))

        def test_negation_folding_2(self):
            """x | (x | w) | !w = (x | 256) = 256"""
            w = World()
            v = w.from_string("v@8")
            expr = w.bitwise_or(w.variable("x", 8), w.from_string("(| x@8 w@8)"), w.bitwise_negate(w.variable("w", 8)))
            w.define(w.variable("v", 8), expr)
            w.simplify()
            w.simplify()  # TODO: do we want to simplify once or twice?
            # expect = Constant.create_maximum_value(w, 8)
            expect = w.from_string("255@8")
            assert World.compare(v, expect)

        def test_zero_folding(self):
            """x | 0 = x"""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_or(w.variable("x", 8), w.constant(0, 8)))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.variable("x", 8))

        def test_max_folding(self):
            """x@8 | 0xffffffff = 0xffffffff"""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_or(w.variable("x", 8), w.constant(2 ** 8 - 1, 8)))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.constant(2 ** 8 - 1, 8))

        def test_variable_folding(self):
            """x | (x & y) = x"""
            w = World()
            w.define(w.variable("v", 8), w.bitwise_or(w.variable("x", 8), w.bitwise_and(w.variable("x", 8), w.variable("y", 8))))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.variable("x", 8))

        def test_variable_folding_multiple_times(self):
            """z | x | (x & y & z) = x | z"""
            w = World()
            w.define(w.variable("v", 8), w.from_string("(| z@8 y@8 (& x@8 y@8 z@8))"))
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.from_string("(| z@8 y@8)"))

        def test_xor_folding(self):
            """x | (x ^ y) = x | y"""
            w = World()
            v = w.variable("v", 8)
            expr = w.from_string("(| x@8 (^ x@8 y@8))")
            # x y expr
            # F F F
            # F T T
            # T F T
            # T T T
            # So it's x | y
            w.define(v, expr)
            w.simplify()
            assert World.compare(w.get_definition(v), w.from_string("(| x@8 y@8)"))

        def test_associative_folding_1a(self):
            """a | (!a & b) = a | b"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(w.variable("a", 8), w.bitwise_and(w.bitwise_negate(w.variable("a", 8)), w.variable("b", 8))),
            )
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.bitwise_or(w.variable("a", 8), w.variable("b", 8)))

        def test_associative_folding_1b(self):
            """(!a & b) | a = a | b"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(w.bitwise_and(w.bitwise_negate(w.variable("a", 8)), w.variable("b", 8)), w.variable("a", 8)),
            )
            w.simplify()
            assert World.compare(w.get_definition(w.variable("v", 8)), w.bitwise_or(w.variable("a", 8), w.variable("b", 8)))

        def test_associative_folding_2a(self):
            """!a | (a & b) = !a | b"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(w.bitwise_negate(w.variable("a", 8)), w.bitwise_and(w.variable("a", 8), w.variable("b", 8))),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)), w.bitwise_or(w.bitwise_negate(w.variable("a", 8)), w.variable("b", 8))
            )

        def test_associative_folding_2b(self):
            """(a & b) | !a = !a | b"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(w.bitwise_and(w.variable("a", 8), w.variable("b", 8)), w.bitwise_negate(w.variable("a", 8))),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)), w.bitwise_or(w.bitwise_negate(w.variable("a", 8)), w.variable("b", 8))
            )

        def test_associative_folding_3a(self):
            """(a&c) | (!(a&c) & b) = (a&c) | b"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(
                    w.bitwise_and(w.variable("a", 8), w.variable("c", 8)),
                    w.bitwise_and(w.bitwise_negate(w.bitwise_and(w.variable("a", 8), w.variable("c", 8))), w.variable("b", 8)),
                ),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)),
                w.bitwise_or(w.bitwise_and(w.variable("a", 8), w.variable("c", 8)), w.variable("b", 8)),
            )

        def test_associative_folding_3b(self):
            """(!(a&c) & b) | (a&c) = (a&c) | b"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(
                    w.bitwise_and(w.bitwise_negate(w.bitwise_and(w.variable("a", 8), w.variable("c", 8))), w.variable("b", 8)),
                    w.bitwise_and(w.variable("a", 8), w.variable("c", 8)),
                ),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)),
                w.bitwise_or(w.bitwise_and(w.variable("a", 8), w.variable("c", 8)), w.variable("b", 8)),
            )

        def test_associative_folding_multiple(self):
            """((~x1 & x2 & ~x3 & x5) | (x3 | ~x2 | x1 | ~x5)) = True"""
            w = World()
            term = w.from_string("(| (& (~ x1@1) x2@1 (~ x3@1) x5@1) (| x3@1 (~ x2@1) x1@1 (~ x5@1)) )")
            w.define(var := w.variable("v", 1), term)
            w.simplify()
            assert World.compare(w.get_definition(var), Constant(w, 1, 1))

        @pytest.mark.skip(reason="Not implemented yet, may be complicated")
        def test_associative_folding_4(self):
            """(a&c) | ((!a | !c) & b) = (a&c) | b"""
            assert False

        def test_associative_folding_5(self):
            """a | (!a & b) | (!a & c) = a | b | c"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(
                    w.variable("a", 8),
                    w.bitwise_and(w.bitwise_negate(w.variable("a", 8)), w.variable("b", 8)),
                    w.bitwise_and(w.bitwise_negate(w.variable("a", 8)), w.variable("c", 8)),
                ),
            )
            w.simplify()
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)),
                w.bitwise_or(
                    w.variable("a", 8),
                    w.variable("b", 8),
                    w.variable("c", 8),
                ),
            )

        def test_associative_folding_6(self):
            """a | (!a & b) | (c & d) = a | b | (c &d)"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(
                    w.variable("a", 8),
                    w.bitwise_and(w.bitwise_negate(w.variable("a", 8)), w.variable("b", 8)),
                    w.bitwise_and(w.variable("c", 8), w.variable("d", 8)),
                ),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)),
                w.bitwise_or(
                    w.variable("a", 8),
                    w.variable("b", 8),
                    w.bitwise_and(w.variable("c", 8), w.variable("d", 8)),
                ),
            )

        def test_associative_folding_7a(self):
            """a | (!a & b & c) = a | (b & c)"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(
                    w.variable("a", 8),
                    w.bitwise_and(
                        w.bitwise_negate(w.variable("a", 8)),
                        w.variable("b", 8),
                        w.variable("c", 8),
                    ),
                ),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)),
                w.bitwise_or(
                    w.variable("a", 8),
                    w.bitwise_and(w.variable("b", 8), w.variable("c", 8)),
                ),
            )

        def test_associative_folding_7b(self):
            """a | (b & c & !a) = a | (b & c)"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(
                    w.bitwise_and(
                        w.variable("b", 8),
                        w.variable("c", 8),
                        w.bitwise_negate(w.variable("a", 8)),
                    ),
                    w.variable("a", 8),
                ),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)),
                w.bitwise_or(
                    w.variable("a", 8),
                    w.bitwise_and(w.variable("b", 8), w.variable("c", 8)),
                ),
            )

        def test_associative_folding_8(self):
            """(x & y) | ( x & !y ) = x because (x & y) | ( x & !y ) = x | (y & !y) = x | False = x"""
            w = World()
            cond = w.from_string("(| (& x@1 y@1) (& x@1 (~y@1)) )")
            w.define(v := w.variable("v", 1), cond)
            w.simplify()
            assert World.compare(v, w.from_string("x@1"))

        def test_associative_folding_9(self):
            """(~x & ~y) | (z & (x | y)) = (~x & ~y) | z"""
            w = World()
            cond = w.from_string("(| (& (~x@1) (~y@1)) (& z@1 (| x@1 y@1)) )")
            w.define(v := w.variable("v", 1), cond)
            w.simplify()
            assert World.compare(v, w.from_string("(| (& (~ x@1) (~ y@1) ) z@1)"))

    class TestFactorize:
        def test_factorize(self):
            """(x & y) | (x & z) = x & (y | z)"""
            w = World()
            w.define(
                w.variable("v", 8),
                w.bitwise_or(w.bitwise_and(w.variable("x", 8), w.variable("y", 8)), w.bitwise_and(w.variable("x", 8), w.variable("z", 8))),
            )
            w.simplify()
            assert World.compare(
                w.get_definition(w.variable("v", 8)),
                w.bitwise_and(w.variable("x", 8), w.bitwise_or(w.variable("y", 8), w.variable("z", 8))),
            )

    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            ("v@4 = (| (& (| x@4 y@4) a@4) (& (| x y) b@4))", "(& (| x y) (| a b))"),
        ],
    )
    def test_factorize_param(self, lhs, rhs):
        w = World()
        w.from_string(lhs)
        w.simplify()
        assert World.compare(w.from_string("v"), w.from_string(rhs))


class TestBitwiseXor:
    @pytest.mark.parametrize(
        "term, result",
        [
            ("(^ x@8 x@8 x@8)", "x@8"),
            ("(^ x@8 x@8 z@8)", "z@8"),
            ("(^ x@8 x@8)", "0@8"),
            ("(^ x@8 0@8)", "x@8"),
            ("(^ x@8 255@8)", "(~ x@8)"),
        ],
    )
    def test_simplify(self, term, result):
        """Simplifications for xor."""
        w = World()
        w.define(v := w.variable("v", 8), w.from_string(term))
        w.simplify()
        assert World.compare(v, w.from_string(result))

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(^ x@8 y@8)", "(^ y@8 x@8)"),
            ("(^ x@8 y@8 z@8)", "(^ z@8 x@8 y@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert w.compare(obj1, obj2)


class TestBitwiseNegate:
    def test_duplicate_folding(self):
        """!(!x) = x"""
        w = World()
        w.define(w.variable("v", 8), w.bitwise_negate(w.bitwise_negate(w.variable("x", 8))))
        w.simplify()
        assert World.compare(w.get_definition(w.variable("v", 8)), w.variable("x", 8))

    def test_add_operand(self):
        w = World()
        negation = w.from_string("(~v@4)")
        with pytest.raises(ValueError):
            negation.add_operand(w.from_string("t@2"))

    @pytest.mark.parametrize(
        "term, result",
        [
            ("(~1@1)", "0@1"),
            ("(~0@1)", "1@1"),
            ("(~0@3)", "-1@3"),
            ("(~1@3)", "-2@3"),
            ("(~2@3)", "-3@3"),
            ("(~3@3)", "-4@3"),
            ("(~4@3)", "3@3"),
            ("(~5@3)", "2@3"),
            ("(~6@3)", "1@3"),
            ("(~7@3)", "0@3"),
        ],
    )
    def test_eval_negate_constants(self, term, result):
        w = World()
        term = w.from_string(term)  # type: BitwiseNegate
        constant = term.eval(term.operands)
        assert w.compare(constant, w.from_string(result))

    @pytest.mark.parametrize(
        "term, result",
        [
            ("(~1@1)", "0@1"),
            ("(~0@1)", "1@1"),
            ("(~0@3)", "-1@3"),
            ("(~1@3)", "-2@3"),
            ("(~2@3)", "-3@3"),
            ("(~3@3)", "-4@3"),
            ("(~4@3)", "3@3"),
            ("(~5@3)", "2@3"),
            ("(~6@3)", "1@3"),
            ("(~7@3)", "0@3"),
        ],
    )
    def test_collapse_constants(self, term, result):
        w = World()
        term = w.from_string(term)  # type: BitwiseNegate
        w.define(var := w.variable("v", 1), term)
        term.collapse()
        assert w.compare(var, w.from_string(result))


class TestNegation:
    """Test the negate function"""

    def test_negate_negation(self):
        # negate Not(1)
        w = World()
        operand = w.bitwise_negate(w.constant(1, 8))
        operand.negate()
        assert w.terms() == (w.constant(1, 8),)

    def test_negate_level_0(self):
        # no predecessor
        w = World()
        operand = w.bitwise_and(w.constant(1, 8), w.constant(2, 8))
        operand.negate(0)
        assert len(w) == 4 and isinstance(neg := w.terms()[0], BitwiseNegate) and neg.operand == operand
        # with predecessor
        w = World()
        w.bitwise_or(operand := w.bitwise_and(w.constant(1, 8), w.constant(2, 8)), w.constant(3, 8))
        operand.negate(0)
        assert (
            len(w) == 6
            and isinstance(bit_or := w.terms()[0], BitwiseOr)
            and bit_or.operands[0] == w.constant(3, 8)
            and isinstance(neg := bit_or.operands[1], BitwiseNegate)
            and neg.operand == operand
        )

    def test_negate_no_predecessor(self):
        w = World()
        operand = w.bitwise_and(w.constant(1, 8), w.constant(2, 8))
        operand.negate()
        head = [h for h in w.terms() if w.get_operands(h)][0]
        assert (
            len(w) == 5
            and len(w.terms()) == 1
            and isinstance(head, BitwiseOr)
            and all(isinstance(op, BitwiseNegate) for op in head.operands)
        )

    def test_negate_with_predecessor_1(self):
        w = World()
        w.bitwise_or(operand := w.bitwise_and(w.constant(1, 8), w.constant(2, 8)), w.constant(3, 8))
        operand.negate()
        head = [h for h in w.terms() if w.get_operands(h)][0]
        assert (
            len(w) == 7
            and isinstance(bit_or := head, BitwiseOr)
            and bit_or.operands[1] == w.constant(3, 8)
            and isinstance(sec_or := bit_or.operands[0], BitwiseOr)
            and all(isinstance(op, BitwiseNegate) for op in sec_or.operands)
        )

    def test_negate_with_predecessor_2(self):
        w = World()
        term = w.bitwise_or(operand := w.bitwise_and(w.constant(1, 8), w.constant(2, 8)), w.constant(3, 8))
        term.negate()
        head = [h for h in w.terms() if w.get_operands(h)][0]
        assert (
            len(w) == 8
            and isinstance(bit_and := head, BitwiseAnd)
            and len(bit_and.operands) == 2
            and any(isinstance(neg_3 := op, BitwiseNegate) for op in bit_and.operands)
            and any(isinstance(bit_or := op, BitwiseOr) for op in bit_and.operands)
            and neg_3.operand == w.constant(3, 8)
            and all(isinstance(op, BitwiseNegate) for op in bit_or.operands)
            and {op.operand for op in bit_or.operands} == {w.constant(1, 8), w.constant(2, 8)}
        )

    def test_negate_partially(self):
        w = World()
        term = w.bitwise_or(operand := w.bitwise_and(w.constant(1, 8), w.constant(2, 8)), w.constant(3, 8))
        term.negate(1)
        head = [h for h in w.terms() if w.get_operands(h)][0]
        assert World.compare(
            head, w.bitwise_and(w.bitwise_negate(w.bitwise_and(w.constant(1, 8), w.constant(2, 8))), w.bitwise_negate(w.constant(3, 8)))
        )

    def test_negate_xor(self):
        w = World()
        term = w.bitwise_and(operand := w.bitwise_xor(w.constant(1, 8), w.constant(2, 8)), w.constant(3, 8))
        operand.negate()
        assert (
            len(w)
            == 6
            # and isinstance(bit_or := head, BitwiseOr)
            # and bit_or.operands[0] == w.constant(3, 8)
            # and isinstance(sec_or := bit_or.operands[1], BitwiseOr)
            # and all(isinstance(op, BitwiseNegate) for op in sec_or.operands)
        )


class TestShiftLeft:
    """Test ShiftLeft."""

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(<< 8@8 1@16)", Constant(WorldType, 16, 16)),
            ("(<< 8@8 1@16 2@16)", Constant(WorldType, 64, 16)),
        ],
    )
    def test_eval(self, test: str, result: Constant):
        """Test the eval method of ShiftLeft."""
        w = World()
        operation = w.from_string(test)
        assert operation.eval(operation.operands) == result

    def test_no_float(self):
        """Test that ShiftLeft does not take a floating point number."""
        w = World()
        with pytest.raises(Exception):
            w.from_string("(<< 8.5@8 1@16)")

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(<< x@8 y@8)", "(<< y@8 x@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert not w.compare(obj1, obj2)


class TestShiftRight:
    """Test ShiftRight."""

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(>> 16@8 1@16)", Constant(WorldType, 8, 16)),
            ("(>> 128@16 1@8 2@8)", Constant(WorldType, 16, 16)),
        ],
    )
    def test_eval(self, test: str, result: Constant):
        """Test the eval method of ShiftRight."""
        w = World()
        operation = w.from_string(test)
        assert operation.eval(operation.operands) == result

    def test_no_float(self):
        """Test that ShiftRight does not take a floating point number."""
        w = World()
        with pytest.raises(Exception):
            w.from_string("(>> 8.5@8 1@16)")

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(>> x@8 y@8)", "(>> y@8 x@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert not w.compare(obj1, obj2)
