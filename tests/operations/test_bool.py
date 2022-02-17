"""Module implementing tests for boolean operations."""
from typing import Optional

import pytest

from simplifier.world.world import World


def run_test(test: str, result: str) -> bool:
    """Run the given test case."""
    w = World()
    a, b = w.variable("a", 32), w.variable("b", 32)
    test_operation = w.from_string(test)
    w.define(a, test_operation)
    w.define(b, w.from_string(result))
    test_operation.simplify()  # type: ignore
    return w.compare(a, b)


def run_test2(test: str, result: str) -> bool:
    """Run the given test case."""
    w = World()
    a, b = w.variable("a", 32), w.variable("b", 32)
    test_operation = w.from_string(test)
    w.define(a, test_operation)
    w.define(b, w.from_string(result))
    a.simplify()  # type: ignore
    return w.compare(a, b)


def run_negation_test(test: str, result: str, level: Optional[int]) -> bool:
    """Run the given test case."""
    w = World()
    a, b = w.variable("a", 32), w.variable("b", 32)
    test_operation = w.from_string(test)
    w.define(a, test_operation)
    w.define(b, w.from_string(result))
    test_operation.negate(level)  # type: ignore
    return w.compare(a, b)


class TestNot:
    """Class testing the boolean Not-Operation."""

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(! (! x@1))", "x@1"),
            ("(! (! (! x@1)))", "(! x@1)"),
        ],
    )
    def test_nesting(self, test: str, result: str):
        assert run_test(test, result)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(! (== x@1 0@1))", "(!= x@1 0@1)"),
        ],
    )
    def test_factorize(self, test: str, result: str):
        assert run_test(test, result)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(! (| (== x@32 0@32) (!= x@32 0@32)))", "0@32"),
            ("(! (| (!= x@32 0@32) (== x@32 0@32)))", "0@32"),
        ],
    )
    def test_folding(self, test: str, result: str):
        assert run_test2(test, result)

    @pytest.mark.parametrize(
        "test, result",
        [("(! (== 1@32 1@32))", "0@32"), ("(! (== 0@32 0@32))", "0@32"), ("(! (== x@32 1@32 1@32 1@32 1@32))", "(! (== x@32 1@32))")],
    )
    def test_multifold(self, test: str, result: str):
        w = World()
        test_operation = w.from_string(test)
        a = test_operation.simplify()  # type: ignore
        return w.compare(a, w.from_string(result))


class TestRelations:
    """Class implementing tests for Relation Operations."""

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(== 1@1 1@1)", "1@1"),
            ("(== 1@1 0@1)", "0@1"),
            ("(== 1@32 1@32)", "1@32"),
            ("(== 0@32 0@32)", "1@32"),
            ("(== 1@1 0@1 x@1)", "0@1"),
            ("(== x@1 x@1)", "1@1"),
            ("(u<= x@1 x@1)", "1@1"),
            ("(s>= x@1 x@1)", "1@1"),
        ],
    )
    def test_equivalence(self, test: str, result: str):
        assert run_test(test, result)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(== 1@1 0@1)", "0@1"),
            ("(== 1@1 0@1 x@1)", "0@1"),
            ("(u< 0@1 x@1 0@1)", "0@1"),
            ("(u< 0@1 0@1 x@1)", "0@1"),
            ("(u<= x@1 x@1)", "1@1"),
            ("(s>= x@1 x@1)", "1@1"),
        ],
    )
    def test_failfast(self, test: str, result: str):
        assert run_test(test, result)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(u< 1@1 1@1)", "0@1"),
            ("(u< 2@2 1@2)", "0@1"),
            ("(u< 1@2 2@2)", "1@1"),
            ("(s< 1@1 1@2)", "1@1"),
            ("(s<= 4@5 6@5)", "1@1"),
            ("(s<= -1@2 1@2)", "1@1"),
            ("(u<= -1@2 1@2)", "0@1"),
            ("(!= 1@1 1@1)", "0@1"),
            ("(!= 1@1 x@1 1@8)", "0@1"),
            ("(s<= 1@1 1@2 2@8)", "1@1"),
            ("(!= 1@1 1@1)", "0@1"),
            ("(!= x@1 x@1)", "0@1"),
        ],
    )
    def test_inequality(self, test: str, result: str):
        assert run_test(test, result)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(== 0@1 x@8)", "(== 0@1 x@8)"),
            ("(== 1@1 1@1 x@1)", "(== 1@1 x@1)"),
            ("(u< 1@8 2@8 v@8)", "(u< 2@8 v@8)"),
            ("(u< 0@1 1@1 v@8 8@8 255@8)", "(u< 1@1 v@8 8@8)"),
            ("(u<= 0@1 1@1 v@8 8@8 255@8)", "(u<= 1@1 v@8 8@8)"),
            ("(u> x@8 8@8 7@8 v@8 5@8 4@8)", "(u> x@8 8@8 7@8 v@8 5@8)"),
            ("(u>= x@8 8@8 8@8 y@8)", "(u>= x@8 8@8 y@8)"),
        ],
    )
    def test_simplify(self, test: str, result: str):
        assert run_test(test, result)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(| (== x@32 0@32) (!= x@32 0@32))", "1@32"),
            ("(| (!= x@32 0@32) (== x@32 0@32))", "1@32"),
        ],
    )
    def test_complex_fold(self, test: str, result: str):
        assert run_test2(test, result)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(== 0@4 x@4)", "(== 0@4 x@4)"),
            ("(== 1@4 y@4 x@4)", "(& (== 1@4 y@4) (== y@4 x@4))"),
            ("(!= 0@4 x@4)", "(!= 0@4 x@4)"),
            ("(!= 1@4 y@4 x@4)", "(& (!= 1@4 y@4) (!= y@4 x@4))"),
            ("(u> 3@4 x@4)", "(u> 3@4 x@4)"),
            ("(u> 5@4 y@4 x@4)", "(& (u> 5@4 y@4) (u> y@4 x@4))"),
            ("(s> 3@4 x@4)", "(s> 3@4 x@4)"),
            ("(s> 5@4 y@4 x@4)", "(& (s> 5@4 y@4) (s> y@4 x@4))"),
            ("(u>= 3@4 x@4)", "(u>= 3@4 x@4)"),
            ("(u>= 5@4 y@4 x@4 z@4)", "(& (u>= 5@4 y@4) (u>= y@4 x@4) (u>= x@4 z@4))"),
            ("(s>= 3@4 x@4)", "(s>= 3@4 x@4)"),
            ("(s>= 5@4 y@4 x@4)", "(& (s>= 5@4 y@4) (s>= y@4 x@4))"),
            ("(u< 3@4 x@4)", "(u< 3@4 x@4)"),
            ("(u< z@4 5@4 y@4 x@4)", "(& (u< z@4 5@4) (u< 5@4 y@4) (u< y@4 x@4))"),
            ("(s< 3@4 x@4)", "(s< 3@4 x@4)"),
            ("(s< 5@4 y@4 x@4)", "(& (s< 5@4 y@4) (s< y@4 x@4))"),
            ("(u<= 3@4 x@4)", "(u<= 3@4 x@4)"),
            ("(u<= 1@4 y@4 x@4 z@4)", "(& (u<= 1@4 y@4) (u<= y@4 x@4) (u<= x@4 z@4))"),
            ("(s<= 3@4 x@4)", "(s<= 3@4 x@4)"),
            ("(s<= -1@4 y@4 x@4)", "(& (s<= -1@4 y@4) (s<= y@4 x@4))"),
        ],
    )
    def test_split(self, test: str, result: str):
        w = World()
        a, b = w.variable("a", 1), w.variable("b", 1)
        test_operation = w.from_string(test)
        w.define(a, test_operation)
        w.define(b, w.from_string(result))
        test_operation.split()  # type: ignore
        assert w.compare(a, b)


class TestNegation:
    @pytest.mark.parametrize(
        "test, result",
        [
            ("(== 0@8 x@8)", "(!= 0@8 x@8)"),
            ("(== 0@8 x@8 y@8 )", "(!= 0@8 x@8 y@8)"),
            ("(!= 1@1 x@1)", "(== 1@1 x@1)"),
            ("(u< 1@8 v@8 7@8)", "(u>= 1@8 v@8 7@8)"),
            ("(u<= v@8 8@8)", "(u> v@8 8@8)"),
            ("(u> x@8 8@8 v@8)", "(u<= x@8 8@8 v@8)"),
            ("(u>= x@8 8@8)", "(u< x@8 8@8)"),
            ("(s< 1@8 v@8 7@8)", "(s>= 1@8 v@8 7@8)"),
            ("(s<= v@8 8@8)", "(s> v@8 8@8)"),
            ("(s> x@8 8@8 v@8)", "(s<= x@8 8@8 v@8)"),
            ("(s>= x@8 8@8)", "(s< x@8 8@8)"),
        ],
    )
    def test_negation_no_level(self, test: str, result: str):
        assert run_negation_test(test, result, None)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(== 0@8 x@8)", "(~ (== 0@8 x@8))"),
            ("(!= 1@1 x@1)", "(~ (!= 1@1 x@1))"),
            ("(u< 1@8 v@8 7@8)", "(~ (u< 1@8 v@8 7@8))"),
            ("(u<= v@8 8@8)", "(~ (u<= v@8 8@8))"),
            ("(s> x@8 8@8 v@8)", "(~ (s> x@8 8@8 v@8))"),
            ("(s>= x@8 8@8)", "(~ (s>= x@8 8@8))"),
        ],
    )
    def test_negation_level_0(self, test: str, result: str):
        assert run_negation_test(test, result, 0)

    @pytest.mark.parametrize(
        "test, level, result",
        [
            ("(== 0@8 x@8)", 1, "(!= 0@8 x@8)"),
            ("(== 0@8 x@8 y@8)", 2, "(!= 0@8 x@8 y@8)"),
            ("(!= 1@1 x@1)", 3, "(== 1@1 x@1)"),
            ("(u> x@8 8@8 v@8)", 10, "(u<= x@8 8@8 v@8)"),
            ("(u>= x@8 8@8)", 5, "(u< x@8 8@8)"),
            ("(s< 1@8 v@8 7@8)", 20, "(s>= 1@8 v@8 7@8)"),
            ("(s<= v@8 8@8)", 1, "(s> v@8 8@8)"),
        ],
    )
    def test_negation_level_greater_0(self, test: str, level: int, result: str):
        assert run_negation_test(test, result, level)
