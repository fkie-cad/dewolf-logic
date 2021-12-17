"""Module implementing tests for arithmetic operations."""
from typing import List

import pytest

from simplifier.operations.arithmetic import *
from simplifier.world.nodes import Constant, WorldType
from simplifier.world.world import World


class TestAddition:
    """Class implementing tests for both SignedAddition and UnsignedAddition."""

    @pytest.mark.parametrize(
        "operation, operands, result",
        [
            (UnsignedAddition(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, 1, 8)], Constant(WorldType, 2, 8)),
            (UnsignedAddition(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, -1, 8)], Constant(WorldType, 0, 8)),
            (UnsignedAddition(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, 255, 8)], Constant(WorldType, 0, 8)),
            (UnsignedAddition(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, 255, 16)], Constant(WorldType, 256, 16)),
            (SignedAddition(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, -1, 8)], Constant(WorldType, 0, 8)),
            (SignedAddition(WorldType), [Constant(WorldType, -1, 8), Constant(WorldType, -5, 8)], Constant(WorldType, -6, 8)),
            (
                SignedAddition(WorldType),
                [Constant(WorldType, 42, 8), Constant(WorldType, -40, 8), Constant(WorldType, 40, 16)],
                Constant(WorldType, 42, 16),
            ),
        ],
    )
    def test_eval(self, operation: BaseAddition, operands: List[Constant], result: Constant):
        """Test the eval method of UnsignedAddition and SignedAddition."""
        assert operation.eval(operands) == result

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(u+ x@8 8@8 8@16)", "(u+ x@8 16@16)"),
            ("(s+ x@8 8@8 8@16)", "(s+ x@8 16@16)"),
            ("(s+ x@8 -1@8 8@8)", "(u+ x@8 7@18)"),
        ],
    )
    def test_fold_constants(self, test: str, result: str):
        w = World()
        a, b = w.variable("a", 32), w.variable("b", 32)
        test_operation = w.from_string(test)
        w.define(a, test_operation)
        w.define(b, w.from_string(result))
        test_operation.fold()  # type: ignore
        assert w.compare(a, b)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(u+ x@8 x@8 y@8)", "(u+ (u* x@8 (u+ 1@8 1@8 )) y@8)"),
            ("(u+ (u* x@8 8@8) (u* y@16 8@8))", "(u* 8@8 (u+ x@8 y@16))"),
            ("(u+ (u* x@8 8@8) (u* x@8 3@8) z@8)", "(u+ (u* x@8 (u+ 8@8 3@8)) z@8)"),
            ("(u+ (u* 3@8 x@8) x@8 (u* 3@8 y@8))", "(u+ (u* 3@8 (u+ x@8 y@8)) x@8)"),
            ("(u+ (s* x@8 w@8 8@8) (s* x@8 3@8) z@8 (s* y@8 (u+ x@8 1@8)))", "(u+ (s* x@8 (u+ (s* w 8@8) 3@8)) z@8 (s* y@8 (u+ x@8 1@8)))"),
        ],
    )
    def test_factorize(self, test: str, result: str):
        w = World()
        a, b = w.variable("a", 32), w.variable("b", 32)
        test_operation = w.from_string(test)
        print("factorized " + str(test_operation.factorize()))
        w.define(a, test_operation.factorize())  # type: ignore
        w.define(b, w.from_string(result))
        assert w.compare(a, b)

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(u+ x@8 y@8)", "(u+ y@8 x@8)"),
            ("(s+ x@8 y@8)", "(s+ y@8 x@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert w.compare(obj1, obj2)


class TestSubtraction:
    """Class implementing tests for both SignedSubtraction and UnsignedSubtraction."""

    @pytest.mark.parametrize(
        "operation, operands, result",
        [
            (UnsignedSubtraction(WorldType), [Constant(WorldType, 2, 8), Constant(WorldType, 1, 8)], Constant(WorldType, 1, 8)),
            (UnsignedSubtraction(WorldType), [Constant(WorldType, 255, 8), Constant(WorldType, 1, 16)], Constant(WorldType, 254, 16)),
            (UnsignedSubtraction(WorldType), [Constant(WorldType, 5.0, 8), Constant(WorldType, 1, 16)], Constant(WorldType, 4.0, 16)),
            (SignedSubtraction(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, -1, 8)], Constant(WorldType, 2, 8)),
            (SignedSubtraction(WorldType), [Constant(WorldType, -1, 8), Constant(WorldType, 5, 8)], Constant(WorldType, -6, 8)),
            (
                SignedSubtraction(WorldType),
                [Constant(WorldType, 1, 8), Constant(WorldType, -10, 8), Constant(WorldType, 20, 16)],
                Constant(WorldType, -9, 16),
            ),
        ],
    )
    def test_eval(self, operation: BaseSubtraction, operands: List[Constant], result: Constant):
        """Test the eval method of UnsignedSubtraction and SignedSubtraction."""
        assert operation.eval(operands) == result

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(u- x@8 8@8 3@16)", "(u- x@8 5@16)"),
            ("(s- x@8 -1@8 7@16)", "(s- x@8 -8@16)"),
            ("(s- x@8 16@8 -5@8)", "(u- x@8 21@18)"),
        ],
    )
    def test_fold_constants(self, test: str, result: str):
        w = World()
        a, b = w.variable("a", 32), w.variable("b", 32)
        test_operation = w.from_string(test)
        w.define(a, test_operation)
        w.define(b, w.from_string(result))
        test_operation.fold()  # type: ignore
        assert w.compare(a, b)

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(u- x@8 x@8 y@8)", "(u- (u* x@8 (u- 1@8 1@8 )) y@8)"),
            ("(u- (u* x@8 8@8) (u* y@16 8@8))", "(u* 8@8 (u- x@8 y@16))"),
            ("(u- (u* x@8 8@8) (u* x@8 3@8) z@8)", "(u- (u* x@8 (u- 8@8 3@8)) z@8)"),
            ("(u- (s* x@8 w@8 8@8) (s* x@8 3@8) z@8 (s* y@8 (u+ x@8 1@8)))", "(u- (s* x@8 (u- (s* w 8@8) 3@8)) z@8 (s* y@8 (u+ x@8 1@8)))"),
        ],
    )
    def test_factorize(self, test: str, result: str):
        w = World()
        a, b = w.variable("a", 32), w.variable("b", 32)
        test_operation = w.from_string(test)
        w.define(a, test_operation.factorize())  # type: ignore
        w.define(b, w.from_string(result))
        assert w.compare(a, b)

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(u- x@8 y@8)", "(u- y@8 x@8)"),
            ("(s- x@8 y@8)", "(s- y@8 x@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert not w.compare(obj1, obj2)


class TestModulo:
    """Class implementing tests for both SignedModulo and UnsignedModulo."""

    @pytest.mark.parametrize(
        "operation, operands, result",
        [
            (UnsignedModulo(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, 1, 8)], Constant(WorldType, 0, 8)),
            (UnsignedModulo(WorldType), [Constant(WorldType, 255, 8), Constant(WorldType, 4, 8)], Constant(WorldType, 3, 8)),
            (UnsignedModulo(WorldType), [Constant(WorldType, 255, 16), Constant(WorldType, 4, 8)], Constant(WorldType, 3, 16)),
            (SignedModulo(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, -1, 8)], Constant(WorldType, 0, 8)),
            (SignedModulo(WorldType), [Constant(WorldType, -10, 8), Constant(WorldType, -4, 8)], Constant(WorldType, -2, 8)),
            (
                SignedModulo(WorldType),
                [Constant(WorldType, 100, 8), Constant(WorldType, 12, 8), Constant(WorldType, 3, 16)],
                Constant(WorldType, 1, 16),
            ),
        ],
    )
    def test_eval(self, operation: BaseModulo, operands: List[Constant], result: Constant):
        """Test the eval method of UnsignedModulo and SignedModulo."""
        assert operation.eval(operands) == result

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(u% x@8 8@8 3@16)", "(u% x@8 2@16)"),
            ("(s% x@8 8@8 8@16)", "(u% x@8 0@16)"),
            ("(s% x@8 -1@8 8@16)", "(s% x@8 -1@16)"),
        ],
    )
    def test_fold_constants(self, test: str, result: str):
        w = World()
        a, b = w.variable("a", 32), w.variable("b", 32)
        test_operation = w.from_string(test)
        w.define(a, test_operation)
        w.define(b, w.from_string(result))
        test_operation.fold()  # type: ignore
        assert w.compare(a, b)

    def test_float(self):
        """Test that trying to do modulo on a floating point number gives exception as it should only be defined for integers"""
        w = World()
        with pytest.raises(Exception):
            a = w.from_string("(u% 8.0@8 3@8)")

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(u% x@8 y@8)", "(u% y@8 x@8)"),
            ("(s% x@8 y@8)", "(s% y@8 x@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert not w.compare(obj1, obj2)


class TestMultiplication:
    """Class implementing tests for both SignedMultiplication and UnsignedMultiplication."""

    @pytest.mark.parametrize(
        "operation, operands, result",
        [
            (UnsignedMultiplication(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, 1, 8)], Constant(WorldType, 1, 8)),
            (UnsignedMultiplication(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, -1, 8)], Constant(WorldType, -1, 8)),
            (UnsignedMultiplication(WorldType), [Constant(WorldType, 2, 8), Constant(WorldType, 255, 8)], Constant(WorldType, 254, 8)),
            (UnsignedMultiplication(WorldType), [Constant(WorldType, 2, 8), Constant(WorldType, 255, 16)], Constant(WorldType, 510, 16)),
            (SignedMultiplication(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, -1, 8)], Constant(WorldType, -1, 8)),
            (SignedMultiplication(WorldType), [Constant(WorldType, -1, 8), Constant(WorldType, -5, 8)], Constant(WorldType, 5, 8)),
            (
                SignedMultiplication(WorldType),
                [Constant(WorldType, 42, 8), Constant(WorldType, -7, 8), Constant(WorldType, 7, 16)],
                Constant(WorldType, -2058, 16),
            ),
        ],
    )
    def test_eval(self, operation: BaseMultiplication, operands: List[Constant], result: Constant):
        """Test the eval method of UnsignedMultiplication and SignedMultiplication."""
        print([operand.signed for operand in operands])
        assert operation.eval(operands) == result

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(u* x@8 8@8 8@16)", "(u* x@8 64@16)"),
            ("(s* x@8 8@8 8@16)", "(s* x@8 64@16)"),
            ("(s* x@8 -8@8 -8@16)", "(s* x@8 64@16)"),
            ("(s* x@8 -1@8 8@16)", "(s* x@8 -8@16)"),
        ],
    )
    def test_fold_constants(self, test: str, result: str):
        w = World()
        a, b = w.variable("a", 32), w.variable("b", 32)
        test_operation = w.from_string(test)
        w.define(a, test_operation)
        w.define(b, w.from_string(result))
        test_operation.fold()  # type: ignore
        assert w.compare(a, b)

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(u* x@8 y@8)", "(u* y@8 x@8)"),
            ("(s* x@8 y@8)", "(s* y@8 x@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert w.compare(obj1, obj2)


class TestDivision:
    """Class implementing tests for both SignedDivision and UnsignedDivision."""

    @pytest.mark.parametrize(
        "operation, operands, result",
        [
            (UnsignedDivision(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, 1, 8)], Constant(WorldType, 1, 8)),
            (UnsignedDivision(WorldType), [Constant(WorldType, 0, 8), Constant(WorldType, 255, 8)], Constant(WorldType, 0, 8)),
            (UnsignedDivision(WorldType), [Constant(WorldType, 1024, 16), Constant(WorldType, 2, 8)], Constant(WorldType, 512, 16)),
            (SignedDivision(WorldType), [Constant(WorldType, 1, 8), Constant(WorldType, -1, 8)], Constant(WorldType, -1, 8)),
            (SignedDivision(WorldType), [Constant(WorldType, -5, 8), Constant(WorldType, -1, 8)], Constant(WorldType, 5, 8)),
            (SignedDivision(WorldType), [Constant(WorldType, -5.0, 8), Constant(WorldType, -1, 8)], Constant(WorldType, 5.0, 8)),
            (
                SignedDivision(WorldType),
                [Constant(WorldType, 4200, 32), Constant(WorldType, -7, 32), Constant(WorldType, 6, 32)],
                Constant(WorldType, -100, 32),
            ),
        ],
    )
    def test_eval(self, operation: BaseDivision, operands: List[Constant], result: Constant):
        """Test the eval method of UnsignedDivision and SignedDivision."""
        assert operation.eval(operands) == result

    @pytest.mark.parametrize(
        "test, result",
        [
            ("(u/ x@8 64@16 8@16)", "(u/ x@8 8@16)"),
            ("(s/ x@8 64@16 8@8)", "(s/ x@8 8@16)"),
            ("(s/ x@8 -8@8 -8@16)", "(u/ x@8 1@16)"),
            ("(s/ x@8 8@16 -1@8)", "(s/ x@8 -8@16)"),
            ("(s/ x@8 8.0@16 -1@8)", "(s/ x@8 -8.0@18)"),
        ],
    )
    def test_fold_constants(self, test: str, result: str):
        w = World()
        a, b = w.variable("a", 32), w.variable("b", 32)
        test_operation = w.from_string(test)
        w.define(a, test_operation)
        w.define(b, w.from_string(result))
        test_operation.fold()  # type: ignore
        assert w.compare(a, b)

    @pytest.mark.parametrize(
        "a, b",
        [
            ("(u/ x@8 y@8)", "(u/ y@8 x@8)"),
            ("(s/ x@8 y@8)", "(s/ y@8 x@8)"),
        ],
    )
    def test_ordering(self, a: str, b: str):
        w = World()
        obj1 = w.from_string(a)
        obj2 = w.from_string(b)
        assert not w.compare(obj1, obj2)
