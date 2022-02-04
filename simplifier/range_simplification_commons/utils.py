"""Module implementing utils for range simplification."""
from __future__ import annotations

from typing import Tuple, Union

from simplifier.world.nodes import Constant


def eval_constant(constant: Constant, is_signed: bool) -> Union[float, int]:
    """Evaluate the value of the given constant depending on the is_signed parameter."""
    return constant.signed if is_signed else constant.unsigned


def smaller(first_constant: Constant, second_constant: Constant, is_signed: bool) -> bool:
    """Evaluate whether the first constant is smaller than the second constant."""
    return eval_constant(first_constant, is_signed) < eval_constant(second_constant, is_signed)


def modify_constant_by(const: Constant, modification: int) -> Constant:
    """Modify the given constant by the given integer and return this modified constant."""
    return Constant(const.world, const.unsigned + modification, const.size)


def get_min_max_number_for_size(size: int, is_signed: bool) -> Tuple[int, int]:
    """Return the minimum and maximum possible value of the given size depending on is_signed."""
    max_number = (1 << (size - int(is_signed))) - 1
    min_number = -max_number - 1 if is_signed else 0
    return min_number, max_number
