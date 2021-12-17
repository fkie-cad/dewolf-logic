"""String parser generating World objects."""
import os

from lark import Lark  # type: ignore

grammar = lambda t: Lark.open(os.path.join(os.path.dirname(__file__), "grammar.lark"), parser="lalr", transformer=t)
