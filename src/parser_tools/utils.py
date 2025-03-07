"""Utility functions for working with regex patterns."""

from typing import Any, Iterable, Optional

from . import dfs
from . import fold
from .constants import *  # pylint: disable=unused-wildcard-import, wildcard-import
from .re import SubPattern

PREDICATE_OPCODES = frozenset({ANY, NOT_LITERAL, IN, LITERAL})
SUBPATTERN_OPCODES = frozenset({ATOMIC_GROUP, SUBPATTERN})
REPEAT_OPCODES = frozenset({MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT})
EXTENDED_OPCODES = frozenset({GROUPREF_EXISTS, GROUPREF, ASSERT, ASSERT_NOT})


def is_nullable(tree: SubPattern) -> bool:
    if has_extended_features(tree):
        raise ValueError("Pattern has extended features")

    def f(
        x: Optional[tuple[NamedIntConstant, Any]], ys: Iterable[bool]
    ) -> bool:
        if x is None:
            return all(ys)

        opcode, operand = x
        if opcode in PREDICATE_OPCODES:
            return False
        elif opcode is AT:
            return True
        elif opcode is BRANCH:
            return any(ys)
        elif opcode in REPEAT_OPCODES:
            return operand[0] == 0 or all(ys)
        elif opcode in SUBPATTERN_OPCODES:
            return all(ys)
        else:
            assert False, f"Unknown opcode: {opcode}"

    return fold(f, tree)


def is_problematic(tree: SubPattern) -> bool:
    def f(
        x: Optional[tuple[NamedIntConstant, Any]], ys: Iterable[bool]
    ) -> bool:
        if x is None:
            return any(ys)
        opcode, operand = x
        if opcode in {MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT}:
            _, n = operand
            if n is MAXREPEAT:
                return is_nullable(tree)
        return any(ys)

    return fold(f, tree)


def has_extended_features(tree: SubPattern) -> bool:
    return any(opcode in EXTENDED_OPCODES for opcode, _ in dfs(tree))


def is_literal_sequence(tree: SubPattern) -> bool:
    return all(opcode is LITERAL for opcode, _ in dfs(tree))


def is_anchored_literal_sequence(tree: SubPattern) -> bool:
    return all(
        opcode in {LITERAL, AT} | SUBPATTERN_OPCODES for opcode, _ in dfs(tree)
    )


def is_anchored_predicate_sequence(tree: SubPattern) -> bool:
    return all(
        opcode
        in {LITERAL, ANY, NOT_LITERAL, IN, AT} | {SUBPATTERN, ATOMIC_GROUP}
        for opcode, _ in dfs(tree)
    )


def is_finite_pattern(tree: SubPattern) -> bool:
    for opcode, operand in dfs(tree):
        if opcode in {MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT}:
            _, n = operand
            if n is MAXREPEAT:
                return False
    return True
