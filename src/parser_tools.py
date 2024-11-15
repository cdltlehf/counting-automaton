"""Parser tools for regular expressions."""

# mypy: disable-error-code=import-untyped
# pylint: disable=useless-import-alias

from itertools import chain
from re._constants import _NamedIntConstant as NamedIntConstant
from re._constants import ANY as ANY
from re._constants import ASSERT as ASSERT
from re._constants import ASSERT_NOT as ASSERT_NOT
from re._constants import AT as AT
from re._constants import ATOMIC_GROUP as ATOMIC_GROUP
from re._constants import BRANCH as BRANCH
from re._constants import GROUPREF as GROUPREF
from re._constants import GROUPREF_EXISTS as GROUPREF_EXISTS
from re._constants import IN as IN
from re._constants import LITERAL as LITERAL
from re._constants import MAX_REPEAT as MAX_REPEAT
from re._constants import MAXREPEAT as MAXREPEAT
from re._constants import MIN_REPEAT as MIN_REPEAT
from re._constants import NOT_LITERAL as NOT_LITERAL
from re._constants import POSSESSIVE_REPEAT as POSSESSIVE_REPEAT
from re._constants import SUBPATTERN as SUBPATTERN
from re._parser import State
from re._parser import SubPattern
import re._parser as parser
from typing import Any, Callable, Iterable, Optional, TypeVar

T = TypeVar("T")


PREDICATE_OPCODES = frozenset({LITERAL, ANY, NOT_LITERAL, IN})
ACTION_OPCODES = frozenset({GROUPREF, AT, GROUPREF_EXISTS})
REPEAT_OPCODES = frozenset({MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT})
OPERATION_OPCODES = REPEAT_OPCODES | {BRANCH}
SUBPATTERN_OPCODES = frozenset({SUBPATTERN, ASSERT, ASSERT_NOT, ATOMIC_GROUP})
EXTENDED_OPCODES = frozenset(
    {
        GROUPREF,
        ASSERT,
        ASSERT_NOT,
        ATOMIC_GROUP,
        GROUPREF_EXISTS,
    }
)


def parse(pattern: str) -> SubPattern:
    return parser.parse(pattern)


def get_operand_and_children(node: SubPattern) -> tuple[Any, list[Any]]:
    opcode, value = node
    if opcode in PREDICATE_OPCODES:
        if opcode is LITERAL:
            return chr(value), []
        else:
            return SubPattern(State(), [node]), []
    elif opcode in ACTION_OPCODES:
        return SubPattern(State(), [node]), []
    elif opcode is SUBPATTERN:
        assert len(value) == 4, value
        a, b, c = value[:3]
        return (a, b, c), [value[3]]
    elif opcode in {MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT}:
        assert len(value) == 3, value
        m, n = value[0], value[1]
        return (m, n), [value[2]]
    elif opcode is BRANCH:
        assert len(value) == 2, value
        assert value[0] is None, value
        return None, value[1]
    elif opcode in {ASSERT, ASSERT_NOT}:
        assert len(value) == 2, value
        assert value[0] in {1, -1}, value
        # value[0] == 1 -> lookahead
        # value[0] == -1 -> lookbehind
        lookbehind = value[0] == 1
        return lookbehind, [value[1]]
    elif opcode is GROUPREF_EXISTS:
        assert len(value) == 3, value
        ref = value[0]
        return ref, [e for e in value[1:] if e is not None]
    elif opcode is ATOMIC_GROUP:
        return None, [value]
    else:
        assert False, f"Unknown opcode: {opcode}"


def fold(
    f: Callable[[Optional[tuple[NamedIntConstant, Any]], Iterable[T]], T],
    tree: SubPattern,
) -> T:

    def _fold(tree: SubPattern) -> Iterable[T]:
        for node in tree:
            opcode, _ = node
            operand, children = get_operand_and_children(node)
            yield f(
                (opcode, operand),
                (fold(f, child) for child in children),
            )

    return f(None, _fold(tree))


def dfs(tree: SubPattern) -> Iterable[tuple[str, Any]]:
    return fold(
        lambda x, ys: chain([] if x is None else [x], chain.from_iterable(ys)),
        tree,
    )


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
        elif opcode in {ATOMIC_GROUP, SUBPATTERN}:
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
        if opcode in REPEAT_OPCODES:
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
        opcode in {PREDICATE_OPCODES, AT} | SUBPATTERN_OPCODES
        for opcode, _ in dfs(tree)
    )


def is_finite_pattern(tree: SubPattern) -> bool:
    for opcode, operand in dfs(tree):
        if opcode in REPEAT_OPCODES:
            _, n = operand
            if n is MAXREPEAT:
                return False
    return True
