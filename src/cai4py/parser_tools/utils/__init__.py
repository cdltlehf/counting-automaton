"""Utility functions for working with regex patterns."""

from re import escape
from typing import Any, Callable, Iterable, Optional, TypedDict, TypeVar

from .. import dfs
from .. import fold
from .. import get_operand_and_children
from .. import in_to_string
from .. import to_string
from ..constants import *  # pylint: disable=unused-wildcard-import, wildcard-import
from ..parser import parse
from ..re import SubPattern

PREDICATE_OPCODES = frozenset({ANY, NOT_LITERAL, IN, LITERAL})
SUBPATTERN_OPCODES = frozenset({ATOMIC_GROUP, SUBPATTERN})
REPEAT_OPCODES = frozenset({MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT})
EXTENDED_OPCODES = frozenset({GROUPREF_EXISTS, GROUPREF, ASSERT, ASSERT_NOT})


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


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


def counting_height(tree: SubPattern) -> int:
    def f(x: Optional[tuple[NamedIntConstant, Any]], ys: Iterable[int]) -> int:
        if x is None:
            return max(ys, default=0)
        opcode, _ = x
        if opcode in {MIN_REPEAT, MAX_REPEAT}:
            return max(ys, default=0) + 1
        return max(ys, default=0)

    return fold(f, tree)


def flatten(
    x: Optional[tuple[NamedIntConstant, Any]], ys: Iterable[str]
) -> str:
    """Modified normalizer to allow for counter expansion."""

    if x is None:
        return "".join(ys)
    opcode, operand = x

    if opcode is LITERAL:
        return f"{escape(operand)}"

    elif opcode is ANY:
        return "."

    elif opcode is NOT_LITERAL:
        _, [(_, c)] = x
        return f"[^{escape(chr(c))}]"

    elif opcode is IN:
        _, [(_, zs)] = x
        return in_to_string(zs)

    elif opcode is BRANCH:
        return f"(?:{'|'.join(ys)})"

    elif opcode is SUBPATTERN:
        return f"({''.join(ys)})"

    elif opcode in {MIN_REPEAT, MAX_REPEAT}:
        n, m = operand

        if m is not MAXREPEAT:
            m = min(m, 65535)
        n = min(n, 65535)

        ys = list(ys)
        ys_str = ys[0] if len(ys) == 1 else "".join(ys)
        ys_str = ys_str if len(ys_str) == 1 else f"(?:{ys_str})"

        def _flatten(ys_str: str, n: int) -> str:
            if n == 0:
                return ""
            return f"(?:{ys_str}{_flatten(ys_str, n-1)})?"

        # Expansion of counters
        if m is MAXREPEAT:
            if n == 0:
                return f"{ys_str}*"
            elif n == 1:
                return f"{ys_str}+"
            else:
                return f"{ys_str*n}+"
        else:
            suffix = _flatten(ys_str, m - n)
            return f"{ys_str*n}{suffix}"

    elif opcode in {MAX_QUESTION, MIN_QUESTION, POSSESSIVE_QUESTION}:
        ys = list(ys)
        ys_str = ys[0] if len(ys) == 1 else "".join(ys)
        ys_str = ys_str if len(ys_str) == 1 else f"(?:{ys_str})"
        if opcode is MAX_QUESTION:
            return f"{ys_str}?"
        elif opcode is MIN_QUESTION:
            return f"{ys_str}??"
        elif opcode is POSSESSIVE_QUESTION:
            return f"{ys_str}?+"
        assert False

    elif opcode in {MAX_STAR, MIN_STAR, POSSESSIVE_STAR}:
        ys = list(ys)
        ys_str = ys[0] if len(ys) == 1 else "".join(ys)
        ys_str = ys_str if len(ys_str) == 1 else f"(?:{ys_str})"
        if opcode is MAX_STAR:
            return f"{ys_str}*"
        elif opcode is MIN_STAR:
            return f"{ys_str}*?"
        elif opcode is POSSESSIVE_STAR:
            return f"{ys_str}*+"
        assert False

    elif opcode in {MAX_PLUS, MIN_PLUS, POSSESSIVE_PLUS}:
        ys = list(ys)
        ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"
        if opcode is MAX_PLUS:
            return f"{ys_str}+"
        elif opcode is MIN_PLUS:
            return f"{ys_str}+?"
        elif opcode is POSSESSIVE_PLUS:
            return f"{ys_str}++"
        assert False

    else:

        raise NotImplementedError(f"Unknown opcode: {opcode}")


def flatten_inner_quantifiers(tree: SubPattern) -> SubPattern:
    if counting_height(tree) < 2:
        return tree
    pattern = quantifier_fold(flatten, tree)
    return parse(pattern)


def flatten_quantifiers(tree: SubPattern) -> SubPattern:
    if counting_height(tree) == 0:
        return tree

    pattern = fold(flatten, tree)
    return parse(pattern)


def quantifier_fold(
    func: Callable[
        [Optional[tuple[NamedIntConstant, Any]], Iterable[str]], str
    ],
    tree: SubPattern,
) -> str:
    def _quantifier_fold(tree: SubPattern) -> Iterable[str]:
        for node in tree:
            opcode, _ = node
            operand, children = get_operand_and_children(node)

            # Perform counter expansion whilst retaining outer counters
            if opcode in {MIN_REPEAT, MAX_REPEAT}:
                n, m = operand
                inner = to_string(flatten_quantifiers(children[0]))

                if m is MAXREPEAT:
                    yield f"(?:{inner}){{{n},}}"
                else:
                    yield f"(?:{inner}){{{n},{m}}}"
            else:
                yield func(
                    (opcode, operand),
                    (quantifier_fold(func, child) for child in children),
                )

    return func(None, _quantifier_fold(tree))
