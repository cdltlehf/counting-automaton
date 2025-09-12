"""Utility functions for working with regex patterns."""

from typing import Any, Union, Iterable, Optional, Literal

from .. import dfs
from .. import fold
from ..constants import *  # pylint: disable=unused-wildcard-import, wildcard-import
from ..re import SubPattern

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


def counting_height(tree: SubPattern) -> int:
    def f(x: Optional[tuple[NamedIntConstant, Any]], ys: Iterable[int]) -> int:
        if x is None:
            return max(ys, default=0)
        opcode, _ = x
        if opcode in {MIN_REPEAT, MAX_REPEAT}:
            return max(ys, default=0) + 1
        return max(ys, default=0)

    return fold(f, tree)


def _expand_inner_counters(
    subexpr: Union[SubPattern, list], in_counter: bool
) -> Union[SubPattern, list]:
    """Remove nested counters by expanding the inner counters"""
    subexpr_is_subpat = isinstance(subexpr, SubPattern)
    tokens = subexpr.data if isinstance(subexpr, SubPattern) else subexpr
    updated_tokens = []

    for op, av in tokens:
        if op in (MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT):
            # av is (min, max, subpattern)
            lower_b, upper_b, subexpr = av
            is_counter = not (lower_b == 0 and upper_b == 1) and not (
                (lower_b == 0 or lower_b == 1) and upper_b == MAXREPEAT
            )  # not ?, *, or +
            assert isinstance(lower_b, int)
            subexpr = _expand_inner_counters(
                subexpr, in_counter=in_counter or is_counter
            )

            if is_counter and in_counter:
                expansion = []
                for i in range(lower_b):
                    if isinstance(subexpr, list):
                        expansion.extend(subexpr)
                    else:
                        expansion.extend([subexpr])
                if upper_b != MAXREPEAT:
                    assert isinstance(upper_b, int)
                    for i in range(upper_b - lower_b):
                        expansion += [
                            (
                                (
                                    MAX_QUESTION
                                    if op is MAX_REPEAT
                                    else MIN_QUESTION
                                ),
                                subexpr,
                            )
                        ]
                else:
                    expansion += [(op, (0, MAXREPEAT, subexpr))]
                updated_tokens.append(expansion)
            else:
                updated_tokens.append((op, (lower_b, upper_b, subexpr)))
        elif op is SUBPATTERN:
            # av is (groupnum, add_flags, del_flags, subpattern)
            groupnum, add_flags, del_flags, subexpr = av
            subexpr = _expand_inner_counters(subexpr, in_counter)
            updated_tokens.append(
                (op, (groupnum, add_flags, del_flags, subexpr))
            )
        elif op is BRANCH:
            x, branches = av
            branches = [
                _expand_inner_counters(branch, in_counter=in_counter)
                for branch in branches
            ]
            updated_tokens.append((op, (x, branches)))
        elif op in (ASSERT, ASSERT_NOT):
            d, subexpr = av
            subexpr = _expand_inner_counters(subexpr, in_counter=in_counter)
            updated_tokens.append((op, (d, subexpr)))
        elif op in (LITERAL, NOT_LITERAL, ANY, IN, RANGE, CATEGORY, AT, NEGATE):
            updated_tokens.append((op, av))
        elif op in (
            MAX_PLUS,
            MAX_QUESTION,
            MAX_STAR,
            MIN_PLUS,
            MIN_QUESTION,
            MIN_STAR,
            POSSESSIVE_STAR,
            POSSESSIVE_PLUS,
            POSSESSIVE_QUESTION,
            ATOMIC_GROUP,
        ):
            av = _expand_inner_counters(av, in_counter=in_counter)
            updated_tokens.append((op, av))
        else:
            raise RuntimeError(f"Unhandled op: {op}")

        if isinstance(av, SubPattern):
            av = _expand_inner_counters(av, in_counter=in_counter)
            updated_tokens.append((op, av))

        if isinstance(av, list):
            av = _expand_inner_counters(av, in_counter=in_counter)
            updated_tokens.append((op, av))

    flattened_token_list = []
    for token in updated_tokens:
        if isinstance(token, list):
            flattened_token_list.extend(token)
        else:
            flattened_token_list.append(token)
    updated_tokens = flattened_token_list

    if subexpr_is_subpat:
        assert subexpr is SubPattern
        subexpr.data = updated_tokens
    else:
        subexpr = updated_tokens
    return subexpr


def _expand_outer_counters(subexpr):
    """Remove nested counters by expanding the outer counters"""
    subexpr_is_subpat = isinstance(subexpr, SubPattern)
    tokens = subexpr.data if isinstance(subexpr, SubPattern) else subexpr
    updated_tokens = []
    contains_counter = False
    for op, av in tokens:
        if op in (MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT):
            # av is (min, max, subpattern)
            lower_b, upper_b, subexpr = av
            is_counter = not (lower_b == 0 and upper_b == 1) and not (
                (lower_b == 0 or lower_b == 1) and upper_b == MAXREPEAT
            )  # not ?, *, or +
            assert isinstance(lower_b, int)
            subexpr, subexpr_contains_counter = _expand_outer_counters(subexpr)

            if is_counter and subexpr_contains_counter:
                expansion = []
                for i in range(lower_b):
                    if isinstance(subexpr, list):
                        expansion.extend(subexpr)
                    else:
                        expansion.extend([subexpr])
                if upper_b != MAXREPEAT:
                    assert isinstance(upper_b, int)
                    for i in range(upper_b - lower_b):
                        if op is MAX_REPEAT:
                            expansion += [(MAX_QUESTION, subexpr)]
                        else:
                            expansion += [(MIN_QUESTION, subexpr)]
                else:
                    expansion += [(op, (0, MAXREPEAT, subexpr))]
                updated_tokens.append(expansion)
            else:
                updated_tokens.append((op, (lower_b, upper_b, subexpr)))
            contains_counter = True
        elif op is SUBPATTERN:
            # av is (groupnum, add_flags, del_flags, subpattern)
            groupnum, add_flags, del_flags, subexpr = av
            subexpr, contains_counter = _expand_outer_counters(subexpr)
            updated_tokens.append(
                (op, (groupnum, add_flags, del_flags, subexpr))
            )
        elif op is BRANCH:
            x, branches = av
            new_branches = []
            for branch in branches:
                new_branch, branch_contains_counter = _expand_outer_counters(
                    branch
                )
                new_branches += [new_branch]
                contains_counter = contains_counter or branch_contains_counter
            updated_tokens.append((op, (x, new_branches)))
        elif op in (ASSERT, ASSERT_NOT):
            d, subexpr = av
            subexpr, contains_counter = _expand_outer_counters(subexpr)
            updated_tokens.append((op, (d, subexpr)))
        elif op in (LITERAL, NOT_LITERAL, ANY, IN, RANGE, CATEGORY, AT, NEGATE):
            updated_tokens.append((op, av))
        elif op in (
            MAX_PLUS,
            MAX_QUESTION,
            MAX_STAR,
            MIN_PLUS,
            MIN_QUESTION,
            MIN_STAR,
            POSSESSIVE_STAR,
            POSSESSIVE_PLUS,
            POSSESSIVE_QUESTION,
            ATOMIC_GROUP,
        ):
            av, av_contains_counter = _expand_outer_counters(av)
            contains_counter = contains_counter or av_contains_counter
            updated_tokens.append((op, av))
        else:
            print(subexpr)
            raise RuntimeError(f"Unhandled op: {op}")

        if isinstance(av, SubPattern):
            av, av_contains_counter = _expand_outer_counters(av)
            contains_counter = contains_counter or av_contains_counter
            updated_tokens.append((op, av))

        if isinstance(av, list):
            av, av_contains_counter = _expand_outer_counters(av)
            contains_counter = contains_counter or av_contains_counter
            updated_tokens.append((op, av))
    flattened_token_list = []
    for token in updated_tokens:
        if isinstance(token, list):
            flattened_token_list.extend(token)
        else:
            flattened_token_list.append(token)
    updated_tokens = flattened_token_list
    if subexpr_is_subpat:
        assert isinstance(subexpr, SubPattern)
        subexpr.data = updated_tokens
    else:
        subexpr = updated_tokens
    return subexpr, contains_counter


def expand_nested_counters(tree, method: Literal["inner", "outer"]):
    match method:
        case "inner":
            return _expand_inner_counters(tree, in_counter=False)
        case "outer":
            tree, contains_counter = _expand_outer_counters(tree)
            return tree
