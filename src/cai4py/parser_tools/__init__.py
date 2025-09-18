"""Parser tools for regular expressions."""

# mypy: disable-error-code=import-untyped
# pylint: disable=useless-import-alias

from itertools import chain
from re import escape
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, TypeVar
import warnings

from .constants import *
from .parser import parse as parse  # type: ignore
from .re import State
from .re import SubPattern as SubPattern

warnings.simplefilter(action="ignore", category=FutureWarning)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def get_operand_and_children(node: SubPattern) -> tuple[Any, list[Any]]:
    opcode, value = node
    if opcode in {LITERAL, ANY, NOT_LITERAL, RANGE, CATEGORY, IN}:
        if opcode is LITERAL:
            return chr(value), []
        else:
            return SubPattern(State(), [node]), []
    elif opcode in {GROUPREF, AT, GROUPREF_EXISTS}:
        return SubPattern(State(), [node]), []
    elif opcode is SUBPATTERN:
        assert len(value) == 4, value
        a, b, c = value[:3]
        return (a, b, c), [value[3]]
    elif opcode in {MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT}:
        assert len(value) == 3, value
        m, n = value[0], value[1]
        return (m, n), [value[2]]
    elif opcode in {
        MAX_QUESTION,
        MIN_QUESTION,
        POSSESSIVE_QUESTION,
        MAX_STAR,
        MIN_STAR,
        POSSESSIVE_STAR,
        MAX_PLUS,
        MIN_PLUS,
        POSSESSIVE_PLUS,
    }:
        return None, [value]
    elif opcode is BRANCH:
        assert len(value) == 2, value
        assert value[0] is None, value
        return None, value[1]
    elif opcode in {ASSERT, ASSERT_NOT}:
        assert len(value) == 2, value
        assert value[0] in {1, -1}, value
        # value[0] == 1 -> lookahead
        # value[0] == -1 -> lookbehind
        lookbehind = value[0] != 1
        return lookbehind, [value[1]]
    elif opcode is GROUPREF_EXISTS:
        assert len(value) == 3, value
        ref = value[0]
        return ref, [e for e in value[1:] if e is not None]
    elif opcode is ATOMIC_GROUP:
        return None, [value]
    elif opcode is FAILURE:
        raise NotImplementedError(f"Unknown opcode: {opcode}")
    else:
        assert False, f"Unknown opcode: {opcode}"

"""
    Counter expansion of only nested counters. Can be used instead of normalize.
"""
def flatten_inner_quantifiers(tree: SubPattern):
    pattern = quantifier_fold(flatten, tree)
    return parse(pattern)

"""
    Counter expansion of all counters. Can be used instead of normalize.
"""
def flatten_quantifiers(tree: SubPattern) -> SubPattern:
    pattern = fold(flatten, tree)
    return parse(pattern)

"""
    Fold operation to support inner counter expansion.
"""
def quantifier_fold(
    func: Callable[[Optional[tuple[NamedIntConstant, Any]], Iterable[T]], T],
    tree: SubPattern
) -> T:
    def _quantifier_fold(tree: SubPattern) -> Iterable[T]:
        for node in tree:
            opcode, _ = node
            operand, children = get_operand_and_children(node)

            # Perform counter expansion whilst retaining outer counters
            if opcode in {MIN_REPEAT, MAX_REPEAT}:
                n,m = operand
                inner = to_string(flatten_quantifiers(children[0]))

                if m is MAXREPEAT:
                    yield f"{inner}{{{n},}}"
                else:
                    yield f"{inner}{{{n},{m}}}" 
            else:
                yield func(
                    (opcode, operand),
                    (quantifier_fold(func, child) for child in children),
                )

    return func(None, _quantifier_fold(tree))

"""
    Normal fold of tree.
"""
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


def dfs(tree: SubPattern) -> Iterable[tuple[NamedIntConstant, Any]]:
    return fold(
        lambda x, ys: chain([] if x is None else [x], chain.from_iterable(ys)),
        tree,
    )

def in_to_string(xs: list[Any]) -> str:
    result = ["["]
    for opcode, operand in xs:
        if opcode is NEGATE:
            result.append("^")
        elif opcode is LITERAL:
            result.append(escape(chr(operand)))
        elif opcode is RANGE:
            start, end = operand
            result.append(f"{escape(chr(start))}-{escape(chr(end))}")
        elif opcode is CATEGORY:
            result.append(category_to_string(operand))
        else:
            raise NotImplementedError(f"Unknown opcode: {opcode}")
    result.append("]")
    return "".join(result)

def category_to_string(category: NamedIntConstant) -> str:
    try:
        return {
            CATEGORY_WORD: "\\w",
            CATEGORY_NOT_WORD: "\\W",
            CATEGORY_SPACE: "\\s",
            CATEGORY_NOT_SPACE: "\\S",
            CATEGORY_LINEBREAK: "\\b",
            CATEGORY_NOT_LINEBREAK: "\\B",
            CATEGORY_DIGIT: "\\d",
            CATEGORY_NOT_DIGIT: "\\D",
        }[category]
    except KeyError as e:
        raise NotImplementedError(f"Unknown category: {category}") from e

"""
    Quantifier opcodes turned into string.
"""
def repeat_to_string(
    opcode: NamedIntConstant,
    operand: Any,
    ys: Iterable[str],
) -> str:
    repeat_ch = {
        MAX_REPEAT: "",
        MIN_REPEAT: "?",
        POSSESSIVE_REPEAT: "+",
    }[opcode]

    m, n = operand

    ys = list(ys)
    ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"

    if n is MAXREPEAT:
        # Remove {0,} and {1,} and replace with * and + respectively.
        if m == 0:
            return f"{ys_str}*{repeat_ch}"
        elif m == 1:
            return f"{ys_str}+{repeat_ch}"
        else:
            return f"{ys_str}{{{m},}}{repeat_ch}"
        
    
    # Handles case where {0,0} quantifier is used
    if m == 0 and n == 0:
        return f"{repeat_ch}"

    return f"{ys_str}{{{m},{n}}}{repeat_ch}"


def subpattern_to_string(
    opcode: NamedIntConstant, operand: Any, ys: Iterable[str]
) -> str:
    if opcode is SUBPATTERN:
        _, add_flags, del_flags = operand
        if (add_flags, del_flags) == (0, 0):
            return f"({''.join(ys)})"
        else:
            raise NotImplementedError("Flags not supported")
    else:
        lookbehind = operand
        assert_ch = {
            ASSERT: "=",
            ASSERT_NOT: "!",
            ATOMIC_GROUP: ">",
        }[opcode]
        lookbehind_ch = "<" if lookbehind else ""

        return f"(?{lookbehind_ch}{assert_ch}{''.join(ys)})"


def at_to_string(at: NamedIntConstant) -> str:
    try:
        return {
            AT_BEGINNING: "^",
            AT_BEGINNING_STRING: "\\A",
            AT_BOUNDARY: "\\b",
            AT_END: "$",
            AT_END_STRING: "\\Z",
            AT_NON_BOUNDARY: "\\B",
        }[at]
    except KeyError as e:
        raise NotImplementedError(f"Unknown at: {at}") from e


def to_string_f(
    x: Optional[tuple[NamedIntConstant, Any]], ys: Iterable[str]
) -> str:
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
    elif opcode is AT:
        [(_, at)] = operand
        return at_to_string(at)
    elif opcode is BRANCH:
        return f"(?:{'|'.join(ys)})"
    elif opcode in {MIN_REPEAT, MAX_REPEAT, POSSESSIVE_REPEAT}:
       
        return repeat_to_string(opcode, operand, ys)
    elif opcode in {MAX_QUESTION, MIN_QUESTION, POSSESSIVE_QUESTION}:
        
        ys = list(ys)
        ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"
        if opcode is MAX_QUESTION:
            return f"{ys_str}?"
        elif opcode is MIN_QUESTION:
            return f"{ys_str}??"
        elif opcode is POSSESSIVE_QUESTION:
            return f"{ys_str}?+"
        assert False
    elif opcode in {MAX_STAR, MIN_STAR, POSSESSIVE_STAR}:
        
        ys = list(ys)
        ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"
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
    elif opcode is GROUPREF:
        [(_, ref)] = operand
        return f"\\{ref}(?:)"
    elif opcode in {SUBPATTERN, ASSERT, ASSERT_NOT, ATOMIC_GROUP}:
        return subpattern_to_string(opcode, operand, ys)
    else:
        raise NotImplementedError(f"Unknown opcode: {opcode}")


def to_string(tree: SubPattern) -> str:
    return fold(to_string_f, tree)


def normalize(tree: SubPattern) -> SubPattern:
    """
        Normalize a regular expression pattern.
        This modifies semantics of the pattern to make it easier to analyze:
            1. Remove anchors (at).
            2. Remove flags in subpatterns.
            3. Turn atomic and capturing groups into non-capturing groups.
            4. Raise error if the pattern has possessive quantifiers.
            5. Raise error if the pattern has look-around assertions.
            6. Raise error if the pattern has back-references.
            7. Raise error if the pattern has other features that are not supported.
            8. Let upper-bound of repeat quantifiers be 65,535 (2^16 - 1). Note that
            the maximum bound of re2 is 1,000.
    """

    def f(x: Optional[tuple[NamedIntConstant, Any]], ys: Iterable[str]) -> str:

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
        elif opcode is AT:
            # 1.
            return ""
        elif opcode is BRANCH:
            return f"(?:{'|'.join(ys)})"
        elif opcode in {ATOMIC_GROUP, SUBPATTERN}:
            # 2., 3.
            if opcode is ATOMIC_GROUP:
                opcode = SUBPATTERN
            operand = (0, 0, 0)
            return subpattern_to_string(opcode, operand, ys)
        elif opcode in {MIN_REPEAT, MAX_REPEAT}:
            m, n = operand
            if n is not MAXREPEAT:
                n = min(n, 65535)
            m = min(m, 65535)
            return repeat_to_string(opcode, (m, n), ys)
        elif opcode in {MAX_QUESTION, MIN_QUESTION, POSSESSIVE_QUESTION}:
            ys = list(ys)
            ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"
            if opcode is MAX_QUESTION:
                return f"{ys_str}?"
            elif opcode is MIN_QUESTION:
                return f"{ys_str}??"
            elif opcode is POSSESSIVE_QUESTION:
                return f"{ys_str}?+"
            assert False
        elif opcode in {MAX_STAR, MIN_STAR, POSSESSIVE_STAR}:
            ys = list(ys)
            ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"
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
            # 4., 5., 6., 7.
            raise NotImplementedError(f"Unknown opcode: {opcode}")

    pattern = fold(f, tree)
    return parse(pattern)

"""
    Modified normalizer to allow for counter expansion.
"""
def flatten(x: Optional[tuple[NamedIntConstant, Any]], ys: Iterable[str]) -> str:

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
            return f"(?:{''.join(ys)})"

        elif opcode in {MIN_REPEAT, MAX_REPEAT}:
            n, m = operand

            if m is not MAXREPEAT:
                m = min(m, 65535)
            n = min(n, 65535)

            ys = list(ys)
            ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"

            # Expansion of counters
            if m is MAXREPEAT:

                if n == 0:
                    return f"({ys_str}*)"
                elif n == 1:
                    return f"({ys_str}+)"
                else:
                    return f"({ys_str*n}+)"
            elif n == m:
                return f"({ys_str*n})"
            else:
                suffix = ys_str + "?"
                return f"({ys_str*n}{suffix*(m-n)})"

        elif opcode in {MAX_QUESTION, MIN_QUESTION, POSSESSIVE_QUESTION}:
            ys = list(ys)
            ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"
            if opcode is MAX_QUESTION:
                return f"{ys_str}?"
            elif opcode is MIN_QUESTION:
                return f"{ys_str}??"
            elif opcode is POSSESSIVE_QUESTION:
                return f"{ys_str}?+"
            assert False

        elif opcode in {MAX_STAR, MIN_STAR, POSSESSIVE_STAR}:
            ys = list(ys)
            ys_str = ys[0] if len(ys) == 1 else f"(?:{''.join(ys)})"
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