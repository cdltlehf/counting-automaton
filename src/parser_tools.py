"""Parser tools for regular expressions."""

# mypy: disable-error-code=import-untyped
# pylint: disable=useless-import-alias

from itertools import chain
import re
from re._constants import _NamedIntConstant as Constant
from re._constants import ANY as ANY
from re._constants import ASSERT as ASSERT
from re._constants import ASSERT_NOT as ASSERT_NOT
from re._constants import AT as AT
from re._constants import AT_BEGINNING as AT_BEGINNING
from re._constants import AT_BEGINNING_STRING as AT_BEGINNING_STRING
from re._constants import AT_BOUNDARY as AT_BOUNDARY
from re._constants import AT_END as AT_END
from re._constants import AT_END_STRING as AT_END_STRING
from re._constants import AT_NON_BOUNDARY as AT_NON_BOUNDARY
from re._constants import ATOMIC_GROUP as ATOMIC_GROUP
from re._constants import BRANCH as BRANCH
from re._constants import CATEGORY as CATEGORY
from re._constants import CATEGORY_DIGIT as CATEGORY_DIGIT
from re._constants import CATEGORY_LINEBREAK as CATEGORY_LINEBREAK
from re._constants import CATEGORY_NOT_DIGIT as CATEGORY_NOT_DIGIT
from re._constants import CATEGORY_NOT_LINEBREAK as CATEGORY_NOT_LINEBREAK
from re._constants import CATEGORY_NOT_SPACE as CATEGORY_NOT_SPACE
from re._constants import CATEGORY_NOT_WORD as CATEGORY_NOT_WORD
from re._constants import CATEGORY_SPACE as CATEGORY_SPACE
from re._constants import CATEGORY_WORD as CATEGORY_WORD
from re._constants import FAILURE as FAILURE
from re._constants import GROUPREF as GROUPREF
from re._constants import GROUPREF_EXISTS as GROUPREF_EXISTS
from re._constants import IN as IN
from re._constants import LITERAL as LITERAL
from re._constants import MAX_REPEAT as MAX_REPEAT
from re._constants import MAXREPEAT as MAXREPEAT
from re._constants import MIN_REPEAT as MIN_REPEAT
from re._constants import NEGATE as NEGATE
from re._constants import NOT_LITERAL as NOT_LITERAL
from re._constants import POSSESSIVE_REPEAT as POSSESSIVE_REPEAT
from re._constants import RANGE as RANGE
from re._constants import SUBPATTERN as SUBPATTERN
from re._parser import State
from re._parser import SubPattern as SubPattern
import re._parser as parser
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, TypeVar
import warnings

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

warnings.simplefilter(action="ignore", category=FutureWarning)


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

constants = SimpleNamespace(
    ANY = ANY,
    ASSERT = ASSERT,
    ASSERT_NOT = ASSERT_NOT,
    AT = AT,
    AT_BEGINNING = AT_BEGINNING,
    AT_BEGINNING_STRING = AT_BEGINNING_STRING,
    AT_BOUNDARY = AT_BOUNDARY,
    AT_END = AT_END,
    AT_END_STRING = AT_END_STRING,
    AT_NON_BOUNDARY = AT_NON_BOUNDARY,
    ATOMIC_GROUP = ATOMIC_GROUP,
    BRANCH = BRANCH,
    CATEGORY = CATEGORY,
    CATEGORY_DIGIT = CATEGORY_DIGIT,
    CATEGORY_LINEBREAK = CATEGORY_LINEBREAK,
    CATEGORY_NOT_DIGIT = CATEGORY_NOT_DIGIT,
    CATEGORY_NOT_LINEBREAK = CATEGORY_NOT_LINEBREAK,
    CATEGORY_NOT_SPACE = CATEGORY_NOT_SPACE,
    CATEGORY_NOT_WORD = CATEGORY_NOT_WORD,
    CATEGORY_SPACE = CATEGORY_SPACE,
    CATEGORY_WORD = CATEGORY_WORD,
    FAILURE = FAILURE,
    GROUPREF = GROUPREF,
    GROUPREF_EXISTS = GROUPREF_EXISTS,
    IN = IN,
    LITERAL = LITERAL,
    MAX_REPEAT = MAX_REPEAT,
    MAXREPEAT = MAXREPEAT,
    MIN_REPEAT = MIN_REPEAT,
    NEGATE = NEGATE,
    NOT_LITERAL = NOT_LITERAL,
    POSSESSIVE_REPEAT = POSSESSIVE_REPEAT,
    RANGE = RANGE,
    SUBPATTERN = SUBPATTERN,
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


def fold(
    f: Callable[[Optional[tuple[Constant, Any]], Iterable[T]], T],
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


def in_to_string(xs: list[Any]) -> str:
    result = ["["]
    for opcode, operand in xs:
        if opcode is NEGATE:
            result.append("^")
        elif opcode is LITERAL:
            result.append(re.escape(chr(operand)))
        elif opcode is RANGE:
            start, end = operand
            result.append(f"{re.escape(chr(start))}-{re.escape(chr(end))}")
        elif opcode is CATEGORY:
            result.append(category_to_string(operand))
        else:
            raise NotImplementedError(f"Unknown opcode: {opcode}")
    result.append("]")
    return "".join(result)


def category_to_string(category: Constant) -> str:
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


def repeat_to_string(
    opcode: Constant, operand: Any, ys: Iterable[str]
) -> str:
    repeat_ch = {
        MIN_REPEAT: "?",
        MAX_REPEAT: "",
        POSSESSIVE_REPEAT: "+",
    }[opcode]

    m, n = operand
    if n is MAXREPEAT:
        return f"(?:{''.join(ys)}){{{m},}}{repeat_ch}"
    return f"(?:{''.join(ys)}){{{m},{n}}}{repeat_ch}"


def subpattern_to_string(
    opcode: Constant, operand: Any, ys: Iterable[str]
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


def at_to_string(at: Constant) -> str:
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


def to_string(tree: SubPattern) -> str:
    def f(x: Optional[tuple[Constant, Any]], ys: Iterable[str]) -> str:
        if x is None:
            return "".join(ys)
        opcode, operand = x
        if opcode is LITERAL:
            return f"{re.escape(operand)}"
        elif opcode is ANY:
            return "."
        elif opcode is NOT_LITERAL:
            _, [(_, c)] = x
            return f"[^{re.escape(chr(c))}]"
        elif opcode is IN:
            _, [(_, zs)] = x
            return in_to_string(zs)
        elif opcode is AT:
            [(_, at)] = operand
            return at_to_string(at)
        elif opcode is BRANCH:
            return f"(?:{'|'.join(ys)})"
        elif opcode in REPEAT_OPCODES:
            return repeat_to_string(opcode, operand, ys)
        elif opcode is GROUPREF:
            [(_, ref)] = operand
            return f"\\{ref}(?:)"
        elif opcode in SUBPATTERN_OPCODES:
            return subpattern_to_string(opcode, operand, ys)
        else:
            raise NotImplementedError(f"Unknown opcode: {opcode}")

    return fold(f, tree)


def normalize(tree: SubPattern) -> SubPattern:
    """Normalize a regular expression pattern.
    1. Remove anchors (at).
    2. Remove flags in subpatterns.
    3. Turn atomic and capturing groups into non-capturing groups.
    4. Turn possessive quantifiers into greedy quantifiers.
    5. Remove look-ahead and look-behind assertions.
    6. Raise error if the pattern has back-references.
    7. Raise error if the pattern has other features that are not supported.
    """

    def f(x: Optional[tuple[Constant, Any]], ys: Iterable[str]) -> str:
        if x is None:
            return "".join(ys)
        opcode, operand = x
        if opcode is LITERAL:
            return f"{re.escape(operand)}"
        elif opcode is ANY:
            return "."
        elif opcode is NOT_LITERAL:
            _, [(_, c)] = x
            return f"[^{re.escape(chr(c))}]"
        elif opcode is IN:
            _, [(_, zs)] = x
            return in_to_string(zs)
        elif opcode is AT:
            # 1.
            return "(?:)"
        elif opcode is BRANCH:
            return f"(?:{'|'.join(ys)})"
        elif opcode in {ATOMIC_GROUP, SUBPATTERN}:
            # 2., 3.
            if opcode is ATOMIC_GROUP:
                opcode = SUBPATTERN
            operand = (0, 0, 0)
            return subpattern_to_string(opcode, operand, ys)
        elif opcode in REPEAT_OPCODES:
            # 4.
            if opcode is POSSESSIVE_REPEAT:
                opcode = MAX_REPEAT
            return repeat_to_string(opcode, operand, ys)
        elif opcode in {ASSERT, ASSERT_NOT}:
            # 5.
            return "(?:)"
        else:
            # 6., 7.
            raise NotImplementedError(f"Unknown opcode: {opcode}")

    return parse(fold(f, tree))


def is_nullable(tree: SubPattern) -> bool:
    if has_extended_features(tree):
        raise ValueError("Pattern has extended features")

    def f(
        x: Optional[tuple[Constant, Any]], ys: Iterable[bool]
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
        x: Optional[tuple[Constant, Any]], ys: Iterable[bool]
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
