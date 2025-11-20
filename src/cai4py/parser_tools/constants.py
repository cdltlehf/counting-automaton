"""Constants used by the re module."""

try:
    from re._constants import _NamedIntConstant as NamedIntConstant  # type: ignore
    from re._constants import (
        ANY,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        ASSERT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        ASSERT_NOT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        AT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        AT_BEGINNING,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        AT_BEGINNING_STRING,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        AT_BOUNDARY,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        AT_END,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        AT_END_STRING,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        AT_NON_BOUNDARY,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        ATOMIC_GROUP,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        BRANCH,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY_DIGIT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY_LINEBREAK,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY_NOT_DIGIT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY_NOT_LINEBREAK,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY_NOT_SPACE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY_NOT_WORD,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY_SPACE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        CATEGORY_WORD,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import error
    from re._constants import (
        FAILURE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        GROUPREF,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        GROUPREF_EXISTS,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        IN,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        LITERAL,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        MAX_REPEAT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        MAXGROUPS,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        MAXREPEAT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        MIN_REPEAT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        NEGATE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        NOT_LITERAL,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue],
    )
    from re._constants import OPCODES
    from re._constants import (
        POSSESSIVE_REPEAT,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        RANGE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SRE_FLAG_ASCII,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SRE_FLAG_DEBUG,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SRE_FLAG_DOTALL,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SRE_FLAG_IGNORECASE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SRE_FLAG_LOCALE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SRE_FLAG_MULTILINE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SRE_FLAG_UNICODE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SRE_FLAG_VERBOSE,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SUBPATTERN,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
    from re._constants import (
        SUCCESS,  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    )
except ImportError:
    from sre_constants import _NamedIntConstant as NamedIntConstant
    from sre_constants import ANY
    from sre_constants import ASSERT
    from sre_constants import ASSERT_NOT
    from sre_constants import AT
    from sre_constants import AT_BEGINNING
    from sre_constants import AT_BEGINNING_STRING
    from sre_constants import AT_BOUNDARY
    from sre_constants import AT_END
    from sre_constants import AT_END_STRING
    from sre_constants import AT_NON_BOUNDARY
    from sre_constants import BRANCH
    from sre_constants import CATEGORY
    from sre_constants import CATEGORY_DIGIT
    from sre_constants import CATEGORY_LINEBREAK
    from sre_constants import CATEGORY_NOT_DIGIT
    from sre_constants import CATEGORY_NOT_LINEBREAK
    from sre_constants import CATEGORY_NOT_SPACE
    from sre_constants import CATEGORY_NOT_WORD
    from sre_constants import CATEGORY_SPACE
    from sre_constants import CATEGORY_WORD
    from sre_constants import error
    from sre_constants import FAILURE
    from sre_constants import GROUPREF
    from sre_constants import GROUPREF_EXISTS
    from sre_constants import IN
    from sre_constants import LITERAL
    from sre_constants import MAX_REPEAT
    from sre_constants import MAXGROUPS
    from sre_constants import MAXREPEAT
    from sre_constants import MIN_REPEAT
    from sre_constants import NEGATE
    from sre_constants import NOT_LITERAL
    from sre_constants import OPCODES
    from sre_constants import RANGE
    from sre_constants import SRE_FLAG_ASCII
    from sre_constants import SRE_FLAG_DEBUG
    from sre_constants import SRE_FLAG_DOTALL
    from sre_constants import SRE_FLAG_IGNORECASE
    from sre_constants import SRE_FLAG_LOCALE
    from sre_constants import SRE_FLAG_MULTILINE
    from sre_constants import SRE_FLAG_UNICODE
    from sre_constants import SRE_FLAG_VERBOSE
    from sre_constants import SUBPATTERN
    from sre_constants import SUCCESS

    ATOMIC_GROUP = NamedIntConstant(len(OPCODES) + 2, "ATOMIC_GROUP")
    POSSESSIVE_REPEAT = NamedIntConstant(len(OPCODES) + 3, "POSSESSIVE_REPEAT")

MAX_QUESTION = NamedIntConstant(len(OPCODES) + 4, "MAX_QUESTION")
MIN_QUESTION = NamedIntConstant(len(OPCODES) + 5, "MIN_QUESTION")
MAX_STAR = NamedIntConstant(len(OPCODES) + 6, "MAX_STAR")
MIN_STAR = NamedIntConstant(len(OPCODES) + 7, "MIN_STAR")
MAX_PLUS = NamedIntConstant(len(OPCODES) + 8, "MAX_PLUS")
MIN_PLUS = NamedIntConstant(len(OPCODES) + 9, "MIN_PLUS")

POSSESSIVE_QUESTION = NamedIntConstant(len(OPCODES) + 10, "POSSESSIVE_QUESTION")
POSSESSIVE_STAR = NamedIntConstant(len(OPCODES) + 11, "POSSESSIVE_STAR")
POSSESSIVE_PLUS = NamedIntConstant(len(OPCODES) + 12, "POSSESSIVE_PLUS")

__all__ = [
    "ANY",
    "ASSERT",
    "ASSERT_NOT",
    "AT",
    "ATOMIC_GROUP",
    "AT_BEGINNING",
    "AT_BEGINNING_STRING",
    "AT_BOUNDARY",
    "AT_END",
    "AT_END_STRING",
    "AT_NON_BOUNDARY",
    "BRANCH",
    "CATEGORY",
    "CATEGORY_DIGIT",
    "CATEGORY_LINEBREAK",
    "CATEGORY_NOT_DIGIT",
    "CATEGORY_NOT_LINEBREAK",
    "CATEGORY_NOT_SPACE",
    "CATEGORY_NOT_WORD",
    "CATEGORY_SPACE",
    "CATEGORY_WORD",
    "FAILURE",
    "GROUPREF",
    "GROUPREF_EXISTS",
    "IN",
    "LITERAL",
    "MAXGROUPS",
    "MAXREPEAT",
    "MAXREPEAT",
    "MAX_REPEAT",
    "MIN_REPEAT",
    "NEGATE",
    "NOT_LITERAL",
    "NamedIntConstant",
    "POSSESSIVE_REPEAT",
    "RANGE",
    "SRE_FLAG_ASCII",
    "SRE_FLAG_DEBUG",
    "SRE_FLAG_DOTALL",
    "SRE_FLAG_IGNORECASE",
    "SRE_FLAG_LOCALE",
    "SRE_FLAG_MULTILINE",
    "SRE_FLAG_UNICODE",
    "SRE_FLAG_VERBOSE",
    "SUBPATTERN",
    "SUCCESS",
    "MAX_STAR",
    "MIN_STAR",
    "MAX_QUESTION",
    "MIN_QUESTION",
    "MAX_PLUS",
    "MIN_PLUS",
    "POSSESSIVE_STAR",
    "POSSESSIVE_QUESTION",
    "POSSESSIVE_PLUS",
    "error",
]


# Monkey-patch _NamedIntConstant to make it pickleable
def _get_global_named_int_constant(name: str):
    """Get a _NamedIntConstant by name from globals."""
    return globals().get(name)


def _named_int_constant_reduce(self):
    """Custom pickle support for _NamedIntConstant."""
    return (_get_global_named_int_constant, (str(self),))


NamedIntConstant.__reduce__ = _named_int_constant_reduce  # type: ignore
