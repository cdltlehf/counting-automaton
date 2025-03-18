from enum import Enum


class StrEnum(str, Enum):
    pass


class MatchingInfo(StrEnum):
    EVAL_SYMBOL = "EVAL_SYMBOL"
    EVAL_GUARD = "EVAL_GUARD"
    APPLY_ACTION = "APPLY_ACTION"


VERBOSE = 5
