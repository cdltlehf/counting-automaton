from enum import Enum


class StrEnum(str, Enum):
    pass


class ComputationStep(StrEnum):
    EVAL_SYMBOL = "EVAL_SYMBOL"
    EVAL_PREDICATE = "EVAL_PREDICATE"
    APPLY_OPERATION = "APPLY_OPERATION"
    ACCESS_NODE_MERGE = "ACCESS_NODE_MERGE"
    ACCESS_NODE_CLONE = "ACCESS_NODE_CLONE"


VERBOSE = 5
