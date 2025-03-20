"""This module contains the type definitions for the analysis module."""

from typing import Optional, TypedDict


class ComputationInfo(TypedDict):
    EVAL_SYMBOL: int
    EVAL_PREDICATE: int
    APPLY_OPERATION: int
    ACCESS_NODE_MERGE: int
    ACCESS_NODE_CLONE: int


class TestCaseResult(TypedDict):
    computation_info: ComputationInfo
    is_final: bool


class TestCaseDict(TypedDict):
    text: str
    method_to_result: dict[str, Optional[TestCaseResult]]


class OutputDict(TypedDict):
    pattern: str
    results: Optional[list[TestCaseDict]]
