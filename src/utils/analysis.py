"""This module contains the type definitions for the analysis module."""

from typing import Optional, TypedDict


class TestCaseResult(TypedDict):
    computation_info: dict[str, int]
    is_final: bool


class TestCaseDict(TypedDict):
    text: str
    result: Optional[TestCaseResult]


class OutputDict(TypedDict):
    pattern: str
    results: Optional[list[TestCaseDict]]
