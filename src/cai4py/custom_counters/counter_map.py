"""Counter vector."""

from collections.abc import Hashable
from enum import Enum
from typing import Any, Iterable, Optional

from cai4py.custom_counters.counter_base import CounterBase

from ..utils.util_logging import setup_debugger
from .counter_base import CounterVariable

logger = setup_debugger(__name__)


class StrEnum(str, Enum):
    pass


class CounterMap(dict[CounterVariable, int], Hashable):

    def __init__(self, variables: Iterable[CounterVariable]) -> None:
        """Initialize a counter map."""
        self.variables = variables
        self._index: dict[CounterVariable, int] = {
            c: i for i, c in enumerate(variables)
        }

    @property
    def index(self) -> dict[CounterVariable, int]:
        return self._index

    def to_list(self) -> list[Optional[int]]:
        return [self.get(c, None) for c in self.variables]

    def to_tuple(self) -> tuple[Optional[int], ...]:
        return tuple(self.to_list())

    def __setitem__(self, key: CounterVariable, value: int) -> None:
        if key not in self.index:
            raise ValueError(f"Invalid counter variable: {key}")
        super().__setitem__(key, value)

    def __hash__(self) -> int:  # type: ignore
        return hash(self.to_tuple())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CounterMap):
            return NotImplemented
        return hash(self) == hash(other)


class CounterPredicate(Hashable):
    """Counter Predicate"""

    class Type(StrEnum):
        NOT_LESS_THAN = " >= "
        NOT_GREATER_THAN = " <= "
        LESS_THAN = " < "

    def __init__(self, predicate_type: Type, value: int) -> None:
        self.type = predicate_type
        self.value = value

    @classmethod
    def less_than(cls, value: int) -> "CounterPredicate":
        return cls(CounterPredicate.Type.LESS_THAN, value)

    @classmethod
    def not_less_than(cls, value: int) -> "CounterPredicate":
        return cls(CounterPredicate.Type.NOT_LESS_THAN, value)

    @classmethod
    def not_greater_than(cls, value: int) -> "CounterPredicate":
        return cls(CounterPredicate.Type.NOT_GREATER_THAN, value)

    def __hash__(self) -> int:
        return hash((self.type, self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CounterPredicate):
            return False
        return self.type == other.type and self.value == other.value

    def __call__(self, counter: CounterBase | None) -> bool:
        if counter is None:
            return False
        if self.type is CounterPredicate.Type.NOT_LESS_THAN:
            return counter.ge_lower_bound()

        elif self.type is CounterPredicate.Type.NOT_GREATER_THAN:
            return counter.le_upper_bound()

        elif self.type is CounterPredicate.Type.LESS_THAN:
            return counter.le_upper_bound()
        else:
            raise ValueError(f"Unhandled predicate type: {self.type}")

    def __str__(self) -> str:
        return f"{self.type}{self.value}"
