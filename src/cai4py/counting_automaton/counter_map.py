"""Counter vector."""

from collections import defaultdict as dd
from collections.abc import Hashable
from copy import copy
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, TypeVar

from cai4py.counting_automaton.position_counting_automaton import (
    CounterVariable,
)
from cai4py.custom_counters.counter_base import CounterBase
from cai4py.custom_counters.counter_type import CounterType

from ..utils.util_logging import setup_debugger
from .computation_logging import VERBOSE, ComputationStep

logger = setup_debugger(__name__)


class StrEnum(str, Enum):
    pass


T = TypeVar("T", bound=Hashable)


class CounterMap(dict[T, int], Hashable):

    def __init__(self, variables: Iterable[T]) -> None:
        """Initialize a counter map."""
        self.variables = variables
        self._index: dict[T, int] = {c: i for i, c in enumerate(variables)}

    @property
    def index(self) -> dict[T, int]:
        return self._index

    def to_list(self) -> list[Optional[int]]:
        return [self.get(c, None) for c in self.variables]

    def to_tuple(self) -> tuple[Optional[int], ...]:
        return tuple(self.to_list())

    def __setitem__(self, key: T, value: int) -> None:
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


class Guard(dd[T, list[CounterPredicate]], Hashable):
    """Guard"""

    def __init__(
        self,
        guard: Optional[Mapping[T, list[CounterPredicate]]] = None,
    ) -> None:
        super().__init__(list)
        if guard is not None:
            self.update(guard)

    # All the counter guards that can be applied.

    @classmethod
    def less_than(cls, counter_variable: T, value: int) -> "Guard[T]":
        return cls({counter_variable: [CounterPredicate.less_than(value)]})

    @classmethod
    def not_less_than(cls, counter_variable: T, value: int) -> "Guard[T]":
        return cls({counter_variable: [CounterPredicate.not_less_than(value)]})

    @classmethod
    def not_greater_than(cls, counter_variable: T, value: int) -> "Guard[T]":
        return cls(
            {counter_variable: [CounterPredicate.not_greater_than(value)]}
        )

    def __hash__(self) -> int:  # type: ignore
        return hash(tuple((key, tuple(value)) for key, value in self.items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Guard):
            return False
        return hash(self) == hash(other)

    def __call__(self, counter: CounterBase | None) -> bool:
        if counter is None:
            return False

        for _, predicates in self.items():

            logger.log(VERBOSE, ComputationStep.EVAL_PREDICATE.value)

            for predicate in predicates:
                logger.debug("\t\t\tPredicate: %s", predicate)
                if not predicate(counter):
                    return False
        return True

    def __copy__(self) -> "Guard[T]":
        return Guard(self)

    def __iadd__(self, other: "Guard[T]") -> "Guard[T]":
        for variable in other:
            self[variable] += other[variable]
        return self

    def __add__(self, other: "Guard[T]") -> "Guard[T]":
        new = copy(self)
        new += other
        return new

    def __str__(self) -> str:
        return ", ".join(
            ", ".join(f"c[{counter}]{predicate}" for predicate in predicates)
            for counter, predicates in self.items()
        )

    def __reduce__(self):
        # Custom pickle support - return constructor and args
        return (
            self.__class__,
            (dict(self),),  # Pass the dict as the 'guard' parameter
            None,
            None,
            None,
        )


#    Operations that are applied to the counters.


class CounterOperationComponent:
    """Counter Operation Component"""

    class Type(StrEnum):
        NO_OPERATION = ""
        ACTIVATE_OR_RESET = " = 1"
        INCREASE = "++"
        INACTIVATE = " = None"

    def __init__(
        self, operation_type: Type, counter_range: Optional[tuple[int, int]]
    ) -> None:
        self.type = operation_type

        if counter_range is not None:
            lo, hi = counter_range
            self.lower_bound = lo
            self.upper_bound = hi
        else:
            self.lower_bound = 0
            self.upper_bound = 0

    @classmethod
    def no_operation(cls):
        return cls(CounterOperationComponent.Type.NO_OPERATION, None)

    @classmethod
    def activate_or_reset(cls, lower_bound: int, upper_bound: int):
        return cls(
            CounterOperationComponent.Type.ACTIVATE_OR_RESET,
            (lower_bound, upper_bound),
        )

    @classmethod
    def increase(cls):
        return cls(CounterOperationComponent.Type.INCREASE, None)

    @classmethod
    def inactivate(cls):
        return cls(CounterOperationComponent.Type.INACTIVATE, None)

    def __call__(
        self, counter: Optional[CounterBase], counter_type: CounterType
    ) -> Optional[CounterBase]:
        logger.log(VERBOSE, ComputationStep.APPLY_OPERATION.value)

        if self.type is CounterOperationComponent.Type.NO_OPERATION:
            return counter

        elif self.type is CounterOperationComponent.Type.ACTIVATE_OR_RESET:
            counter = counter_type.create_counter(
                self.lower_bound, self.upper_bound
            )
            return counter

        elif self.type is CounterOperationComponent.Type.INCREASE:
            assert counter is not None
            if counter is not None:
                counter.inc()
            return counter

        elif self.type is CounterOperationComponent.Type.INACTIVATE:
            return None

    def __mul__(self, other: object) -> "CounterOperationComponent":

        if not isinstance(other, CounterOperationComponent):
            return NotImplemented

        if other.type is CounterOperationComponent.Type.NO_OPERATION:
            return self

        elif other.type is CounterOperationComponent.Type.ACTIVATE_OR_RESET:
            return other

        elif other.type is CounterOperationComponent.Type.INCREASE:
            if self.type is CounterOperationComponent.Type.NO_OPERATION:
                return other

            return NotImplemented

        elif other.type is CounterOperationComponent.Type.INACTIVATE:
            return other

        assert False, other


def _default_action_factory():
    """Factory function for Action's default_factory (needed for pickling)"""
    return CounterOperationComponent.no_operation()


class Action(dd[T, CounterOperationComponent], Hashable):
    """Action"""

    def __init__(
        self,
        action: Optional[Mapping[T, CounterOperationComponent]] = None,
    ) -> None:
        super().__init__(_default_action_factory)
        if action is not None:
            self.update(action)

    @classmethod
    def increase(cls, counter_variable: T) -> "Action[T]":
        return cls({counter_variable: CounterOperationComponent.increase()})

    @classmethod
    def activate(
        cls, counter_variable: T, lower_bound: int, upper_bound: Optional[int]
    ) -> "Action[T]":
        if upper_bound is None:
            upper_bound = -1

        return cls(
            {
                counter_variable: CounterOperationComponent.activate_or_reset(
                    lower_bound, upper_bound
                )
            }
        )

    @classmethod
    def inactivate(cls, counter_variable: T) -> "Action[T]":
        return cls({counter_variable: CounterOperationComponent.inactivate()})

    def move_and_apply(
        self, counter: CounterBase | None, counter_type: CounterType
    ) -> CounterBase | None:
        if counter is None:
            return None
        # Loop over keys of default dict

        looped = False
        new_counter = None
        for variable in self:
            looped = True

            logger.debug(
                "\t\t\tCounter operation: %s(%s,%s)",
                self[variable].type,
                self[variable].lower_bound,
                self[variable].upper_bound,
            )
            new_counter = self[variable](counter, counter_type)

        # Need to check if the loop ever activated.
        if not looped:
            return counter
        else:
            assert new_counter is not None
            return new_counter

    def __hash__(self) -> int:  # type: ignore
        return hash(
            tuple(
                (
                    key,
                    (
                        value.type,
                        getattr(value, "lo", None),
                        getattr(value, "hi", None),
                    ),
                )
                for key, value in self.items()
            )
        )

    def __call__(
        self,
        counter: CounterBase,
        counter_type: CounterType = CounterType.BIT_VECTOR,
    ) -> CounterBase | None:
        return self.move_and_apply(copy(counter), counter_type)

    def __copy__(self) -> "Action[T]":
        return Action(self)

    def __iadd__(self, other: "Action[T]") -> "Action[T]":
        for variable in other:
            self[variable] *= other[variable]
        return self

    def __add__(self, other: "Action[T]") -> "Action[T]":
        new = copy(self)
        new += other
        return new

    def __str__(self) -> str:
        return ", ".join(
            f"c[{counter}]{operation.type}({operation.lower_bound},{operation.upper_bound})"
            for counter, operation in self.items()
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return hash(self) == hash(other)

    def __reduce__(self):
        # Custom pickle support - return constructor and args
        return (
            self.__class__,
            (dict(self),),  # Pass the dict as the 'action' parameter
            None,
            None,
            None,
        )
