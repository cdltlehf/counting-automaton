"""Counter vector."""

from collections import defaultdict
from copy import copy
from enum import Enum
from typing import Any, Generic, Hashable, Iterable, Mapping, Optional, TypeVar


class StrEnum(str, Enum):
    pass


T = TypeVar("T", bound=Hashable)


class CounterVector(Generic[T], dict[T, int], Hashable):
    """Counter vector."""

    def __init__(self, variables: Iterable[T]) -> None:
        """Initialize a counter vector."""
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

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CounterVector):
            return NotImplemented
        return hash(self) == hash(other)


class CounterPredicateType(StrEnum):
    NOT_LESS_THAN = " >= "
    NOT_GREATER_THAN = " <= "
    LESS_THAN = " < "


class CounterPredicate(Hashable):
    """Counter Predicate"""

    def __init__(
        self, predicate_type: CounterPredicateType, value: int
    ) -> None:
        self.predicate_type = predicate_type
        self.value = value

    @classmethod
    def less_than(cls, value: int) -> "CounterPredicate":
        return cls(CounterPredicateType.LESS_THAN, value)

    @classmethod
    def not_less_than(cls, value: int) -> "CounterPredicate":
        return cls(CounterPredicateType.NOT_LESS_THAN, value)

    @classmethod
    def not_greater_than(cls, value: int) -> "CounterPredicate":
        return cls(CounterPredicateType.NOT_GREATER_THAN, value)

    def __hash__(self) -> int:
        return hash((self.predicate_type, self.value))

    def __call__(self, counter_value: int) -> bool:
        if self.predicate_type is CounterPredicateType.NOT_LESS_THAN:
            return counter_value >= self.value
        elif self.predicate_type is CounterPredicateType.NOT_GREATER_THAN:
            return counter_value <= self.value
        elif self.predicate_type is CounterPredicateType.LESS_THAN:
            return counter_value < self.value

    def __str__(self) -> str:
        return f"{self.predicate_type}{self.value}"


class Guard(defaultdict[T, list[CounterPredicate]], Hashable):
    """Guard"""

    def __init__(
        self,
        guard: Optional[Mapping[T, list[CounterPredicate]]] = None,
    ) -> None:
        super().__init__(list)
        if guard is not None:
            self.update(guard)

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

    def __hash__(self) -> int:
        return hash(tuple((key, tuple(value)) for key, value in self.items()))

    def __call__(self, counter_vector: CounterVector[T]) -> bool:
        for counter_variable, predicates in self.items():
            for predicate in predicates:
                if not predicate(counter_vector[counter_variable]):
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


class CounterOperationComponent(StrEnum):
    """Counter Operation Component"""

    NO_OPERATION = ""
    ACTIVATE_OR_RESET = " = 1"
    INCREASE = "++"
    INACTIVATE = " = None"

    def __call__(self, counter_value: Optional[int]) -> Optional[int]:
        if self is CounterOperationComponent.NO_OPERATION:
            return counter_value
        elif self is CounterOperationComponent.ACTIVATE_OR_RESET:
            return 1
        elif self is CounterOperationComponent.INCREASE:
            assert counter_value is not None
            return counter_value + 1
        elif self is CounterOperationComponent.INACTIVATE:
            return None

    def __mul__(self, other: object) -> "CounterOperationComponent":
        if not isinstance(other, CounterOperationComponent):
            return NotImplemented
        if other is CounterOperationComponent.NO_OPERATION:
            return self
        elif other is CounterOperationComponent.ACTIVATE_OR_RESET:
            return other
        elif other is CounterOperationComponent.INCREASE:
            if self is CounterOperationComponent.NO_OPERATION:
                return other
            return NotImplemented
        elif other is CounterOperationComponent.INACTIVATE:
            return other
        assert False, other


class Action(Generic[T], defaultdict[T, CounterOperationComponent], Hashable):
    """Action"""

    def __init__(
        self,
        action: Optional[Mapping[T, CounterOperationComponent]] = None,
    ) -> None:
        super().__init__(lambda: CounterOperationComponent.NO_OPERATION)
        if action is not None:
            self.update(action)

    @classmethod
    def increase(cls, counter_variable: T) -> "Action[T]":
        return cls({counter_variable: CounterOperationComponent.INCREASE})

    @classmethod
    def activate(cls, counter_variable: T) -> "Action[T]":
        return cls(
            {counter_variable: CounterOperationComponent.ACTIVATE_OR_RESET}
        )

    @classmethod
    def inactivate(cls, counter_variable: T) -> "Action[T]":
        return cls({counter_variable: CounterOperationComponent.INACTIVATE})

    def move_and_apply(
        self, counter_vector: CounterVector[T]
    ) -> CounterVector[T]:
        for variable in self:
            value = counter_vector.get(variable, None)
            # value = reduce(lambda x, y: y(x), self[variable], value)
            value = self[variable](value)
            if value is not None:
                counter_vector[variable] = value
            else:
                del counter_vector[variable]
        return counter_vector

    def __hash__(self) -> int:
        return hash(tuple((key, tuple(value)) for key, value in self.items()))

    def __call__(self, counter_vector: CounterVector[T]) -> CounterVector[T]:
        return self.move_and_apply(copy(counter_vector))

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
            f"c[{counter}]{operation}" for counter, operation in self.items()
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return hash(self) == hash(other)
