"""Counter vector."""

from copy import copy
from enum import StrEnum
from typing import Any, Hashable, Iterator, Mapping, Optional

CounterVariable = int
CounterValue = int


class CounterVector(Mapping[CounterVariable, CounterValue], Hashable):
    """Counter vector."""

    def __init__(self, counters: dict[CounterVariable, int]) -> None:
        """Initialize a counter vector."""
        self._values: tuple[Optional[CounterValue], ...]
        self._values = tuple([None] * len(counters))
        self._upper_bound = counters

    def upper_bound(self, counter_variable: CounterVariable) -> int:
        return self._upper_bound[counter_variable]

    def to_list(self) -> list[Optional[CounterValue]]:
        return list(self._values)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CounterVector):
            return NotImplemented
        return self._values == other._values

    def __str__(self) -> str:
        return str(self._values)

    def __iter__(self) -> Iterator[CounterVariable]:
        for i in range(len(self._values)):
            if self._values[i] is not None:
                yield i

    def __setitem__(self, key: CounterVariable, value: CounterValue) -> None:
        self._values = tuple(
            value if i == key else self._values[i]
            for i in range(len(self._values))
        )

    def __getitem__(self, key: CounterVariable) -> CounterValue:
        value = self._values[key]
        if value is None:
            raise KeyError(key)
        return value

    def __delitem__(self, key: CounterVariable) -> None:
        self._values = tuple(
            None if i == key else self._values[i]
            for i in range(len(self._values))
        )

    def __len__(self) -> int:
        return len(self._values)

    def __copy__(self) -> "CounterVector":
        new = CounterVector(self._upper_bound)
        new._values = self._values
        return new

    def __hash__(self) -> int:
        return hash(self._values)


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

    def __call__(self, counter_value: CounterValue) -> bool:
        if self.predicate_type is CounterPredicateType.NOT_LESS_THAN:
            return counter_value >= self.value
        elif self.predicate_type is CounterPredicateType.NOT_GREATER_THAN:
            return counter_value <= self.value
        elif self.predicate_type is CounterPredicateType.LESS_THAN:
            return counter_value < self.value

    def __str__(self) -> str:
        return f"{self.predicate_type}{self.value}"


class Guard(Mapping[CounterVariable, list[CounterPredicate]], Hashable):
    """Guard"""

    def __init__(
        self,
        guard: Optional[
            Mapping[CounterVariable, list[CounterPredicate]]
        ] = None,
    ) -> None:
        if guard is None:
            guard = {}
        self._guard = {
            counter_variable: copy(predicates)
            for counter_variable, predicates in guard.items()
        }

    @classmethod
    def less_than(
        cls, counter_variable: CounterVariable, value: int
    ) -> "Guard":
        return cls({counter_variable: [CounterPredicate.less_than(value)]})

    @classmethod
    def not_less_than(
        cls, counter_variable: CounterVariable, value: int
    ) -> "Guard":
        return cls({counter_variable: [CounterPredicate.not_less_than(value)]})

    @classmethod
    def not_greater_than(
        cls, counter_variable: CounterVariable, value: int
    ) -> "Guard":
        return cls(
            {counter_variable: [CounterPredicate.not_greater_than(value)]}
        )

    def __hash__(self) -> int:
        return hash(
            tuple((key, tuple(value)) for key, value in self._guard.items())
        )

    def __getitem__(self, key: CounterVariable) -> list[CounterPredicate]:
        if key not in self._guard:
            return []
        return self._guard[key]

    def __iter__(self) -> Iterator[CounterVariable]:
        return iter(self._guard)

    def __len__(self) -> int:
        return len(self._guard)

    def __setitem__(
        self, key: CounterVariable, value: list[CounterPredicate]
    ) -> None:
        self._guard[key] = value

    def __call__(self, counter_vector: CounterVector) -> bool:
        for counter_variable, predicates in self.items():
            # assert len(predicates) < 2, "Maybe?"
            for predicate in predicates:
                if not predicate(counter_vector[counter_variable]):
                    return False
        return True

    def __copy__(self) -> "Guard":
        return Guard(self)

    def __iadd__(self, other: "Guard") -> "Guard":
        for variable in other:
            self[variable] += other[variable]
        return self

    def __add__(self, other: "Guard") -> "Guard":
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

    def __call__(
        self, counter_value: Optional[CounterValue]
    ) -> Optional[CounterValue]:
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


class Action(
    Mapping[CounterVariable, CounterOperationComponent], Hashable
):
    """Action"""

    def __init__(
        self,
        action: Optional[
            Mapping[CounterVariable, CounterOperationComponent]
        ] = None,
    ) -> None:
        if action is None:
            action = {}
        self._action = {
            counter_variable: copy(components)
            for counter_variable, components in action.items()
        }

    @classmethod
    def increase(cls, counter_variable: CounterVariable) -> "Action":
        return cls({counter_variable: CounterOperationComponent.INCREASE})

    @classmethod
    def activate(cls, counter_variable: CounterVariable) -> "Action":
        return cls(
            {counter_variable: CounterOperationComponent.ACTIVATE_OR_RESET}
        )

    @classmethod
    def inactivate(cls, counter_variable: CounterVariable) -> "Action":
        return cls({counter_variable: CounterOperationComponent.INACTIVATE})

    def move_and_apply(self, counter_vector: CounterVector) -> CounterVector:
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
        return hash(
            tuple((key, tuple(value)) for key, value in self._action.items())
        )

    def __call__(self, counter_vector: CounterVector) -> CounterVector:
        return self.move_and_apply(copy(counter_vector))

    def __copy__(self) -> "Action":
        return Action(self)

    def __getitem__(
        self, key: CounterVariable
    ) -> CounterOperationComponent:
        if key not in self._action:
            return CounterOperationComponent.NO_OPERATION
        return self._action[key]

    def __setitem__(
        self, key: CounterVariable, value: CounterOperationComponent
    ) -> None:
        self._action[key] = value

    def __len__(self) -> int:
        return len(self._action)

    def __iter__(self) -> Iterator[CounterVariable]:
        return iter(self._action)

    def __iadd__(self, other: "Action") -> "Action":
        for variable in other:
            self[variable] *= other[variable]
        return self

    def __add__(self, other: "Action") -> "Action":
        new = copy(self)
        new += other
        return new

    def __str__(self) -> str:
        return ", ".join(
            f"c[{counter}]{operation}"
            for counter, operation in self.items()
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self._action == other._action
