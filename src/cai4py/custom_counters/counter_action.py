from cai4py.custom_counters.counter_base import CounterVariable
from cai4py.custom_counters.counter_base import CounterBase
from cai4py.custom_counters.counter_type import CounterType
from collections import defaultdict as dd
from collections.abc import Hashable
from typing import Any, Mapping, Optional
from copy import copy
from ..utils.util_logging import setup_debugger

from cai4py.custom_counters.counter_operation_component import (
    CounterOperationComponent,
)

logger = setup_debugger(__name__)


def _default_action_factory():
    """Factory function for Action's default_factory (needed for pickling)"""
    return CounterOperationComponent.no_operation()


class Action(dd[CounterVariable, CounterOperationComponent], Hashable):
    """Action"""

    def __init__(
        self,
        action: Optional[
            Mapping[CounterVariable, CounterOperationComponent]
        ] = None,
    ) -> None:
        super().__init__(_default_action_factory)
        if action is not None:
            self.update(action)

    @classmethod
    def increase(cls, counter_variable: CounterVariable) -> "Action":
        return cls({counter_variable: CounterOperationComponent.increase()})

    @classmethod
    def activate(
        cls,
        counter_variable: CounterVariable,
        lower_bound: int,
        upper_bound: Optional[int],
    ) -> "Action":
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
    def inactivate(cls, counter_variable: CounterVariable) -> "Action":
        return cls({counter_variable: CounterOperationComponent.inactivate()})

    def __call__(
        self,
        counters: dict[CounterVariable, CounterBase],
        counter_type: CounterType,
    ) -> dict[CounterVariable, CounterBase]:
        return self.move_and_apply(copy(counters), counter_type)

    def move_and_apply(
        self,
        counters: dict[CounterVariable, CounterBase],
        counter_type: CounterType,
    ) -> dict[CounterVariable, CounterBase]:
        # Loop over keys of default dict

        looped = False
        new_counters: dict[CounterVariable, CounterBase] = {}
        for variable in self:
            looped = True

            logger.debug(
                "\t\t\tCounter operation: %s(%s,%s)",
                self[variable].type,
                self[variable].lower_bound,
                self[variable].upper_bound,
            )
            action = self[variable]
            # Allow activation when counter variable is not yet present.
            if variable not in counters and action.type == action.Type.ACTIVATE_OR_RESET:  # type: ignore[attr-defined]
                # Create a placeholder counter (will be reset inside action call).
                counters[variable] = counter_type.create_counter(0, 0)
            counter = counters.get(variable)
            assert counter is not None or action.type == action.Type.INACTIVATE  # type: ignore[attr-defined]
            new_counter = action(counter, counter_type)  # type: ignore[arg-type]
            if new_counter is not None:
                new_counters[variable] = new_counter
            else:
                # The counter has been inactivated
                pass

        # Need to check if the loop ever activated.
        if not looped:
            return counters
        else:
            assert new_counters is not None
            return new_counters

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

    def __copy__(self) -> "Action":
        return Action(self)

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
