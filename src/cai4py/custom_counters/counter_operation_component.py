from cai4py.custom_counters.counter_base import CounterBase
from cai4py.custom_counters.counter_type import CounterType
from typing import Optional
from cai4py.custom_counters.counter_base import CounterBase
from cai4py.custom_counters.counter_type import CounterType
from typing import Optional
from cai4py.custom_counters.counter_map import (
    StrEnum,
)
from ..utils.util_logging import setup_debugger
from ..counting_automaton.computation_logging import ComputationStep
from ..counting_automaton.computation_logging import VERBOSE

#    Operations that are applied to the counters.
logger = setup_debugger(__name__)


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
        self, counter: CounterBase, counter_type: CounterType
    ) -> CounterBase | None:
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
