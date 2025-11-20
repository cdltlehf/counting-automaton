from collections import defaultdict as dd
from collections.abc import Hashable
from copy import copy
from typing import Mapping, Optional

from cai4py.custom_counters.counter_base import CounterBase

from ..counting_automaton.computation_logging import ComputationStep
from ..counting_automaton.computation_logging import VERBOSE
from ..utils.util_logging import setup_debugger
from .counter_base import CounterVariable
from .counter_map import CounterPredicate

logger = setup_debugger(__name__)


class Guard(dd[CounterVariable, list[CounterPredicate]], Hashable):
    """Guard"""

    def __init__(
        self,
        guard: Optional[
            Mapping[CounterVariable, list[CounterPredicate]]
        ] = None,
    ) -> None:
        super().__init__(list)
        if guard is not None:
            self.update(guard)

    # All the counter guards that can be applied.

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

    def __hash__(self) -> int:  # type: ignore
        return hash(tuple((key, tuple(value)) for key, value in self.items()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Guard):
            return False
        return hash(self) == hash(other)

    def __call__(self, counters: dict[CounterVariable, CounterBase]) -> bool:
        for counter_var, predicates in self.items():
            counter = counters.get(counter_var)

            logger.log(VERBOSE, ComputationStep.EVAL_PREDICATE.value)

            for predicate in predicates:
                logger.debug("\t\t\tPredicate: %s", predicate)
                if not predicate(counter):
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

    def __reduce__(self):
        # Custom pickle support - return constructor and args
        return (
            self.__class__,
            (dict(self),),  # Pass the dict as the 'guard' parameter
            None,
            None,
            None,
        )
