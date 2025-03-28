"""Counter Config Base"""

import abc
from collections import defaultdict as dd
from copy import copy
import logging
from typing import Callable, Generic, Hashable, Iterator, Mapping, Optional, TypeVar

from more_collections import OrderedSet

from ..counter_vector import CounterOperationComponent
from ..counter_vector import Guard
from ..counting_set import CountingSet
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import FINAL_STATE
from ..position_counting_automaton import INITIAL_STATE
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State
from .super_config_base import SuperConfigBase

logger = logging.getLogger(__name__)

GLOBAL_COUNTER = CounterVariable(0)

_T = TypeVar("_T", bound=CountingSet)


class StateToCountingSet(dd[State, _T], Hashable, Generic[_T]):
    """Mapping from state to counting set"""

    def __init__(
        self,
        constructor: Callable[[int, Optional[int]], _T],
        low: int,
        high: Optional[int],
    ) -> None:
        super().__init__(lambda: constructor(low, high))
        self._constructor = constructor
        self.low = low
        self.high = high

    def __hash__(self) -> int:  # type: ignore
        return hash(tuple(sorted(self.items())))

    def apply_next_state_to_r_terms(
        self,
        next_state_to_r_terms: dict[
            State, dd[State, set[CounterOperationComponent]]
        ],
        current_state_to_reference_count: Optional[dd[State, int]] = None,
    ) -> tuple["StateToCountingSet[_T]", set[State]]:

        next_state_to_counting_set = StateToCountingSet(
            self._constructor, self.low, self.high
        )
        removed_next_states: set[State] = set()
        for next_state, r_terms in next_state_to_r_terms.items():
            logger.debug("Updating next state %d", next_state)

            is_one_added = False
            for current_state, operations in r_terms.items():
                logger.debug("Current state %d", current_state)
                for operation in operations:
                    assert operation != CounterOperationComponent.INACTIVATE

                    if operation == CounterOperationComponent.ACTIVATE_OR_RESET:
                        logger.debug("Activating or resetting")
                        is_one_added = True
                        continue

                    current_counting_set = self[current_state]
                    if current_state_to_reference_count is not None:
                        if current_state_to_reference_count[current_state] > 1:
                            current_counting_set = copy(current_counting_set)
                        current_state_to_reference_count[current_state] -= 1
                        assert (
                            current_state_to_reference_count[current_state] >= 0
                        )

                    if operation == CounterOperationComponent.INCREASE:
                        logger.debug("Increasing")
                        current_counting_set.increase()
                    else:
                        logger.debug("No operation")

                    if next_state not in next_state_to_counting_set:
                        next_state_to_counting_set[next_state] = (
                            current_counting_set
                        )
                    else:
                        next_state_to_counting_set[
                            next_state
                        ] |= current_counting_set  # type: ignore

                if is_one_added:
                    if next_state in next_state_to_counting_set:
                        next_state_to_counting_set[next_state].add_one()
                    else:
                        _ = next_state_to_counting_set[next_state]

                if next_state in next_state_to_counting_set:
                    logger.debug(
                        "Result of next state %d: %s",
                        next_state,
                        next_state_to_counting_set[next_state],
                    )

            if (
                next_state not in next_state_to_counting_set
                or next_state_to_counting_set[next_state].is_empty()
            ):
                removed_next_states.add(next_state)

        return next_state_to_counting_set, removed_next_states


class CounterConfigBase(
    SuperConfigBase,
    Mapping[CounterVariable, StateToCountingSet[_T]],
    abc.ABC,
    Generic[_T],
):
    """Class for super-configurations using a counting set"""

    _constructor: Callable[[int, Optional[int]], _T]

    def __init__(
        self,
        automaton: PositionCountingAutomaton,
        states: OrderedSet[State],
        counter_to_state_to_counting_set: dict[
            CounterVariable, StateToCountingSet[_T]
        ],
    ):
        super().__init__(automaton)

        self.counter_scopes = automaton.counter_scopes
        self.state_scopes = automaton.state_scopes
        self.counters: dict[CounterVariable, tuple[int, Optional[int]]]
        self.counters = automaton.counters

        self.states = states
        self._counter_to_state_to_counting_set = (
            counter_to_state_to_counting_set
        )

    @classmethod
    def get_initial(
        cls, automaton: PositionCountingAutomaton
    ) -> "CounterConfigBase[_T]":
        return cls(automaton, OrderedSet({INITIAL_STATE}), {})

    def __getitem__(self, counter: CounterVariable) -> StateToCountingSet[_T]:
        low, high = self.counters[counter]
        return self._counter_to_state_to_counting_set.get(
            counter, StateToCountingSet(self._constructor, low, high)
        )

    def __iter__(self) -> Iterator[CounterVariable]:
        return iter(self._counter_to_state_to_counting_set)

    def __len__(self) -> int:
        raise NotImplementedError()

    def to_json(self) -> dict[CounterVariable, dict[str, list[int]]]:
        jsonified = {
            counter: {
                str(state): list(counting_set)
                for state, counting_set in state_to_counting_set.items()
            }
            for counter, state_to_counting_set in self.items()
        }
        jsonified[GLOBAL_COUNTER] = {str(state): [1] for state in self.states}
        return jsonified

    def satisfies(self, state: State, guard: Guard[CounterVariable]) -> bool:
        for counter_variable, predicates in guard.items():
            if state not in self[counter_variable]:
                continue

            if len(predicates) == 0:
                continue

            counting_set = self[counter_variable][state]
            if not counting_set.check():
                return False

        return True

    def update(self, symbol: str) -> "SuperConfigBase":
        assert len(symbol) == 1
        logger.debug("Symbol: %s", symbol)

        # Compute next_states, r-terms and reference count
        next_states: OrderedSet[State] = OrderedSet()
        counter_variable_to_next_state_to_r_terms: dd[
            CounterVariable,
            dd[State, dd[State, set[CounterOperationComponent]]],
        ]
        counter_variable_to_next_state_to_r_terms = dd(
            lambda: dd(lambda: dd(set))
        )
        counter_variable_to_current_state_to_reference_count: dd[
            CounterVariable, dd[State, int]
        ]
        counter_variable_to_current_state_to_reference_count = dd(
            lambda: dd(int)
        )
        for current_state in self.states:
            arcs = self.automaton.follow[current_state]
            for guard, action, adjacent_state in arcs:
                if adjacent_state is FINAL_STATE:
                    continue

                if not self.satisfies(current_state, guard):
                    continue

                if not self.automaton.eval_state(adjacent_state, symbol):
                    continue

                next_states.append(adjacent_state)
                for counter_variable in self.state_scopes[adjacent_state]:
                    r_terms = counter_variable_to_next_state_to_r_terms[
                        counter_variable
                    ][adjacent_state]
                    current_state_to_reference_count = (
                        counter_variable_to_current_state_to_reference_count[
                            counter_variable
                        ]
                    )

                    operation = action.get(
                        counter_variable, CounterOperationComponent.NO_OPERATION
                    )
                    r_terms[current_state].add(operation)
                    if operation in {
                        CounterOperationComponent.INCREASE,
                        CounterOperationComponent.NO_OPERATION,
                    }:
                        current_state_to_reference_count[current_state] += 1

        logger.debug("Next states: %s", list(next_states))

        # Log
        for counter_variable in self.counter_scopes:
            next_state_to_r_terms = counter_variable_to_next_state_to_r_terms[
                counter_variable
            ]
            for next_state, r_terms in next_state_to_r_terms.items():
                logger.debug(
                    f"R-terms of next state {next_state}: {{%s}}",
                    ", ".join(
                        [
                            ", ".join(
                                [
                                    f"(variable {counter_variable}, "
                                    f"current state {current_state}){operation}"
                                    for operation in operations
                                ]
                            )
                            for current_state, operations in r_terms.items()
                        ]
                    ),
                )

        for counter_variable in self.counter_scopes:
            current_state_to_reference_count = (
                counter_variable_to_current_state_to_reference_count[
                    counter_variable
                ]
            )
            for (
                current_state,
                reference_count,
            ) in current_state_to_reference_count.items():
                logger.debug(
                    "Reference count of current state %d: %d",
                    current_state,
                    reference_count,
                )

        # Compute next counting sets
        counter_variable_to_next_state_to_counting_set: dict[
            CounterVariable, StateToCountingSet[_T]
        ] = {}
        removed_next_states: set[State] = set()
        for (
            counter_variable,
            next_state_to_r_terms,
        ) in counter_variable_to_next_state_to_r_terms.items():
            logger.debug("Updating counter variable %d", counter_variable)
            current_state_to_counting_set = self[counter_variable]
            current_state_to_reference_count = (
                counter_variable_to_current_state_to_reference_count[
                    counter_variable
                ]
            )
            next_state_to_counting_set, some_removed_next_states = (
                current_state_to_counting_set.apply_next_state_to_r_terms(
                    next_state_to_r_terms, current_state_to_reference_count
                )
            )
            removed_next_states |= some_removed_next_states
            counter_variable_to_next_state_to_counting_set[counter_variable] = (
                next_state_to_counting_set
            )

        logger.debug("Removed next states: %s", list(removed_next_states))
        for next_state in removed_next_states:
            next_states.remove(next_state)

        next_counter_config = self.__class__(
            self.automaton,
            next_states,
            counter_variable_to_next_state_to_counting_set,
        )
        return next_counter_config

    def is_final(self) -> bool:
        for current_state in self.states:
            arcs = self.automaton.follow[current_state]
            for guard, _, adjacent_state in arcs:

                if not self.satisfies(current_state, guard):
                    continue

                if adjacent_state == FINAL_STATE:
                    return True
        return False
