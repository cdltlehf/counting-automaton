"""Counter Config"""

from collections import defaultdict as dd
from copy import copy
import logging
from typing import Callable, Iterator, Mapping, MutableMapping, Optional

from ..counter_vector import CounterOperationComponent
from ..counter_vector import Guard
from ..counting_set import CountingSet
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import FINAL_STATE
from ..position_counting_automaton import INITIAL_STATE
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State
from .super_config_base import SuperConfigBase

GLOBAL_COUNTER = CounterVariable(0)


class CounterConfig(
    SuperConfigBase,
    Mapping[CounterVariable, MutableMapping[State, CountingSet]],
):
    """Class for super-configurations using a counting set"""

    def __init__(
        self,
        automaton: PositionCountingAutomaton,
        constructor: Callable[[int, Optional[int]], CountingSet] = CountingSet,
    ):
        super().__init__(automaton)
        self._constructor = constructor

        self.states = {INITIAL_STATE} | set(automaton.states)
        self.counter_scopes = {GLOBAL_COUNTER: self.states}
        self.counter_scopes.update(automaton.counter_scopes)
        self.counters: dict[CounterVariable, tuple[int, Optional[int]]]
        self.counters = {GLOBAL_COUNTER: (0, None)}
        self.counters.update(automaton.counters)

        self._counter_to_state_to_counting_set: dict[
            CounterVariable, dict[State, CountingSet]
        ]
        self._counter_to_state_to_counting_set = {
            counter: {
                state: self._constructor(low, high)
                for state in self.counter_scopes[counter]
            }
            for counter, (low, high) in self.counters.items()
        }
        self._counter_to_state_to_counting_set[GLOBAL_COUNTER][
            INITIAL_STATE
        ].add_one()

    def __getitem__(self, counter: CounterVariable) -> dict[State, CountingSet]:
        return self._counter_to_state_to_counting_set[counter]

    def __iter__(self) -> Iterator[CounterVariable]:
        return iter(self._counter_to_state_to_counting_set)

    def __len__(self) -> int:
        raise NotImplementedError()

    def to_json(self) -> dict[CounterVariable, dict[State, list[int]]]:
        return {
            counter: {
                state: list(counting_set)
                for state, counting_set in state_to_counting_set.items()
            }
            for counter, state_to_counting_set in self.items()
        }

    def satisfies(self, state: State, guard: Guard[CounterVariable]) -> bool:
        if self[GLOBAL_COUNTER][state].is_empty():
            return False

        for counter_variable, predicates in guard.items():
            if state not in self[counter_variable]:
                assert len(predicates) == 0
                continue
            counting_set = self[counter_variable][state]
            if counting_set.is_empty():
                return False
            if len(predicates) == 0 or counting_set.check():
                continue
            return False
        return True

    def update(self, symbol: str) -> "SuperConfigBase":
        assert len(symbol) == 1
        logging.debug("Symbol: %s", symbol)

        counter_variable_to_next_state_to_r_terms: dict[
            CounterVariable,
            dict[State, dd[State, set[CounterOperationComponent]]],
        ]
        counter_variable_to_next_state_to_r_terms = {
            counter: {state: dd(set) for state in counter_scope}
            for counter, counter_scope in self.counter_scopes.items()
        }
        counter_variable_to_current_state_to_reference_count: dict[
            CounterVariable, dd[State, int]
        ]
        counter_variable_to_current_state_to_reference_count = {
            counter: dd(int) for counter in self
        }

        def compute_r_terms(counter_variable: CounterVariable) -> None:
            counter_scope = self.counter_scopes[counter_variable]
            next_state_to_r_terms = counter_variable_to_next_state_to_r_terms[
                counter_variable
            ]
            current_state_to_reference_count = (
                counter_variable_to_current_state_to_reference_count[
                    counter_variable
                ]
            )

            for current_state in self.states:
                arcs = self.automaton.follow[current_state]
                for guard, action, adjacent_state in arcs:
                    if not self.satisfies(current_state, guard):
                        continue

                    if adjacent_state is not FINAL_STATE:
                        if not self.automaton.eval_state(
                            adjacent_state, symbol
                        ):
                            continue

                    operation = action.get(
                        counter_variable, CounterOperationComponent.NO_OPERATION
                    )
                    if adjacent_state not in counter_scope:
                        continue
                    r_terms = next_state_to_r_terms[adjacent_state]
                    r_terms[current_state].add(operation)
                    if operation in {
                        CounterOperationComponent.INCREASE,
                        CounterOperationComponent.NO_OPERATION,
                    }:
                        current_state_to_reference_count[current_state] += 1

        for counter_variable in self:
            compute_r_terms(counter_variable)

        for counter_variable in self:
            next_state_to_r_terms = counter_variable_to_next_state_to_r_terms[
                counter_variable
            ]
            for next_state, r_terms in next_state_to_r_terms.items():
                logging.debug(
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

        new_counter_config = self.__class__(self.automaton)
        for counter_variable in self:
            low, high = self.counters[counter_variable]
            next_state_to_counting_set = new_counter_config[counter_variable]
            next_state_to_r_terms = counter_variable_to_next_state_to_r_terms[
                counter_variable
            ]
            current_state_to_counting_set = self[counter_variable]
            current_state_to_reference_count = (
                counter_variable_to_current_state_to_reference_count[
                    counter_variable
                ]
            )

            for next_state in self.counter_scopes[counter_variable]:
                logging.debug(
                    "===Updating counter %d of next state %d===",
                    counter_variable,
                    next_state,
                )
                next_counting_set = self._constructor(low, high)
                r_terms = next_state_to_r_terms[next_state]

                for current_state, operations in r_terms.items():
                    for operation in operations:
                        assert operation != CounterOperationComponent.INACTIVATE

                        if (
                            operation
                            == CounterOperationComponent.ACTIVATE_OR_RESET
                        ):
                            next_counting_set.add_one()
                            logging.debug("Adding one")
                            continue

                        current_counting_set = current_state_to_counting_set[
                            current_state
                        ]

                        if current_state_to_reference_count[current_state] > 1:
                            current_counting_set = copy(current_counting_set)
                        current_state_to_reference_count[current_state] -= 1
                        assert (
                            current_state_to_reference_count[current_state] >= 0
                        )

                        if operation == CounterOperationComponent.INCREASE:
                            current_counting_set.increase()

                        logging.debug("Merging %s", current_counting_set)
                        next_counting_set += current_counting_set

                next_state_to_counting_set[next_state] = next_counting_set
                logging.debug(
                    "===Result of counter %d of next state %d: %s===",
                    counter_variable,
                    next_state,
                    next_counting_set,
                )

        return new_counter_config

    def is_final(self) -> bool:
        for current_state in self.states:
            arcs = self.automaton.follow[current_state]
            for guard, _, adjacent_state in arcs:

                if not self.satisfies(current_state, guard):
                    continue

                if adjacent_state == FINAL_STATE:
                    return True
        return False
