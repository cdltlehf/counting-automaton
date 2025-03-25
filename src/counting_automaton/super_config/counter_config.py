"""Counter Config"""

from collections import defaultdict as dd
from copy import copy
import logging
from typing import Callable, Iterator, Mapping, MutableMapping, Optional

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

        self.states = OrderedSet({INITIAL_STATE})
        self.counter_scopes = automaton.counter_scopes
        self.state_scopes = automaton.state_scopes
        self.counters: dict[CounterVariable, tuple[int, Optional[int]]]
        self.counters = automaton.counters

        self._counter_to_state_to_counting_set: dict[
            CounterVariable, dd[State, CountingSet]
        ] = {}
        # self._counter_to_state_to_counting_set = dd(
        #     lambda counter: dd(
        #         lambda state: self._constructor(*self.counters[counter])
        #     )
        # )

    def __getitem__(self, counter: CounterVariable) -> dict[State, CountingSet]:
        low, high = self.counters[counter]
        state_to_counting_set = (
            self._counter_to_state_to_counting_set.setdefault(
                counter, dd(lambda: self._constructor(low, high))
            )
        )
        return state_to_counting_set

    def __iter__(self) -> Iterator[CounterVariable]:
        return iter(self._counter_to_state_to_counting_set)

    def __len__(self) -> int:
        raise NotImplementedError()

    def to_json(self) -> dict[CounterVariable, dict[State, list[int]]]:
        jsonified = {
            counter: {
                state: list(counting_set)
                for state, counting_set in state_to_counting_set.items()
            }
            for counter, state_to_counting_set in self.items()
        }
        jsonified[GLOBAL_COUNTER] = {state: [1] for state in self.states}
        return jsonified

    def satisfies(self, state: State, guard: Guard[CounterVariable]) -> bool:
        for counter_variable, predicates in guard.items():
            if state not in self[counter_variable]:
                continue

            if len(predicates) == 0:
                continue

            counting_set = self[counter_variable][state]
            if counting_set.check():
                continue
            return False
        return True

    def update(self, symbol: str) -> "SuperConfigBase":
        assert len(symbol) == 1
        logger.debug("Symbol: %s", symbol)

        counter_variable_to_next_state_to_r_terms: dict[
            CounterVariable,
            dict[State, dd[State, set[CounterOperationComponent]]],
        ]
        counter_variable_to_next_state_to_r_terms = {
            counter: {state: dd(set) for state in counter_scope}
            for counter, counter_scope in self.counter_scopes.items()
        }
        counter_variable_to_current_state_to_reference_count: dd[
            CounterVariable, dd[State, int]
        ]
        counter_variable_to_current_state_to_reference_count = dd(
            lambda: dd(int)
        )

        next_counter_config = self.__class__(self.automaton)
        next_counter_config.states.remove(INITIAL_STATE)

        for current_state in self.states:
            arcs = self.automaton.follow[current_state]
            for guard, action, adjacent_state in arcs:
                if adjacent_state is FINAL_STATE:
                    continue

                if not self.satisfies(current_state, guard):
                    continue

                if not self.automaton.eval_state(adjacent_state, symbol):
                    continue

                next_counter_config.states.append(adjacent_state)

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

        states_with_empty_counting_set = set()
        for next_state in next_counter_config.states:
            for counter_variable in next_counter_config.state_scopes[
                next_state
            ]:
                next_state_to_counting_set = next_counter_config[
                    counter_variable
                ]
                next_state_to_r_terms = (
                    counter_variable_to_next_state_to_r_terms[counter_variable]
                )
                current_state_to_counting_set = self[counter_variable]
                current_state_to_reference_count = (
                    counter_variable_to_current_state_to_reference_count[
                        counter_variable
                    ]
                )

                logger.debug(
                    "===Updating counter %d of next state %d===",
                    counter_variable,
                    next_state,
                )
                r_terms = next_state_to_r_terms[next_state]

                for current_state, operations in r_terms.items():
                    for operation in operations:
                        assert operation != CounterOperationComponent.INACTIVATE

                        if (
                            operation
                            == CounterOperationComponent.ACTIVATE_OR_RESET
                        ):
                            if self.counters[counter_variable][1] == 0:
                                continue
                            next_state_to_counting_set[next_state].add_one()
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

                        if current_counting_set.is_empty():
                            continue

                        logger.debug("Merging %s", current_counting_set)
                        if next_state not in next_state_to_counting_set:
                            next_state_to_counting_set[next_state] = (
                                current_counting_set
                            )
                        else:
                            next_state_to_counting_set[
                                next_state
                            ] += current_counting_set

                if next_state not in next_state_to_counting_set:
                    states_with_empty_counting_set.add(next_state)
                    continue

                if next_state in next_state_to_counting_set:
                    logger.debug(
                        "===Result of counter %d of next state %d: %s===",
                        counter_variable,
                        next_state,
                        next_state_to_counting_set[next_state],
                    )
        for next_state in states_with_empty_counting_set:
            next_counter_config.states.remove(next_state)
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
