"""Determinized Counter Config Base"""

import abc
from collections import defaultdict as dd
from copy import copy
import logging
from typing import Callable, Hashable, Iterable, Iterator, Mapping, MutableMapping, Optional

from more_collections import OrderedSet

from ..counter_vector import CounterOperationComponent
from ..counter_vector import Guard
from ..counting_set import CountingSet
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import FINAL_STATE
from ..position_counting_automaton import INITIAL_STATE
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State
from .multi_head_counting_set import MultiHeadCountingSetBase
from .super_config_base import SuperConfigBase

logger = logging.getLogger(__name__)

GLOBAL_COUNTER = CounterVariable(0)


class DeterminizedKeyBase(dict[State, CountingSet], Hashable):
    """Determinized key"""

    def __init__(
        self,
        low: int,
        high: Optional[int],
        constructor: Callable[[int, Optional[int]], CountingSet],
    ) -> None:
        self[INITIAL_STATE] = constructor(low, high)

    def to_json(self) -> dict[str, list[int]]:
        return {
            str(state): list(self[state])
            for state, counting_set in self.items()
        }

    def apply(self, follow: Follow) -> "DeterminizedKeyBase":
        next_key = copy(self)
        for current_state, counting_set in self.items():
            next_key[current_state] = counting_set.apply(follow)
        return next_key

    def sanity_check(self) -> None:
        has_zero = False
        for _, vs in self.items():
            assert vs.list.head is not None
            if vs.offset - vs.list.head.value == 0:
                has_zero = True
                break
        assert has_zero

    def __hash__(self) -> int:
        return hash(tuple())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeterminizedKey):
            return NotImplemented
        return self._entries == other._entries

    def __iter__(self) -> Iterator[tuple[State, int]]:
        return iter(self._entries)

    def __str__(self) -> str:
        return str(self._entries)


class DeterminizedCounterConfigBase(
    SuperConfigBase,
    Mapping[
        CounterVariable,
        MutableMapping[DeterminizedKey, MultiHeadCountingSetBase],
    ],
    abc.ABC,
):
    """Class for determinized super-configurations using a counting set"""

    def __init__(
        self,
        automaton: PositionCountingAutomaton,
        constructor: Callable[[int, Optional[int]], MultiHeadCountingSetBase],
    ):
        super().__init__(automaton)
        self._constructor = constructor

        self.states = OrderedSet({INITIAL_STATE})
        self.counter_scopes = automaton.counter_scopes
        self.state_scopes = automaton.state_scopes
        self.counters: dict[CounterVariable, tuple[int, Optional[int]]]
        self.counters = automaton.counters

        self.counter_to_key_to_counting_set: dict[
            CounterVariable, dd[DeterminizedKey, MultiHeadCountingSetBase]
        ] = {}

    def __getitem__(
        self, counter: CounterVariable
    ) -> dict[DeterminizedKey, MultiHeadCountingSetBase]:
        low, high = self.counters[counter]
        key_to_counting_set = self.counter_to_key_to_counting_set.setdefault(
            counter, dd(lambda: self._constructor(low, high))
        )
        return key_to_counting_set

    def __iter__(self) -> Iterator[CounterVariable]:
        return iter(self.counter_to_key_to_counting_set)

    def __len__(self) -> int:
        raise NotImplementedError()

    def to_json(self) -> dict[CounterVariable, dict[str, list[int]]]:
        jsonified = {
            counter: {
                str(key): list(counting_set.values(0))
                for key, counting_set in key_to_counting_set.items()
            }
            for counter, key_to_counting_set in self.items()
        }
        jsonified[GLOBAL_COUNTER] = {str(state): [1] for state in self.states}
        return jsonified

    def satisfies(self, state: State, guard: Guard[CounterVariable]) -> bool:
        for counter_variable, predicates in guard.items():
            if state not in self[counter_variable]:
                continue

            if len(predicates) == 0:
                continue

            key_to_counting_set = self[counter_variable]
            for key, counting_set in key_to_counting_set.items():
                for current_state, d in key:
                    if current_state != state:
                        continue
                    if not counting_set.check(d):
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
                            low, high = self.counters[counter_variable]
                            if next_state not in next_state_to_counting_set:
                                _ = next_state_to_counting_set[next_state]
                            else:
                                counting_set_one = self._constructor(low, high)
                                next_state_to_counting_set[
                                    next_state
                                ] += counting_set_one
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
