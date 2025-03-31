"""Determinized Counter Config Base"""

import abc
from collections import defaultdict as dd
from copy import copy
import logging
from typing import Callable, Generic, Iterator, Mapping, Optional, TypeVar

from more_collections import OrderedSet

from ..counter_vector import CounterOperationComponent
from ..counter_vector import Guard
from ..counting_set import CountingSet
from ..logging import ComputationStep
from ..logging import ComputationStepMark
from ..logging import VERBOSE
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import FINAL_STATE
from ..position_counting_automaton import INITIAL_STATE
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State
from .counter_config_base import follow_to_r_terms
from .counter_config_base import StateToCountingSet
from .multi_head_counting_set import MultiHeadCountingSetBase
from .super_config_base import SuperConfigBase

logger = logging.getLogger(__name__)

GLOBAL_COUNTER = CounterVariable(0)

_T = TypeVar("_T", bound=CountingSet)
_Self = TypeVar("_Self", bound="KeyToCountingSet[CountingSet]")


class KeyToCountingSet(
    dd[StateToCountingSet[_T], MultiHeadCountingSetBase[_T]], Generic[_T]
):
    """Mapping from StateToCountingSet to MultiHeadCountingSetBase"""

    def __init__(
        self,
        key_constructor: Callable[[int, Optional[int]], StateToCountingSet[_T]],
        value_constructor: Callable[
            [int, Optional[int]], MultiHeadCountingSetBase[_T]
        ],
        low: int,
        high: Optional[int],
    ) -> None:
        super().__init__(lambda: value_constructor(low, high))
        self.low = low
        self.high = high
        self._key_constructor = key_constructor
        self._value_constructor = value_constructor

    def to_json(self) -> dict[str, list[int]]:
        return {
            str(key): list(value.counting_set) for key, value in self.items()
        }

    def __str__(self) -> str:
        return str(self.to_json())

    def apply_next_state_to_r_terms(
        self,
        next_state_to_r_terms: dd[
            State, dd[State, set[CounterOperationComponent]]
        ],
        current_state_to_reference_count: dd[State, int],
    ) -> tuple["KeyToCountingSet[_T]", set[State]]:
        next_key_to_counting_set = KeyToCountingSet(
            self._key_constructor, self._value_constructor, self.low, self.high
        )

        new_states: set[State] = set()
        removed_next_states: set[State] = set()
        states_of_removed_arcs: set[State] = set()

        for next_state, r_terms in next_state_to_r_terms.items():
            removed_next_states.add(next_state)

            removed_current_states: set[State] = set()
            for current_state, operations in r_terms.items():
                if CounterOperationComponent.ACTIVATE_OR_RESET in operations:
                    new_states.add(next_state)
                    operations.remove(
                        CounterOperationComponent.ACTIVATE_OR_RESET
                    )
                if not operations:
                    removed_current_states.add(current_state)
            for current_state in removed_current_states:
                r_terms.pop(current_state)
            if not r_terms:
                states_of_removed_arcs.add(next_state)
        for next_state in states_of_removed_arcs:
            next_state_to_r_terms.pop(next_state)

        for r_terms in next_state_to_r_terms.values():
            for operations in r_terms.values():
                assert (
                    CounterOperationComponent.ACTIVATE_OR_RESET
                    not in operations
                ), "ACTIVATE_OR_RESET not removed"

        logger.log(
            VERBOSE,
            ComputationStepMark.START_DETERMINIZED_KEY_COMPUTATION.value,
        )
        new_key = self._key_constructor(self.low, self.high)
        for state in new_states:
            removed_next_states.remove(state)
            new_key[state].add_zero()
        logger.log(
            VERBOSE, ComputationStepMark.END_DETERMINIZED_KEY_COMPUTATION.value
        )

        if len(new_key) > 0:
            next_key_to_counting_set[new_key].add_one()

        for key, value in self.items():
            logger.debug("Key: %s", key)
            logger.debug("Value: %s", value.counting_set)

            logger.log(
                VERBOSE,
                ComputationStepMark.START_DETERMINIZED_KEY_COMPUTATION.value,
            )
            next_key: StateToCountingSet[_T]
            next_key, some_removed_next_states = (
                key.apply_next_state_to_r_terms(
                    next_state_to_r_terms,
                    copy(current_state_to_reference_count),
                )
            )
            logger.log(
                VERBOSE,
                ComputationStepMark.END_DETERMINIZED_KEY_COMPUTATION.value,
            )
            if len(next_key) == 0:
                continue

            logger.debug("Key after applying operations: %s", key)
            minimum_delta = min(
                next(iter(deltas)) for deltas in next_key.values()
            )
            assert minimum_delta <= 1

            if minimum_delta == 1:
                for key_deltas in next_key.values():
                    key_deltas.offset -= 1
                next_value = value.increase()
            else:
                next_value = value

            new_deltas: list[int] = []
            for deltas in next_key.values():
                new_deltas.extend(deltas)

            next_value = next_value.update_deltas(new_deltas)
            if next_key not in next_key_to_counting_set:
                next_key_to_counting_set[next_key] = next_value
            else:
                next_key_to_counting_set[next_key] |= next_value

            removed_next_states &= some_removed_next_states
            logger.debug("Next key: %s", next_key)

        logger.debug("Removed next states: %s", list(removed_next_states))
        return next_key_to_counting_set, removed_next_states


class DeterminizedCounterConfigBase(
    SuperConfigBase,
    Mapping[CounterVariable, KeyToCountingSet[_T]],
    abc.ABC,
    Generic[_T],
):
    """Class for determinized super-configurations using a counting set"""

    def __init__(
        self,
        automaton: PositionCountingAutomaton,
        states: OrderedSet[State],
        counter_to_key_to_counting_set: dict[
            CounterVariable, KeyToCountingSet[_T]
        ],
    ) -> None:
        super().__init__(automaton)

        self.counter_scopes = automaton.counter_scopes
        self.state_scopes = automaton.state_scopes
        self.counters: dict[CounterVariable, tuple[int, Optional[int]]]
        self.counters = automaton.counters

        self.states = states
        self._counter_to_key_to_counting_set = counter_to_key_to_counting_set

    @staticmethod
    def _key_constructor(
        low: int, high: Optional[int]
    ) -> StateToCountingSet[_T]:
        raise NotImplementedError()

    @staticmethod
    def _value_constructor(
        low: int, high: Optional[int]
    ) -> MultiHeadCountingSetBase[_T]:
        raise NotImplementedError()

    @classmethod
    def get_initial(
        cls, automaton: PositionCountingAutomaton
    ) -> "DeterminizedCounterConfigBase[_T]":
        return cls(automaton, OrderedSet([INITIAL_STATE]), {})

    def __getitem__(self, counter: CounterVariable) -> KeyToCountingSet[_T]:
        low, high = self.counters[counter]
        return self._counter_to_key_to_counting_set.get(
            counter,
            KeyToCountingSet(
                self._key_constructor, self._value_constructor, low, high
            ),
        )

    def __iter__(self) -> Iterator[CounterVariable]:
        return iter(self._counter_to_key_to_counting_set)

    def __len__(self) -> int:
        raise NotImplementedError()

    def to_json(self) -> dict[str, dict[str, list[int]]]:
        jsonified: dict[str, dict[str, list[int]]] = {}
        jsonified[str(GLOBAL_COUNTER)] = {
            str({str(state): [0] for state in self.states}): [1]
        }
        for counter, key_to_counting_set in self.items():
            jsonified[str(counter)] = key_to_counting_set.to_json()
        return jsonified

    def __str__(self) -> str:
        return (
            "{\n"
            + ",\n".join(
                [
                    f"  {counter}: "
                    + "{\n"
                    + ",\n".join(
                        [
                            f"    {key}: {counting_set}"
                            for key, counting_set in key_to_counting_set.items()
                        ]
                    )
                    + "\n  }"
                    for counter, key_to_counting_set in self.to_json().items()
                ]
            )
            + "\n}"
        )

    def satisfies(self, state: State, guard: Guard[CounterVariable]) -> bool:
        for counter_variable, predicates in guard.items():
            if len(predicates) == 0:
                continue

            key_to_counting_set = self[counter_variable]
            for (
                state_to_counting_set,
                multi_head_counting_set,
            ) in key_to_counting_set.items():

                if state not in state_to_counting_set:
                    continue

                delayed_increments = state_to_counting_set[state]

                if all(
                    not multi_head_counting_set.check(increment)
                    for increment in delayed_increments
                ):
                    return False
        return True

    def update(self, symbol: str) -> "DeterminizedCounterConfigBase[_T]":
        assert len(symbol) == 1
        logger.debug("Symbol: %s", symbol)

        # Compute next_states, r-terms and reference count
        (
            next_states,
            counter_variable_to_next_state_to_r_terms,
            counter_variable_to_next_state_to_reference_count,
        ) = follow_to_r_terms(
            symbol,
            self.states,
            self.automaton,
            self.satisfies,
            self.state_scopes,
        )

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

        # Compute next counting sets
        counter_variable_to_next_key_to_counting_set: dict[
            CounterVariable, KeyToCountingSet[_T]
        ] = {}
        removed_next_states: set[State] = set()
        for (
            counter_variable,
            next_state_to_r_terms,
        ) in counter_variable_to_next_state_to_r_terms.items():
            current_state_to_reference_count = (
                counter_variable_to_next_state_to_reference_count[
                    counter_variable
                ]
            )
            logger.debug("Updating counter variable %d", counter_variable)
            current_state_to_counting_set = self[counter_variable]
            next_key_to_counting_set: KeyToCountingSet[_T]
            next_key_to_counting_set, some_removed_next_states = (
                current_state_to_counting_set.apply_next_state_to_r_terms(
                    next_state_to_r_terms,
                    current_state_to_reference_count,
                )
            )
            removed_next_states &= some_removed_next_states
            counter_variable_to_next_key_to_counting_set[counter_variable] = (
                next_key_to_counting_set
            )

        logger.debug("Removed next states: %s", list(removed_next_states))
        for next_state in removed_next_states:
            next_states.remove(next_state)

        next_counter_config = self.__class__(
            self.automaton,
            next_states,
            counter_variable_to_next_key_to_counting_set,
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
