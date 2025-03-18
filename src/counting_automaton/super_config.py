""" "Abstract class for super-configurations"""

import abc
from collections import defaultdict as dd
from copy import copy
from json import dumps
import logging
from typing import Any, Callable, Collection, Iterator, Mapping, MutableMapping, Optional

from more_collections import OrderedSet

from .counter_vector import CounterOperationComponent
from .counter_vector import CounterVector
from .counter_vector import Guard
from .counting_set import CountingSet
from .counting_set import SparseCountingSet
from .position_counting_automaton import Config
from .position_counting_automaton import CounterVariable
from .position_counting_automaton import FINAL_STATE
from .position_counting_automaton import INITIAL_STATE
from .position_counting_automaton import PositionCountingAutomaton
from .position_counting_automaton import State


class SuperConfigBase(abc.ABC):
    """Abstract class for super-configurations"""

    def __init__(self, automaton: PositionCountingAutomaton):
        self.automaton = automaton

    @abc.abstractmethod
    def to_json(self) -> Any:
        pass

    @abc.abstractmethod
    def update(self, symbol: str) -> "SuperConfigBase":
        pass

    @abc.abstractmethod
    def is_final(self) -> bool:
        pass

    @classmethod
    def get_computation(
        cls, automaton: PositionCountingAutomaton, w: str
    ) -> Iterator["SuperConfigBase"]:
        super_config = cls(automaton)
        yield super_config
        for symbol in w:
            super_config = super_config.update(symbol)
            yield super_config

    def __str__(self) -> str:
        return dumps(self.to_json())


ConfigDictType = dd[State, OrderedSet[CounterVector[CounterVariable]]]


class SuperConfig(SuperConfigBase, Collection[Config]):
    """Class for super-configurations using a set of configurations"""

    def __init__(self, automaton: PositionCountingAutomaton):
        super().__init__(automaton)
        self._configs: ConfigDictType = dd(OrderedSet)
        initial_config = automaton.get_initial_config()
        initial_state, initial_counter_vector = initial_config
        self._configs[initial_state].append(initial_counter_vector)

    def __iter__(self) -> Iterator[Config]:
        for state, counter_vectors in self._configs.items():
            for counter_vector in counter_vectors:
                yield (state, counter_vector)

    def to_json(self) -> list[tuple[int, list[Optional[int]]]]:
        return [
            (state, counter_vector.to_list())
            for (state, counter_vector) in self
        ]

    def __len__(self) -> int:
        return sum(map(len, self._configs.values()))

    def __contains__(self, config: object) -> bool:
        if not isinstance(config, tuple):
            return False
        state, counter_vector = config
        return counter_vector in self._configs[state]

    def update(self, symbol: str) -> "SuperConfig":
        assert len(symbol) == 1

        next_super_config: ConfigDictType = dd(OrderedSet)
        for config in self:
            next_configs = self.automaton.get_next_configs(config, symbol)
            for state, counter_vector in next_configs:
                next_super_config[state].append(counter_vector)
        self._configs = next_super_config
        return self

    def is_final(self) -> bool:
        return any(map(self.automaton.check_final, self))


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
        self.configs: ConfigDictType = dd(OrderedSet)
        self._counter_to_state_to_counting_set: dict[
            CounterVariable, dict[State, CountingSet]
        ] = {}
        self._constructor = constructor

        self._counter_to_state_to_counting_set[GLOBAL_COUNTER] = {}
        self._counter_to_state_to_counting_set[GLOBAL_COUNTER][
            INITIAL_STATE
        ] = self._constructor(0, None)
        self._counter_to_state_to_counting_set[GLOBAL_COUNTER].update(
            {state: self._constructor(0, None) for state in automaton.states}
        )

        for counter, (low, high) in automaton.counters.items():
            self._counter_to_state_to_counting_set[counter] = {
                state: self._constructor(low, high)
                for state in automaton.counter_scopes[counter]
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
        new_counter_config = self.__class__(self.automaton)

        assert len(symbol) == 1
        logging.debug("Symbol: %s", symbol)

        for counter_variable in self:
            logging.debug("Counter variable: %s", counter_variable)

            if counter_variable == GLOBAL_COUNTER:
                low, high = (0, None)
                counter_scope = set(self.automaton.states.keys()) | {
                    INITIAL_STATE
                }
            else:
                low, high = self.automaton.counters[counter_variable]
                counter_scope = set(
                    self.automaton.counter_scopes[counter_variable]
                )

            next_state_to_r_terms: dict[
                State, dd[State, set[CounterOperationComponent]]
            ]
            next_state_to_r_terms = {state: dd(set) for state in counter_scope}

            reference_count: dd[State, int] = dd(int)
            current_state_to_counting_set = self[counter_variable]
            next_state_to_counting_set = new_counter_config[counter_variable]

            for current_state in {INITIAL_STATE} | self.automaton.states.keys():
                arcs = self.automaton.follow[current_state]
                for guard, action, adjacent_state in arcs:
                    if not self.satisfies(current_state, guard):
                        continue

                    if adjacent_state is not FINAL_STATE:
                        if not self.automaton.eval_state(adjacent_state, symbol):
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
                        reference_count[adjacent_state] += 1

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

            for next_state in counter_scope:
                logging.debug("Updating next state %s", next_state)
                next_counting_set = self._constructor(low, high)
                next_state_to_counting_set[next_state] = next_counting_set

                for current_state, operations in next_state_to_r_terms[
                    next_state
                ].items():
                    for operation in operations:
                        assert operation != CounterOperationComponent.INACTIVATE

                        if (
                            operation
                            == CounterOperationComponent.ACTIVATE_OR_RESET
                        ):
                            next_counting_set.add_one()
                            continue

                        current_counting_set = current_state_to_counting_set[
                            current_state
                        ]

                        if reference_count[next_state] > 1:
                            logging.warning("Duplicating")
                            current_counting_set = copy(current_counting_set)
                            reference_count[next_state] -= 1

                        if operation == CounterOperationComponent.INCREASE:
                            current_counting_set.increase()
                        next_counting_set += current_counting_set

        return new_counter_config

    def is_final(self) -> bool:
        for current_state in {INITIAL_STATE} | self.automaton.states.keys():
            arcs = self.automaton.follow[current_state]
            for guard, _, adjacent_state in arcs:
                if not self.satisfies(current_state, guard):
                    continue

                if adjacent_state == FINAL_STATE:
                    return True
        return False


class SparseCounterConfig(CounterConfig):
    """Class for super-configurations using a sparse counting set"""

    def __init__(
        self,
        automaton: PositionCountingAutomaton,
    ):
        super().__init__(automaton, SparseCountingSet)
