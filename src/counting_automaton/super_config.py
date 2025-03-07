""" "Abstract class for super-configurations"""

import abc
from collections import defaultdict as dd
from json import dumps
from typing import Iterable, Iterator, Optional

from more_collections import OrderedSet

from .counter_vector import CounterVector
from .position_counting_automaton import Config
from .position_counting_automaton import CounterVariable
from .position_counting_automaton import PositionCountingAutomaton
from .position_counting_automaton import State


class SuperConfigBase(abc.ABC, Iterable[Config]):
    """Abstract class for super-configurations"""

    def __init__(self, automaton: PositionCountingAutomaton):
        self.automaton = automaton

    @abc.abstractmethod
    def to_json(self) -> dict:  # type: ignore
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


ConfigsType = dd[State, OrderedSet[CounterVector[CounterVariable]]]


class SuperConfig(SuperConfigBase):
    """Class for super-configurations using a set of configurations"""

    def __init__(self, automaton: PositionCountingAutomaton):
        super().__init__(automaton)
        self.configs: ConfigsType = dd(OrderedSet)
        initial_config = automaton.get_initial_config()
        initial_state, initial_counter_vector = initial_config
        self.configs[initial_state].append(initial_counter_vector)

    def __iter__(self) -> Iterator[Config]:
        for state, counter_vectors in self.configs.items():
            for counter_vector in counter_vectors:
                yield (state, counter_vector)

    def to_json(self) -> dict[State, list[list[Optional[int]]]]:
        return {
            state: [
                counter_vector.to_list() for counter_vector in counter_vectors
            ]
            for state, counter_vectors in self.configs.items()
        }

    def update(self, symbol: str) -> "SuperConfig":
        assert len(symbol) == 1

        next_super_config: ConfigsType = dd(OrderedSet)
        for config in self:
            next_configs = self.automaton.get_next_configs(config, symbol)
            for state, counter_vector in next_configs:
                next_super_config[state].append(counter_vector)
        self.configs = next_super_config
        return self

    def is_final(self) -> bool:
        return any(map(self.automaton.check_final, self))


class CounterConfig(SuperConfigBase):
    """Class for super-configurations using a set of configurations"""

    def __init__(self, automaton: PositionCountingAutomaton):
        super().__init__(automaton)
        self.configs: ConfigsType = dd(OrderedSet)

        initial_config = automaton.get_initial_config()
        initial_state, initial_counter_vector = initial_config
        self.configs[initial_state].append(initial_counter_vector)

    def __iter__(self) -> Iterator[Config]:
        for state, counter_vectors in self.configs.items():
            for counter_vector in counter_vectors:
                yield (state, counter_vector)

    def to_json(self) -> dict[State, list[list[Optional[int]]]]:
        return {
            state: [
                counter_vector.to_list() for counter_vector in counter_vectors
            ]
            for state, counter_vectors in self.configs.items()
        }

    def update(self, symbol: str) -> "SuperConfigBase":
        assert len(symbol) == 1

        next_super_config: ConfigsType = dd(OrderedSet)
        for config in self:
            next_configs = self.automaton.get_next_configs(config, symbol)
            for state, counter_vector in next_configs:
                next_super_config[state].append(counter_vector)
        self.configs = next_super_config
        return self

    def is_final(self) -> bool:
        return any(map(self.automaton.check_final, self))
