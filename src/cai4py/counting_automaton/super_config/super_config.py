"""SuperConfig"""

from collections import defaultdict as dd
from typing import Collection, Iterator, Optional

from cai4py.more_collections import OrderedSet

from ..counter_vector import CounterVector
from ..position_counting_automaton import Config
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import FINAL_STATE
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State
from .super_config_base import SuperConfigBase

ConfigDictType = dd[State, OrderedSet[CounterVector[CounterVariable]]]


class SuperConfig(SuperConfigBase, Collection[Config]):
    """Class for super-configurations using a set of configurations"""

    def __init__(self, automaton: PositionCountingAutomaton):
        super().__init__(automaton)
        self._configs: ConfigDictType = dd(OrderedSet)
        initial_config = automaton.get_initial_config()
        initial_state, initial_counter_vector = initial_config
        self._configs[initial_state].append(initial_counter_vector)

    @classmethod
    def get_initial(cls, automaton: PositionCountingAutomaton) -> "SuperConfig":
        return cls(automaton)

    def __iter__(self) -> Iterator[Config]:
        for state, counter_vectors in self._configs.items():
            if state == FINAL_STATE:
                continue
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
