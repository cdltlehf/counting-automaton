"""SuperConfigBase"""

import abc
from collections import defaultdict as dd
from typing import Any, Iterator

from more_collections import OrderedSet

from ..counter_vector import CounterVector
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State

ConfigDictType = dd[State, OrderedSet[CounterVector[CounterVariable]]]


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
        super_config = cls.get_initial(automaton)
        yield super_config
        for symbol in w:
            super_config = super_config.update(symbol)
            yield super_config

    @classmethod
    @abc.abstractmethod
    def get_initial(
        cls, automaton: PositionCountingAutomaton
    ) -> "SuperConfigBase":
        pass

    def match(self, w: str) -> bool:
        if len(w) == 0:
            return self.is_final()

        last_super_config = None
        for super_config in self.get_computation(self.automaton, w):
            last_super_config = super_config
        assert last_super_config is not None
        return last_super_config.is_final()

    def __str__(self) -> str:
        return (
            "{\n"
            + ",\n".join(
                [
                    f"  {counter}: {state_to_counting_set}"
                    for counter, state_to_counting_set in self.to_json().items()
                ]
            )
            + "}"
        )
