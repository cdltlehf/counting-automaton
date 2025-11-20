"""SuperConfigBase"""

import abc
from json import dumps
from typing import Any, Iterator

from ..position_counting_automaton import PositionCountingAutomaton
from cai4py.custom_counters.counter_type import CounterType


class SuperConfigBase(abc.ABC):
    """Abstract class for super-configurations"""

    def __init__(
        self, automaton: PositionCountingAutomaton, counter_type: CounterType
    ):
        self.automaton = automaton
        self.counter_type = counter_type

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
        return dumps(self.to_json())

    def __hash__(self) -> int:
        raise NotImplementedError("This class is mutable and cannot be hashed.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SuperConfigBase):
            return False
        return str(self) == str(other)

    def __getstate__(self) -> dict:
        """Get the state for pickling."""
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        """Set the state from pickling."""
        self.__dict__.update(state)
