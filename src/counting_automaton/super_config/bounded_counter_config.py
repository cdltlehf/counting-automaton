"""BoundedCounterConfig class module."""

from ..counting_set import BoundedCountingSet
from ..position_counting_automaton import PositionCountingAutomaton
from .counter_config import CounterConfig


class BoundedCounterConfig(CounterConfig):
    """Class for super-configurations using a sparse counting set"""

    def __init__(
        self,
        automaton: PositionCountingAutomaton,
    ):
        super().__init__(automaton, BoundedCountingSet)
