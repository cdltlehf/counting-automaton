"""SparseCounterConfig class module."""

from ..counting_set import SparseCountingSet
from ..position_counting_automaton import PositionCountingAutomaton
from .counter_config import CounterConfig


class SparseCounterConfig(CounterConfig):
    """Class for super-configurations using a sparse counting set"""

    def __init__(
        self,
        automaton: PositionCountingAutomaton,
    ):
        super().__init__(automaton, SparseCountingSet)
