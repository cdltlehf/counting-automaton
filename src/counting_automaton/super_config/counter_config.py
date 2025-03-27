"""Counter Config"""

from ..counting_set import BoundedCountingSet
from ..counting_set import CountingSet
from ..counting_set import SparseCountingSet
from ..position_counting_automaton import PositionCountingAutomaton
from .counter_config_base import CounterConfigBase


class CounterConfig(CounterConfigBase):
    """Class for counter-configuration"""

    _constructor = CountingSet


class BoundedCounterConfig(CounterConfigBase):
    """Class for bounded counter-configuration"""

    _constructor = BoundedCountingSet


class SparseCounterConfig(CounterConfigBase):
    """Class for sparse counter-configuration"""

    _constructor = SparseCountingSet
