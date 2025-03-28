"""Counter Config"""

from ..counting_set import BoundedCountingSet
from ..counting_set import CountingSet
from ..counting_set import SparseCountingSet
from .counter_config_base import CounterConfigBase


class CounterConfig(CounterConfigBase[CountingSet]):
    """Class for counter-configuration"""

    _constructor = CountingSet


class BoundedCounterConfig(CounterConfigBase[BoundedCountingSet]):
    """Class for bounded counter-configuration"""

    _constructor = BoundedCountingSet


class SparseCounterConfig(CounterConfigBase[SparseCountingSet]):
    """Class for sparse counter-configuration"""

    _constructor = SparseCountingSet
