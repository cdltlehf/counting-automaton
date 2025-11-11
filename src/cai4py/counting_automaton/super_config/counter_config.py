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

    def __hash__(self) -> int:  # type: ignore[override]
        """Compute a hash consistent with Mapping.__eq__ (based on items).

        We hash the tuple of current states and a sorted sequence of
        (counter, hash(state_to_counting_set)) pairs so the hash does not
        depend on dict insertion order.
        """
        # States are NewType(State, int) so convert to plain ints for stable hashing
        states_tuple = tuple(int(s) for s in self.states)

        # Each StateToCountingSet is hashable; create a stable, sorted representation
        items = tuple(
            sorted(
                (
                    (int(counter), hash(state_to_counting_set))
                    for counter, state_to_counting_set in self._counter_to_state_to_counting_set.items()
                )
            )
        )

        return hash((states_tuple, items))
