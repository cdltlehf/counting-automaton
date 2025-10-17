"""Determinized Counter Config"""

from typing import Optional, IO

from ..counting_set import BoundedCountingSet
from ..counting_set import CountingSet
from ..counting_set import SparseCountingSet
from .counter_config_base import StateToCountingSet
from .determinized_counter_config_base import DeterminizedCounterConfigBase
from .multi_head_counting_set import MultiHeadBoundedCountingSet
from .multi_head_counting_set import MultiHeadCountingSet
from .multi_head_counting_set import MultiHeadCountingSetBase
from .multi_head_counting_set import MultiHeadSparseCountingSet


class DeterminizedCounterConfig(DeterminizedCounterConfigBase[CountingSet]):
    """Determinized counter configuration"""

    @staticmethod
    def _key_constructor(
        low: int, high: Optional[int], log_file: IO
    ) -> StateToCountingSet[CountingSet]:
        return StateToCountingSet(CountingSet, low, high, log_file)

    @staticmethod
    def _value_constructor(
        low: int, high: Optional[int], log_file: IO
    ) -> MultiHeadCountingSetBase[CountingSet]:
        return MultiHeadCountingSet(low, high, log_file)


class DeterminizedBoundedCounterConfig(
    DeterminizedCounterConfigBase[BoundedCountingSet]
):
    """Multi-head bounded counting set"""

    @staticmethod
    def _key_constructor(
        low: int, high: Optional[int], log_file: IO
    ) -> StateToCountingSet[BoundedCountingSet]:
        return StateToCountingSet(BoundedCountingSet, low, high, log_file)

    @staticmethod
    def _value_constructor(
        low: int, high: Optional[int], log_file: IO
    ) -> MultiHeadCountingSetBase[BoundedCountingSet]:
        return MultiHeadBoundedCountingSet(low, high, log_file)


class DeterminizedSparseCounterConfig(
    DeterminizedCounterConfigBase[SparseCountingSet]
):
    """Multi-head sparse counting set"""

    @staticmethod
    def _key_constructor(
        low: int, high: Optional[int], log_file: IO
    ) -> StateToCountingSet[SparseCountingSet]:
        return StateToCountingSet(SparseCountingSet, low, high, log_file)

    @staticmethod
    def _value_constructor(
        low: int, high: Optional[int], log_file: IO
    ) -> MultiHeadCountingSetBase[SparseCountingSet]:
        return MultiHeadSparseCountingSet(low, high, log_file)
