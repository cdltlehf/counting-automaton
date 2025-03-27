"""Multi-head counting set"""

import abc
import logging
from typing import Callable, Iterable, Optional
import warnings

from more_collections import Node

from ..counting_set import BoundedCountingSet
from ..counting_set import CountingSet
from ..counting_set import SparseCountingSet

logger = logging.getLogger(__name__)


class MultiHeadCountingSetBase(abc.ABC):
    """Multi-head counting set"""

    def __init__(
        self,
        deltas: Iterable[int],
        constructor: Callable[[int, Optional[int]], CountingSet] = CountingSet,
        low: int,
        high: Optional[int],
    ) -> None:
        self.counting_set = constructor(low, high)
        self.heads = {delta: self.counting_set.head for delta in deltas}

    def update_deltas(self, deltas: set[int]) -> "MultiHeadCountingSetBase":
        for delta in deltas:
            if delta not in self.heads:
                if delta - 1 not in self.heads:
                    raise ValueError(
                        f"Cannot update delta {delta} without delta {delta - 1}"
                    )
                self.heads[delta] = self.get_head_after_increase(delta - 1)

        for delta in self.heads:
            if delta not in deltas:
                self.heads.pop(delta)
        return self

    def get_head_after_increase(self, delta: int) -> Optional[Node[int]]:
        """Return the head for the given delta as if the set was increased"""
        head = self.heads[delta]
        if head is not None and self.counting_set.high is not None:
            if self.head_value(delta) + 1 > self.counting_set.high:
                return head.prev
        return None

    def increase(self) -> "MultiHeadCountingSetBase":
        self.counting_set.increase()
        return self

    def update_delta(self, delta: int) -> Optional[Node[int]]:
        head = self.heads[delta]
        if head is None:
            return None
        return head.prev

    def head_value(self, delta: int) -> int:
        head = self.heads[delta]
        if head is None:
            return -1
        return delta + self.counting_set.offset - head.value

    def check(self, delta: int) -> bool:
        return self.head_value(delta) is not None

    def values(self, delta: int) -> Iterable[int]:
        for value in self.counting_set:
            yield value + delta

    def sanity_check(self) -> None:
        if not __debug__:
            warnings.warn("Sanity checks are disabled", RuntimeWarning)
            return
        self.counting_set.sanity_check()
        assert 0 in self.heads

        for delta in self.heads:
            max_value = -1
            for value in self.values(delta):
                if self.high is not None and value > self.high:
                    break
                max_value = value
            assert max_value == self.head_value(delta)


class MultiHeadCountingSet(MultiHeadCountingSetBase):
    def __init__(
        self, low: int, high: Optional[int], deltas: Iterable[int]
    ) -> None:
        super().__init__(low, high, deltas, CountingSet)


class MultiHeadBoundedCountingSet(MultiHeadCountingSetBase):
    def __init__(
        self, low: int, high: Optional[int], deltas: Iterable[int]
    ) -> None:
        super().__init__(low, high, deltas, BoundedCountingSet)


class MultiHeadSparseCountingSet(MultiHeadCountingSetBase):
    def __init__(
        self, low: int, high: Optional[int], deltas: Iterable[int]
    ) -> None:
        super().__init__(low, high, deltas, SparseCountingSet)
