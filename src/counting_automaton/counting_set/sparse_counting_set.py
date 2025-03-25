"""Sparse counting set"""

import logging
from typing import Optional
import warnings

from .bounded_counting_set import BoundedCountingSet
from .counting_set import CountingSet

logger = logging.getLogger(__name__)


class SparseCountingSet(BoundedCountingSet):
    """Sparse counting set"""

    @property
    def k(self) -> Optional[int]:
        if self.high is None:
            return None
        return self.high - self.low + 1

    def add_one(self) -> "SparseCountingSet":
        super().add_one()
        head = self.list.head
        if head is None:
            return self
        assert head is not None
        head2 = head.next
        if head2 is None:
            return self
        head3 = head2.next
        if head3 is None:
            return self
        if self.k is None or head3.value + self.k >= head.value:
            self.list.remove(head2)

        if __debug__:
            SparseCountingSet.sanity_check(self)
        return self

    def merge(self, other: "CountingSet") -> "SparseCountingSet":
        if not isinstance(other, BoundedCountingSet):
            raise TypeError("Can only merge with another sparse counting set")
        node = other.list.tail
        super().merge(other)

        while node is not None:
            current_node = node
            node = node.prev
            node2 = current_node.next
            if node2 is None:
                continue
            node3 = node2.next
            if node3 is None:
                continue

            if self.k is None or node3.value + self.k >= current_node.value:
                self.list.remove(node2)

        if __debug__:
            self.sanity_check()
        return self

    def sanity_check(self) -> None:
        if not __debug__:
            warnings.warn("Sanity checks are disabled", RuntimeWarning)
            return
        super().sanity_check()
        values = list(self)
        for v1, v2 in zip(values, values[2:]):
            assert self.k is not None, str(self)
            assert v1 < v2 - self.k
