"""Sparse counting set"""

import logging
from typing import Optional

from .counting_set import CountingSet


class SparseCountingSet(CountingSet):
    """Sparse counting set"""

    def increase(self) -> "SparseCountingSet":
        tail = self.list.tail
        super().increase()
        if tail is None or self.high is None:
            return self
        if self.offset - tail.value > self.high:
            self.list.remove(tail)
        return self

    @property
    def k(self) -> Optional[int]:
        if self.high is None:
            return None
        return self.high - self.low + 1

    def add_one(self) -> "SparseCountingSet":
        super().add_one()
        head = self.list.head
        assert head is not None
        head2 = head.next
        if head2 is None:
            return self
        head3 = head2.next
        if head3 is None:
            return self
        if self.k is None or head3.value + self.k >= head.value:
            self.list.remove(head2)
        return self

    def merge(self, other: "CountingSet") -> "CountingSet":
        logging.debug("Merging %s with %s", self, other)
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
        return self
