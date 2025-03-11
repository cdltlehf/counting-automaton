"""Counting set"""

import logging
from typing import Iterable, Iterator, Optional

from more_collections import LinkedList, Node


class CountingSet(Iterable[int]):
    """Counting set"""

    def __init__(self, low: int, high: Optional[int]) -> None:
        self.low = low
        self.high = high
        self.offset = 1
        self.list: LinkedList[int] = LinkedList()
        self.max_node: Optional[Node[int]] = None

    def is_empty(self) -> bool:
        return self.list.is_empty()

    def __iter__(self) -> Iterator[int]:
        for node in self.list:
            yield self.offset - node.value

    def increase(self) -> "CountingSet":
        self.offset += 1
        if self.max_node is not None and self.high is not None:
            if self.max_node.value < self.offset - self.high:
                self.max_node = self.max_node.prev
        return self

    def merge(self, other: "CountingSet") -> "CountingSet":
        assert (self.low, self.high) == (other.low, other.high)
        if self.offset < other.offset:
            raise ValueError("Cannot merge with a set that has a higher offset")

        for other_node in other.list:
            other_node.value += self.offset - other.offset

        self.list.merge(other.list, key=lambda x, y: y < x)
        return self

    def __iadd__(self, other: "CountingSet") -> "CountingSet":
        assert (self.low, self.high) == (other.low, other.high)
        if self.offset < other.offset:
            (self.offset, other.offset) = (other.offset, self.offset)
            (self.list, other.list) = (other.list, self.list)
        return self.merge(other)

    def check(self) -> bool:
        if self.max_node is None:
            return False
        return self.max_node.value <= self.offset - self.low

    def add_one(self) -> "CountingSet":
        if (
            self.list.head is not None
            and self.offset - self.list.head.value == 1
        ):
            return self
        self.list.prepend(self.offset - 1)
        if self.max_node is None:
            self.max_node = self.list.tail
        return self

    def __copy__(self) -> "CountingSet":
        return CountingSet.from_list(list(self), self.low, self.high)

    def __str__(self) -> str:
        return " -> ".join(map(str, self))

    @classmethod
    def from_list(
        cls, l: list[int], low: int, high: Optional[int]
    ) -> "CountingSet":
        s = cls(low, high)
        last_n = None
        for n in reversed(l):
            if last_n is None:
                s.add_one()
                last_n = n
                continue

            if n >= last_n:
                raise ValueError("List must be sorted in increasing order")

            for _ in range(last_n - n):
                s.increase()

            s.add_one()
            last_n = n

        if last_n is not None:
            for _ in range(last_n - 1):
                s.increase()

        return s


class SparseCountingSet(CountingSet):
    """Sparse counting set"""

    def __init__(self, low: int, high: Optional[int]) -> None:
        super().__init__(low, high)

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
            _node = node
            node = node.prev

            node2 = _node.next
            if node2 is None:
                continue
            node3 = node2.next
            if node3 is None:
                continue

            if self.k is None or node3.value + self.k >= _node.value:
                self.list.remove(node2)
        return self
