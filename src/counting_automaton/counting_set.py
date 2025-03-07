"""Counting set"""

from typing import Iterable, Iterator, Optional

from more_collections import LinkedList
from more_collections import Node


class CountingSet(Iterable[int]):
    """Counting set"""

    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high
        self.offset = 1
        self.list: LinkedList[int] = LinkedList()
        self.max_node: Optional[Node[int]] = None

    def __iter__(self) -> Iterator[int]:
        for node in self.list:
            yield self.offset - node.value

    def increase(self) -> "CountingSet":
        self.offset += 1
        if self.max_node is not None:
            if self.max_node.value < self.offset - self.high:
                self.max_node = self.max_node.prev
        return self

    def merge(self, other: "CountingSet") -> "CountingSet":
        if self.offset < other.offset:
            raise ValueError("Cannot merge with a set that has a higher offset")
        raise NotImplementedError("Not implemented")
        # self.list.merge(other.list)
        # return self

    def check(self) -> bool:
        if self.max_node is None:
            return False
        return self.max_node.value <= self.offset - self.low

    def add_one(self) -> "CountingSet":
        self.list.prepend(self.offset - 1)
        if self.max_node is None:
            self.max_node = self.list.tail
        return self

    def __copy__(self) -> "CountingSet":
        return CountingSet.from_list(list(self), self.low, self.high)

    def __str__(self) -> str:
        return " -> ".join(map(str, self))

    @classmethod
    def from_list(cls, l: list[int], low: int, high: int) -> "CountingSet":
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

    def increase(self) -> "SparseCountingSet":
        tail = self.list.tail
        super().increase()
        if tail is None:
            return self
        if self.offset - tail.value > self.high:
            self.list.remove(tail)
        return self

    @property
    def k(self) -> int:
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
        if head3.value + self.k >= head.value:
            self.list.remove(head2)
        return self
