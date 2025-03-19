"""Counting set"""

from copy import copy
import logging
from typing import Iterable, Iterator, Optional

from more_collections import LinkedList
from more_collections import Node

from ..logging import ComputationStep
from ..logging import VERBOSE


class CountingSet(Iterable[int]):
    """Counting set"""

    def __init__(self, low: int, high: Optional[int]) -> None:
        self.low = low
        self.high = high
        self.offset = 1
        self.list: LinkedList[int] = LinkedList()
        self.max_node: Optional[Node[int]] = None
        self._dirty = False

    def sanity_check(self) -> None:
        return
        # pylint: disable=unreachable
        logging.warning("Do not use sanity_check in release code")
        max_node: Optional[Node[int]] = None
        for node in self.list:
            if node.next is not None:
                assert node.value > node.next.value
            if self.high is None or self.offset - node.value <= self.high:
                max_node = node

        assert self.max_node == max_node, (self.max_node, max_node)

        if self.max_node is None:
            return
        for node in self.list:
            if node == self.max_node:
                return
        assert False

    def mark_dirty(self) -> None:
        self._dirty = True

    def is_dirty(self) -> bool:
        return self._dirty

    def is_empty(self) -> bool:
        return self.list.is_empty()

    def __iter__(self) -> Iterator[int]:
        for node in self.list:
            yield self.offset - node.value

    def increase(self) -> "CountingSet":
        logging.log(VERBOSE, ComputationStep.APPLY_OPERATION.value)
        self.offset += 1
        if self.max_node is not None and self.high is not None:
            assert self.max_value is not None
            if self.max_value > self.high:
                self.max_node = self.max_node.prev
        self.sanity_check()
        return self

    def merge(self, other: "CountingSet") -> "CountingSet":
        logging.debug("Merging counter-set %s with %s", self, other)
        self.sanity_check()
        other.sanity_check()
        assert (self.low, self.high) == (other.low, other.high)
        if self.offset < other.offset:
            raise ValueError("Cannot merge with a set that has a higher offset")
        assert not self.is_dirty() and not other.is_dirty()

        max_node: Optional[Node[int]]
        if self.max_value < other.max_value:
            max_node = other.max_node
        else:
            max_node = self.max_node

        if other.list.tail is not None:
            for _ in other:
                logging.log(VERBOSE, ComputationStep.ACCESS_NODE_MERGE.value)
            other_max_value = other.offset - other.list.tail.value
            for value in self:
                logging.log(VERBOSE, ComputationStep.ACCESS_NODE_MERGE.value)
                if value >= other_max_value:
                    break

        for other_node in other.list:
            other_node.value += self.offset - other.offset

        self.list.merge(other.list, key=lambda x, y: y < x)
        other.mark_dirty()
        self.max_node = max_node
        self.sanity_check()
        return self

    def __iadd__(self, other: "CountingSet") -> "CountingSet":
        assert (self.low, self.high) == (other.low, other.high)

        if self.offset < other.offset:
            (self.offset, other.offset) = (other.offset, self.offset)
            (self.list, other.list) = (other.list, self.list)
            (self.max_node, other.max_node) = (other.max_node, self.max_node)
        return self.merge(other)

        # if self.offset < other.offset:
        #     return other.merge(self)
        # else:
        #     return self.merge(other)

    @property
    def max_value(self) -> int:
        if self.max_node is None:
            return -1
        return self.offset - self.max_node.value

    def check(self) -> bool:
        logging.log(VERBOSE, ComputationStep.EVAL_PREDICATE.value)
        max_value = self.max_value
        if max_value is None:
            return False
        return self.high is None or max_value >= self.low

    def add_one(self) -> "CountingSet":
        logging.log(VERBOSE, ComputationStep.APPLY_OPERATION.value)
        if (
            self.list.head is not None
            and self.offset - self.list.head.value == 1
        ):
            return self
        self.list.prepend(self.offset - 1)
        if self.max_node is None:
            self.max_node = self.list.tail
        self.sanity_check()
        return self

    def __copy__(self) -> "CountingSet":
        for _ in self:
            logging.log(VERBOSE, ComputationStep.ACCESS_NODE_CLONE.value)
        new = CountingSet(self.low, self.high)
        new.offset = self.offset
        new.list = copy(self.list)
        new.max_node = None
        if self.max_node is not None:
            for node in new.list:
                if node.value == self.max_node.value:
                    new.max_node = node
                    break
        return new

    def __str__(self) -> str:
        max_node_value = (
            self.max_node.value if self.max_node is not None else None
        )
        return (
            " -> ".join(map(str, self))
            + f" (offset: {self.offset}, max_node: {max_node_value}, max: {self.max_value})"
        )

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
