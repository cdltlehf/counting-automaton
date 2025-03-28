"""Counting set"""

from copy import copy
import logging
from typing import Iterable, Iterator, Optional, TypeVar
import warnings

from more_collections import Node
from more_collections import SortedLinkedList

from ..logging import ComputationStep
from ..logging import VERBOSE

logger = logging.getLogger(__name__)

Self = TypeVar("Self", bound="CountingSet")


class CountingSet(Iterable[int]):
    """Counting set"""

    def __init__(self, low: int, high: Optional[int]) -> None:
        self.low = low
        self.high = high
        self.offset = 1
        self.list: SortedLinkedList[int] = SortedLinkedList(lambda x, y: y < x)
        self.head: Optional[Node[int]] = None
        self._dirty = False
        self.add_one()

    def sanity_check(self) -> None:
        if not __debug__:
            warnings.warn("Sanity checks are disabled", RuntimeWarning)
            return
        assert not self._dirty
        self.list.sanity_check()
        max_value = -1
        for value in self:
            if self.high is not None and value > self.high:
                break
            max_value = value
        assert max_value == self.head_value, f"{max_value} != {self.head_value}"

    def mark_dirty(self) -> None:
        self._dirty = True

    def is_empty(self) -> bool:
        return self.list.is_empty()

    def __iter__(self) -> Iterator[int]:
        for node in self.list:
            yield self.offset - node.value

    def increase(self: Self) -> Self:
        logger.debug("Increasing counting-set %s", self)
        logger.log(VERBOSE, ComputationStep.APPLY_OPERATION.value)
        self.offset += 1
        if self.head is not None and self.high is not None:
            if self.head_value > self.high:
                self.head = self.head.prev
        if __debug__:
            CountingSet.sanity_check(self)
        return self

    def ordered_merge(self: Self, other: Self) -> Self:
        logger.debug("Merging counter-set %s with %s", self, other)
        assert (self.low, self.high) == (other.low, other.high)
        if self.offset < other.offset:
            raise ValueError("Cannot merge with a set that has a higher offset")

        max_node: Optional[Node[int]]
        if self.head_value < other.head_value:
            max_node = other.head
        else:
            max_node = self.head

        if other.list.tail is not None:
            for _ in other:
                logger.log(VERBOSE, ComputationStep.ACCESS_NODE_MERGE.value)
            other_max_value = other.offset - other.list.tail.value
            for value in self:
                logger.log(VERBOSE, ComputationStep.ACCESS_NODE_MERGE.value)
                if value >= other_max_value:
                    break

        for other_node in other.list:
            other_node.value += self.offset - other.offset

        self.list.merge(other.list)
        other.mark_dirty()
        self.head = max_node
        if __debug__:
            CountingSet.sanity_check(self)
        return self

    def __ior__(self: Self, other: Self) -> Self:
        if (self.low, self.high) != (other.low, other.high):
            raise ValueError("Cannot union counting-sets with different bounds")
        if self.offset < other.offset:
            return other.ordered_merge(self)
        else:
            return self.ordered_merge(other)

    @property
    def head_value(self) -> int:
        if self.head is None:
            return -1
        return self.offset - self.head.value

    def check(self) -> bool:
        logger.log(VERBOSE, ComputationStep.EVAL_PREDICATE.value)
        return self.head_value >= self.low

    def add_one(self: Self) -> Self:
        logger.log(VERBOSE, ComputationStep.APPLY_OPERATION.value)
        if self.list.head is not None:
            if self.offset - self.list.head.value == 1:
                return self
        self.list.prepend(self.offset - 1)
        if self.head is None and self.high != 0:
            self.head = self.list.head
        if __debug__:
            CountingSet.sanity_check(self)
        return self

    def __copy__(self: Self) -> Self:
        for _ in self:
            logger.log(VERBOSE, ComputationStep.ACCESS_NODE_CLONE.value)
        new = self.__class__(self.low, self.high)
        new.offset = self.offset
        new.list = copy(self.list)
        new.head = None
        if self.head is not None:
            for node in new.list:
                if node.value == self.head.value:
                    new.head = node
                    break
        return new

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
