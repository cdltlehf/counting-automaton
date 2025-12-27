"""Counting set"""

from copy import copy
import logging
from typing import Iterable, Iterator, Optional, TypeVar
import warnings

from cai4py.collections import Node, SortedLinkedList

# import op_name_to_count and merge_set_sizes, clone_set_sizes from computation_logging
from ..instrumentation_vars import (
    op_name_to_count,
    merge_set_sizes,
    clone_set_sizes,
)

from ..computation_logging import ComputationStep
from ..computation_logging import VERBOSE

logger = logging.getLogger(__name__)

Self = TypeVar("Self", bound="CountingSet")


def _greater_than(x: int, y: int) -> bool:
    """Top-level key function used by CountingSet SortedLinkedList.

    Placed at module-level so it can be pickled.
    It preserves the original behaviour of lambda x,y: y < x.
    """
    return x > y


class CountingSet(Iterable[int]):
    """Counting-set data structure for counting automata"""

    def __init__(self, low: int, high: Optional[int]) -> None:
        """Initialize the counting-set"""
        self.low = low
        self.high = high
        self.offset = 1
        self.list: SortedLinkedList[int] = SortedLinkedList(_greater_than)
        self.head: Optional[Node[int]] = None
        self._dirty = False

    def sanity_check(self) -> None:
        """
        Perform a sanity check on the counting-set.

        The head value should be the maximum value in the counting-set.

        Raises: AssertionError: If the sanity check fails.
        """
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
        # TODO: Comment on what it means to be dirty
        self._dirty = True

    def is_empty(self) -> bool:
        """Check if the counting-set is empty"""
        return self.list.is_empty()

    def __iter__(self) -> Iterator[int]:
        """Iterate over the values in the counting-set"""
        for node in self.list:
            yield self.offset - node.value

    def increase(self: Self) -> Self:
        """Implicitly increment all the values in the counting-set"""
        logger.debug("Increasing counting-set %s", self)
        logger.log(VERBOSE, ComputationStep.APPLY_OPERATION.value)
        self._inc_op_count("INCREASE")
        self.offset += 1
        if self.head is not None and self.high is not None:
            if self.head_value > self.high:
                self.head = self.head.prev
        if __debug__:
            CountingSet.sanity_check(self)
        return self

    def _inc_op_count(self, op_name: str) -> None:
        """Increment the operation count for the given operation name"""
        global op_name_to_count
        op_name_to_count[op_name] = op_name_to_count[op_name] + 1

    def _save_merge_set_sizes(
        self, size1: int, density1: float, size2: int, density2: float
    ) -> None:
        """Save the sizes and densities of the two counting-sets involved in a MERGE operation"""
        merge_set_sizes.append((size1, density1, size2, density2))

    def _save_clone_set_sizes(self, size: int, density: float) -> None:
        """Save the size and density of the counting-set involved in a CLONE operation"""
        clone_set_sizes.append((size, density))

    def merge(self: Self, other: Self) -> Self:
        """Merge `other` counting-set into `self`"""
        logger.debug("Merging counter-set %s with %s", self, other)
        density = len(self.list) / self.high if self.high is not None else 0
        other_density = (
            len(other.list) / other.high if other.high is not None else 0
        )

        self._inc_op_count("MERGE")
        self._save_merge_set_sizes(
            len(self.list), density, len(other.list), other_density
        )

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
        """Merge the counting-set with the smaller offset into the counting-set with the larger offset"""
        if self.offset < other.offset:
            return other.merge(self)
        else:
            return self.merge(other)

    @property
    def head_value(self) -> int:
        """Get the head value of the counting-set"""
        if self.head is None:
            return -1
        return self.offset - self.head.value

    def check(self) -> bool:
        """Check if the head value is at least the low value"""
        logger.log(VERBOSE, ComputationStep.EVAL_PREDICATE.value)
        self._inc_op_count("CHECK")
        logger.debug(
            "Checking counting-set %s, head: %s, low: %s",
            self,
            self.head_value,
            self.low,
        )
        return self.head_value >= self.low

    def add_one(self: Self) -> Self:
        """Put the value 1 into the counting-set"""
        logger.log(VERBOSE, ComputationStep.APPLY_OPERATION.value)
        self._inc_op_count("ADD_ONE")
        if self.list.head is not None:
            if next(iter(self)) == 1:
                return self
        self.list.prepend(self.offset - 1)
        if self.head is None and self.high != 0:
            self.head = self.list.head
        if __debug__:
            CountingSet.sanity_check(self)
        return self

    def add_zero(self: Self) -> Self:
        """Put the value 0 into the counting-set"""
        self._inc_op_count("ADD_ZERO")
        logger.log(VERBOSE, ComputationStep.APPLY_OPERATION.value)
        if self.list.head is not None:
            if next(iter(self)) == 0:
                return self
        self.list.prepend(self.offset)
        if self.head is None:
            self.head = self.list.head
        if __debug__:
            CountingSet.sanity_check(self)
        return self

    def __copy__(self: Self) -> Self:
        """
        Create a shallow copy of the counting-set
        """
        density = len(self.list) / self.high if self.high is not None else 0
        self._save_clone_set_sizes(len(self.list), density)
        self._inc_op_count("CLONE")
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
        """String representation of the counting-set"""
        return " -> ".join(map(str, self))

    @classmethod
    def from_list(
        cls, l: list[int], low: int, high: Optional[int]
    ) -> "CountingSet":
        """Create a counting-set from a list of integers"""
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
