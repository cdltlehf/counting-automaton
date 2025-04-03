"""Multi-head counting set"""

import abc
import logging
from typing import Callable, Generic, Iterable, Optional, TypeVar
import warnings

from cai4py.more_collections import Node

from ..counting_set import BoundedCountingSet
from ..counting_set import CountingSet
from ..counting_set import SparseCountingSet

logger = logging.getLogger(__name__)


_T_co = TypeVar("_T_co", bound=CountingSet, covariant=True)
_Self = TypeVar("_Self", bound="MultiHeadCountingSetBase[CountingSet]")


class MultiHeadCountingSetBase(abc.ABC, Generic[_T_co]):
    """Multi-head counting set"""

    _constructor: Callable[[int, Optional[int]], CountingSet]

    def __init__(
        self,
        low: int,
        high: Optional[int],
    ) -> None:
        self.counting_set = self._constructor(low, high)
        self.heads = {0: self.counting_set.head}

    def update_deltas(self: _Self, deltas: list[int]) -> _Self:
        new_heads = {}
        for delta in deltas:
            if delta not in self.heads:
                assert delta - 1 in self.heads
                new_heads[delta] = self.get_head_after_increase(delta - 1)
            else:
                new_heads[delta] = self.heads[delta]
        self.heads = new_heads
        return self

    def get_head_after_increase(self, delta: int) -> Optional[Node[int]]:
        """Return the head for the given delta as if the set was increased"""
        head = self.heads[delta]

        if head is None:
            return None

        if self.counting_set.high is None:
            return head

        if self.head_value(delta) + 1 > self.counting_set.high:
            return head.prev
        else:
            return head

    def add_one(self: _Self) -> _Self:
        logger.debug("Adding one to multi-head counting set")
        if tuple(self.heads.keys()) != (0,):
            raise ValueError(
                "Cannot add one to multi-head counting set with non-zero deltas"
            )

        self.counting_set.add_one()
        self.heads[0] = self.counting_set.head
        if __debug__:
            self.sanity_check()
        return self

    def increase(self: _Self) -> _Self:
        logger.debug("Increasing multi-head counting set")
        new_heads = {
            delta: self.get_head_after_increase(delta) for delta in self.heads
        }
        self.heads = new_heads
        self.counting_set.increase()
        if __debug__:
            self.sanity_check()
        return self

    def head_value(self, delta: int) -> int:
        head = self.heads[delta]
        if head is None:
            return -1
        return delta + self.counting_set.offset - head.value

    def check(self, delta: int) -> bool:
        return self.head_value(delta) >= self.counting_set.low

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
                if (
                    self.counting_set.high is not None
                    and value > self.counting_set.high
                ):
                    break
                max_value = value
            assert max_value == self.head_value(delta), (
                max_value,
                delta,
                self.head_value(delta),
            )

    def merge(self: _Self, other: _Self) -> _Self:
        if list(self.heads.keys()) != list(other.heads.keys()):
            raise ValueError("Cannot merge sets with different deltas")
        new_heads = {}
        for delta, self_head, other_head in zip(
            self.heads.keys(), self.heads.values(), other.heads.values()
        ):
            self_value = (
                self.counting_set.offset - self_head.value if self_head else -1
            )
            other_value = (
                other.counting_set.offset - other_head.value
                if other_head
                else -1
            )
            if self_value < other_value:
                new_heads[delta] = other_head
            else:
                new_heads[delta] = self_head
        for delta in other.heads:
            self.heads[delta] = other.heads[delta]

        self.counting_set.merge(other.counting_set)
        self.heads = new_heads
        if __debug__:
            self.sanity_check()
        return self

    def __ior__(self: _Self, other: _Self) -> _Self:
        if self.counting_set.offset < other.counting_set.offset:
            return other.merge(self)
        else:
            return self.merge(other)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"counter-values ({self.counting_set.low}, {self.counting_set.high}): {self.counting_set}",
                f"heads: {[(delta, self.head_value(delta)) for delta in self.heads]}",
            ]
        )


class MultiHeadCountingSet(MultiHeadCountingSetBase[CountingSet]):
    """Multi-head counting set"""

    _constructor = CountingSet


class MultiHeadBoundedCountingSet(MultiHeadCountingSetBase[BoundedCountingSet]):
    """Multi-head bounded counting set"""

    _constructor = BoundedCountingSet


class MultiHeadSparseCountingSet(MultiHeadCountingSetBase[SparseCountingSet]):
    """Multi-head sparse counting set"""

    _constructor = SparseCountingSet
