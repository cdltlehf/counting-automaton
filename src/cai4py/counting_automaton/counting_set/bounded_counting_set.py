"""Bounded counting set"""

import logging
import warnings

from .counting_set import CountingSet

logger = logging.getLogger(__name__)


class BoundedCountingSet(CountingSet):
    """Bounded counting set"""

    def increase(self) -> "BoundedCountingSet":
        """
        Implicitly increment all the counter values by one.
        """
        tail = self.list.tail
        super().increase()
        if tail is None or self.high is None:
            return self
        if self.offset - tail.value > self.high:
            self.list.remove(tail)
        if __debug__:
            BoundedCountingSet.sanity_check(self)
        return self

    def add_one(self) -> "BoundedCountingSet":
        """
        Put the value 1 in the counting-set
        """
        if self.high == 0:
            return self

        super().add_one()
        if __debug__:
            BoundedCountingSet.sanity_check(self)
        return self

    def sanity_check(self) -> None:
        if not __debug__:
            warnings.warn("Sanity checks are disabled", RuntimeWarning)
            return
        super().sanity_check()
        assert self.head == self.list.tail
