"""Counter Base"""

import abc
from typing import Self


class CounterBase(abc.ABC):
    """
    Initialise the counter using the lower and upper bounds.
    """

    @abc.abstractmethod
    def __init__(self, lower_bound: int, upper_bound: int) -> None:
        pass

    @abc.abstractmethod
    def inc(self) -> Self:
        """Increment the counter by 1."""
        pass

    # @staticmethod
    # @abc.abstractmethod
    # def union(
    #     counter1: "CounterBase", counter2: "CounterBase"
    # ) -> "CounterBase":
    #     """Perform a union of two counters and return the result."""
    #     pass

    @abc.abstractmethod
    def ge_lower_bound(self) -> bool:
        """Is the counter greater than or equal to the lower bound."""
        pass

    @abc.abstractmethod
    def le_upper_bound(self) -> bool:
        """Is the counter less than or equal to the upper bound."""
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Dunder method for official string representation of counter."""
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        """String representation of the counter."""
        pass

    @abc.abstractmethod
    def to_list(self) -> list[int]:
        pass

    # @classmethod
    # @abc.abstractmethod
    # def data_collection_details(cls):
    #     """Return the abstract measurement details of the class."""
    #     pass
