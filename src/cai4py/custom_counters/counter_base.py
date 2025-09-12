import abc

class CounterBase(abc.ABC):

    """
        Initialise the counter using the lower and upper bounds.
    """
    @abc.abstractmethod
    def __init__(self, lower_bound: int, upper_bound: int):
        pass

    """
        Increment the counter by 1.
    """
    @abc.abstractmethod
    def inc(self):
        pass

    """
        Perform a union of two counters and return the result.
    """
    @staticmethod
    @abc.abstractmethod
    def union(counter1: "CounterBase", counter2: "CounterBase") -> "CounterBase":
        pass

    """
        Is the counter greater than or equal to the lower bound.
    """
    @abc.abstractmethod
    def ge_lower_bound(self) -> bool:
        pass

    """
        Is the counter less than or equal to the upper bound.
    """
    @abc.abstractmethod
    def le_upper_bound(self) -> bool:
        pass
   
    """
        Dunder method for official string representation of counter.
    """
    @abc.abstractmethod
    def __repr__(self):
        pass

    """
        String representation of the counter.
    """
    @abc.abstractmethod
    def __str__(self):
        pass

    """
        Return the abstract measurement details of the class.
    """
    @classmethod
    @abc.abstractmethod
    def data_collection_details(cls):
        pass