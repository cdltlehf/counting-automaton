from enum import Enum

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.custom_counters.bit_vector import BitVector
from cai4py.custom_counters.naive_counter import NaiveCounter
from cai4py.custom_counters.counting_set import CountingSet
from cai4py.custom_counters.sparse_counting_set import SparseCountingSet

"""
    Allow easy swap between counter methods.
"""


class CounterType(Enum):
    BIT_VECTOR = "Bit Vector"
    NAIVE_COUNTER = "Naive Counter"
    COUNTING_SET = "Counting-set"
    SPARSE_COUNTING_SET = "Sparse Counting-set"

    """
        Create a counter according to the necessary counter type.
    """

    def create_counter(self, lower_bound: int, upper_bound: int) -> CounterBase:
        match self:
            case CounterType.BIT_VECTOR:
                return BitVector(lower_bound, upper_bound)
            case CounterType.NAIVE_COUNTER:
                return NaiveCounter(lower_bound, upper_bound)
            case CounterType.COUNTING_SET:
                return CountingSet(lower_bound, upper_bound)
            case CounterType.SPARSE_COUNTING_SET:
                return SparseCountingSet(lower_bound, upper_bound)
            case _:
                raise Exception("Unknown Counter Type!")

    """
        Get union function of necessary counter.
    """

    def get_union(self):
        match self:
            case CounterType.BIT_VECTOR:
                return BitVector.union
            case CounterType.NAIVE_COUNTER:
                return NaiveCounter.union
            case CounterType.COUNTING_SET:
                return CountingSet.union
            case CounterType.SPARSE_COUNTING_SET:
                return SparseCountingSet.union
            case _:
                raise Exception("Unknown Counter Type!")

    """
        Extract abstract measurements.
    """

    def get_data_collection(self):
        match self:
            case CounterType.BIT_VECTOR:
                return BitVector.data_collection_details()
            case CounterType.NAIVE_COUNTER:
                return NaiveCounter.data_collection_details()
            case CounterType.COUNTING_SET:
                return CountingSet.data_collection_details()
            case CounterType.SPARSE_COUNTING_SET:
                return SparseCountingSet.data_collection_details()
            case _:
                raise Exception("Unknown Counter Type!")
