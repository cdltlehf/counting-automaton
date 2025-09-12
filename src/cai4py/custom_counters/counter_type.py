from enum import Enum, auto

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.custom_counters.bit_vector import BitVector
from cai4py.custom_counters.lazy_counting_set import LazyCountingSet
from cai4py.custom_counters.counting_set import CountingSet
from cai4py.custom_counters.sparse_counting_set import SparseCountingSet

class CounterType(Enum):
    BIT_VECTOR = auto()
    LAZY_COUNTING_SET = auto()
    COUNTING_SET = auto()
    SPARSE_COUNTING_SET = auto()

    """
        Create a counter according to the necessary counter type.
    """
    def create_counter(self, lower_bound: int, upper_bound: int) -> CounterBase:
        match self:
            case CounterType.BIT_VECTOR:
                return BitVector(lower_bound, upper_bound)
            case CounterType.LAZY_COUNTING_SET:
                return LazyCountingSet(lower_bound, upper_bound)
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
            case CounterType.LAZY_COUNTING_SET:
                return LazyCountingSet.union
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
            case CounterType.LAZY_COUNTING_SET:
                return LazyCountingSet.data_collection_details()
            case CounterType.COUNTING_SET:
                return CountingSet.data_collection_details()
            case CounterType.SPARSE_COUNTING_SET:
                return SparseCountingSet.data_collection_details()
            case _:
                raise Exception("Unknown Counter Type!")