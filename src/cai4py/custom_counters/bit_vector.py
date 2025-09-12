import numpy as np
import sys

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.data_collection import DataCollection, Operation

class BitVector(CounterBase):

    _data_collection = DataCollection({
        Operation.INIT: 2, 
        Operation.INC: 1, 
        Operation.UNION: 2, 
        Operation.GE: 1, 
        Operation.LE: 1
    })

    def __init__(self, lower_bound: int, upper_bound: int) -> "BitVector":

        BitVector._data_collection.full_update(Operation.INIT)
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if upper_bound == -1: # Infinity indicator
            self.counter = 1
            self.overflow = False
            BitVector._data_collection.half_update(1, Operation.INIT)
        else:
            self.vector = np.zeros(upper_bound, dtype=np.uint8)

            BitVector._data_collection.half_update(upper_bound // 20, Operation.INIT)
            BitVector._data_collection.max_memory_update(sys.getsizeof(self.vector))

            self.vector[0] = 1
            self.size = 1    # Vector has value 0 at all positions i >= size
            self.start = 0

            # Lowest in range is only initially valid if lower bound is 0 or 1
            if self.lower_bound == 1 or self.lower_bound == 0:
                self.lowest_in_range = 1
            else:
                self.lowest_in_range = -1

    def inc(self):

        if self.upper_bound == -1:
            self.counter += 1
            self.overflow = self.counter >= self.lower_bound
        else:

            BitVector._data_collection.full_update(Operation.INC)
            BitVector._data_collection.half_update(4, Operation.INC)

            # Move pointer left
            self.start = (self.start - 1) % len(self.vector)
            self.vector[self.start] = 0

            if self.size < len(self.vector):
                self.size = self.size + 1

            # Set minimum value within range
            if self.vector[(self.start + self.lower_bound - 1) % len(self.vector)] == 1:
                self.lowest_in_range = self.lower_bound
            elif self.vector[(self.start + self.lowest_in_range) % len(self.vector)] == 1:
                self.lowest_in_range += 1
            else:
                self.lowest_in_range = -1

    @staticmethod
    def union(counter_1: "BitVector", counter_2: "BitVector") -> "BitVector":

        if counter_1.upper_bound == -1:
            new_counter = BitVector(counter_1.lower_bound, counter_1.upper_bound)
            new_counter.counter = max(counter_1.counter, counter_2.counter)
            new_counter.overflow = counter_1.overflow | counter_2.overflow

            return new_counter
        else:
            BitVector._data_collection.full_update(Operation.UNION)
            BitVector._data_collection.half_update(len(counter_1.vector), Operation.UNION)

            assert len(counter_1.vector) == len(counter_2.vector), "Vectors not of equal length!"
            
            # Quick return option for union
            if counter_1.size == 0:
                return counter_2
            elif counter_2.size == 0:
                return counter_1
            
            # Perform union
            aux = []
            for i in range(len(counter_1.vector)):
                pos_1 = (counter_1.start + i) % len(counter_1.vector)
                pos_2 = (counter_2.start + i) % len(counter_2.vector)

                if pos_1 >= counter_1.size:
                    aux_1 = 0
                else:
                    aux_1 = counter_1.vector[pos_1]
                if pos_2 >= counter_2.size:
                    aux_2 = 0
                else:
                    aux_2 = counter_2.vector[pos_2]

                aux.append(aux_1 | aux_2)
            
            # Set new parameters
            new_counter = BitVector(counter_1.lower_bound, counter_1.upper_bound)
            new_counter.vector = aux
            new_counter.size =  max(counter_1.size, counter_2.size)
            new_counter.start = 0
           
            # Set new lowest in range
            if counter_1.lowest_in_range == -1 or counter_2.lowest_in_range == -1:
                new_counter.lowest_in_range = max(counter_1.lowest_in_range, counter_2.lowest_in_range)
            else:
                new_counter.lowest_in_range = min(counter_1.lowest_in_range, counter_2.lowest_in_range)

            return new_counter

    def ge_lower_bound(self) -> bool:

        if self.upper_bound == -1:
            return self.overflow

        BitVector._data_collection.full_update(Operation.GE)
        BitVector._data_collection.half_update(1, Operation.GE)

        # Check lowest in range
        return self.lowest_in_range != -1
    
    def le_upper_bound(self) -> bool:

        # If upper bound is infinity
        if self.upper_bound == -1:
            return True

        BitVector._data_collection.full_update(Operation.LE)
        BitVector._data_collection.half_update(1, Operation.LE)

        # Check lowest in range
        # NOTE: not technically correct but will never affect operation
        return self.lowest_in_range != -1

    def __str__(self):
        
        if self.upper_bound == -1:
            star = ""
            if self.overflow:
                star = "*"
    
            return f"({self.counter})" + star
        else:
            vals = []
            for i in range(0, len(self.vector)):
                pos = (self.start + i) % len(self.vector)

                if self.vector[pos]:
                    vals.append(i + 1)

            return vals.__str__() + f"({self.lowest_in_range})"
        
    def __repr__(self):
        return self.__str__()
    
    @classmethod
    def data_collection_details(cls):
        if cls._data_collection is None:
            return None
        else:
            data = cls._data_collection.get_data()
            cls._data_collection.reset_data()
            return data