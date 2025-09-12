import numpy as np
import math
from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.abstract_measurement import AbstractMeasurement, Operation

class ImprovedBitVector():

    _masks = [128, 64, 32, 16, 8, 4, 2, 1]

    _abstract_measurement = AbstractMeasurement({
        Operation.INIT: 2, 
        Operation.INC: 1, 
        Operation.UNION: 2, 
        Operation.GE: 1, 
        Operation.LE: 1
    })

    def __init__(self, lower_bound: int, upper_bound: int) -> "ImprovedBitVector":

        ImprovedBitVector._abstract_measurement.full_update(Operation.INIT)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if upper_bound == -1: # Infinity indicator
            self.vector = np.zeros(lower_bound - 1 / 8, dtype=np.uint8)
            ImprovedBitVector._abstract_measurement.half_update(lower_bound - 1, Operation.INIT)
        else:
            self.vector = np.zeros(math.ceil(upper_bound / 8), dtype=np.uint8)
            ImprovedBitVector._abstract_measurement.half_update(upper_bound, Operation.INIT)

        self.vector[0] = ImprovedBitVector._masks[0]
        self.global_pos = 0
        self.offset_pos = 0
        self.overflow = False

        # Lowest in range is only initially valid if lower bound is 0 or 1
        if self.lower_bound == 1 or self.lower_bound == 0:
            self.lowest_in_range = 1
        else:
            self.lowest_in_range = -1
        

    def inc(self):
        ImprovedBitVector._abstract_measurement.full_update(Operation.INC)
        ImprovedBitVector._abstract_measurement.half_update(1, Operation.INC)

        # Move pointer left
        if self.offset_pos == 0 and self.global_pos == 0:
            self.offset_pos = (self.upper_bound - 1) % 8 
            self.global_pos = len(self.vector) - 1

        elif self.offset_pos == 0:
            self.offset_pos == 7
            self.global_pos -= 1

        else:
            self.offset_pos -= 1

        print(f"Global pos {self.global_pos}")
        print(f"Offset {self.offset_pos}")

        self.vector[self.global_pos] = self.vector[self.global_pos] & ImprovedBitVector._masks[self.offset_pos]

    @staticmethod
    def _bitvector_value(counter: "ImprovedBitVector", index: int):
        # pos = counter.global_pos*8 + counter.offset_pos + index
        # pos = pos % counter.upper_bound

        pos = (counter.offset_pos + index) % counter.upper_bound
        print(f"Position: {pos}")

        if counter.vector[pos % len(counter.vector)] & ImprovedBitVector._masks[pos % 8] != 0:
            return index + 1
        else:
            return 0

    def __str__(self):
        
        vals = []
        for i in range(self.upper_bound):
            value = ImprovedBitVector._bitvector_value(self, i)
            print(value)
            if value != 0:
                vals.append(value)

        star = ""
        if self.overflow:
            star = "*"

        return vals.__str__() + "(" + str(self.lowest_in_range) + ")" + star
    
    def __repr__(self):
        return self.__str__()
    
    @classmethod
    def abstract_measurement_details(cls):
        if cls._abstract_measurement is None:
            return None
        else:
            abstraction = cls._abstract_measurement.get_abstraction()
            cls._abstract_measurement.reset_abstraction()
            return abstraction
        
if __name__ == "__main__":
    bv_1 = ImprovedBitVector(3, 7)
    print(bv_1)
    bv_1.inc()
    print(bv_1)
    
