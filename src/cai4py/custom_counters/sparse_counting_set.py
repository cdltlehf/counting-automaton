from collections import deque
import sys
from typing import Self

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.data_collection import DataCollection
from cai4py.utils.data_collection import Operation


class SparseCountingSet(CounterBase):

    _data_collection = DataCollection(
        {
            Operation.INIT: 1,
            Operation.INC: 1,
            Operation.UNION: 2,
            Operation.GE: 1,
            Operation.LE: 1,
        }
    )

    def __init__(self, lower_bound: int, upper_bound: int) -> None:
        SparseCountingSet._data_collection.count_update(Operation.INIT)
        SparseCountingSet._data_collection.full_update(Operation.INIT)
        SparseCountingSet._data_collection.half_update(2, Operation.INIT)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.queue: deque[int] = deque()
        self.queue.append(1)  # append at right end
        self.offset = 0
        self.overflow = False

        SparseCountingSet._data_collection.max_memory_update(
            sys.getsizeof(self.queue)
        )

    def inc(self) -> Self:
        SparseCountingSet._data_collection.count_update(Operation.INC)
        SparseCountingSet._data_collection.full_update(Operation.INC)
        SparseCountingSet._data_collection.half_update(1, Operation.INC)

        if len(self.queue) != 0:

            self.offset += 1
            if self.queue[0] + self.offset > self.upper_bound:
                self.queue.popleft()

                if self.upper_bound == -1:
                    self.overflow = True

            # only need 1 element in range
            if len(self.queue) >= 2:
                if self.queue[1] + self.offset >= self.lower_bound:
                    self.queue.popleft()
        return self

    @staticmethod
    def _combine_sets(
        counter_1: SparseCountingSet, counter_2: SparseCountingSet
    ) -> SparseCountingSet:
        combination = []

        while len(counter_1.queue) > 0 or len(counter_2.queue) > 0:
            SparseCountingSet._data_collection.half_update(1, Operation.UNION)
            val_1 = -1
            if len(counter_1.queue) > 0:
                val_1 = counter_1.queue[0] + counter_1.offset

            val_2 = -1
            if len(counter_2.queue) > 0:
                val_2 = counter_2.queue[0] + counter_2.offset

            if val_1 > val_2:
                combination.append(val_1)
                counter_1.queue.popleft()
            elif val_2 > val_1:
                combination.append(val_2)
                counter_2.queue.popleft()
            else:
                combination.append(val_1)
                counter_1.queue.popleft()
                counter_2.queue.popleft()

        return combination

    @staticmethod
    def union(
        counter_1: SparseCountingSet, counter_2: SparseCountingSet
    ) -> SparseCountingSet:
        SparseCountingSet._data_collection.count_update(Operation.UNION)
        SparseCountingSet._data_collection.full_update(Operation.UNION)

        aux: list[int] = deque()

        # combine both (all values are out of range)
        combined = SparseCountingSet._combine_sets(counter_1, counter_2)
        SparseCountingSet._data_collection.half_update(
            len(combined) // 2, Operation.UNION
        )

        # infinity indicator (only have to know max value)
        if counter_1.upper_bound == -1:
            aux.append(combined[0])

        else:
            delta = counter_1.upper_bound - counter_1.lower_bound

            # milestone for first element
            if len(combined) >= 1:
                aux.append(combined[0])
                milestone = combined[0]

            if len(combined) >= 2:
                for element in combined[1:]:

                    # best lowest in bound
                    if element > counter_1.lower_bound:
                        aux.pop()
                        milestone = element
                    else:

                        # must be a milestone
                        if milestone - element >= delta:

                            # check if previous was marked as milestone
                            if aux[-1] != milestone:
                                aux.pop()
                            milestone = element

                        # previous incorrectly added
                        else:
                            if aux[-1] != milestone:
                                aux.pop()

                    aux.append(element)

        new_counter = SparseCountingSet(
            counter_1.lower_bound, counter_1.upper_bound
        )
        new_counter.queue = aux
        new_counter.offset = 0
        new_counter.overflow = counter_1.overflow | counter_2.overflow

        SparseCountingSet._data_collection.max_memory_update(sys.getsizeof(aux))

        return new_counter

    def ge_lower_bound(self):
        SparseCountingSet._data_collection.count_update(Operation.GE)
        SparseCountingSet._data_collection.full_update(Operation.GE)
        SparseCountingSet._data_collection.half_update(1, Operation.GE)

        # Check overflow
        if self.upper_bound == -1 and self.overflow:
            return True

        if len(self.queue) == 0:
            return False

        # get rightmost element
        return self.queue[0] + self.offset >= self.lower_bound

    def le_upper_bound(self):
        SparseCountingSet._data_collection.count_update(Operation.LE)
        SparseCountingSet._data_collection.full_update(Operation.LE)
        SparseCountingSet._data_collection.half_update(1, Operation.LE)

        if len(self.queue) == 0:
            return False

        # get leftmost element
        return self.queue[-1] + self.offset <= self.upper_bound

    def __str__(self):
        vals = [self.queue[i] + self.offset for i in range(len(self.queue))]

        star = ""
        if self.overflow:
            star = "*"

        return vals.__str__() + star

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
