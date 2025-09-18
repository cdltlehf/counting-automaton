from collections import deque
import sys

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.data_collection import DataCollection
from cai4py.utils.data_collection import Operation


class CountingSet(CounterBase):

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

        CountingSet._data_collection.full_update(Operation.INIT)
        CountingSet._data_collection.half_update(1, Operation.INIT)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.queue = deque()
        self.queue.append(1)  # append at right end
        self.offset = 0
        self.overflow = False

        CountingSet._data_collection.max_memory_update(
            sys.getsizeof(self.queue)
        )

    def inc(self):
        if len(self.queue) != 0:

            CountingSet._data_collection.full_update(Operation.INC)
            CountingSet._data_collection.half_update(1, Operation.INC)

            self.offset += 1

            # get rightmost element
            if self.queue[-1] + self.offset > self.upper_bound:
                self.queue.pop()
                self.overflow = self.upper_bound == -1  # Infinity indicator

    @staticmethod
    def union(
        counter_1: "CountingSet", counter_2: "CountingSet"
    ) -> "CountingSet":

        CountingSet._data_collection.full_update(Operation.UNION)

        aux = deque()
        while len(counter_1.queue) != 0 and len(counter_2.queue) != 0:
            CountingSet._data_collection.half_update(1, Operation.UNION)

            if (
                counter_1.queue[0] + counter_1.offset
                > counter_2.queue[0] + counter_2.offset
            ):  # >
                aux.append(counter_1.queue.popleft() + counter_1.offset)
            elif (
                counter_1.queue[0] + counter_1.offset
                < counter_2.queue[0] + counter_2.offset
            ):  # <
                aux.append(counter_2.queue.popleft() + counter_2.offset)
            else:  # ==
                aux.append(counter_1.queue.popleft() + counter_1.offset)
                counter_2.queue.popleft()

        if len(counter_1.queue) == 0:
            leftover_counter = counter_2
        else:
            leftover_counter = counter_1

        CountingSet._data_collection.half_update(
            len(leftover_counter.queue) // 2, Operation.UNION
        )
        while len(leftover_counter.queue) != 0:
            aux.append(
                leftover_counter.queue.popleft() + leftover_counter.offset
            )

        new_counter = CountingSet(counter_1.lower_bound, counter_1.upper_bound)
        new_counter.queue = aux
        new_counter.offset = 0
        new_counter.overflow = counter_1.overflow | counter_2.overflow

        CountingSet._data_collection.max_memory_update(
            sys.getsizeof(new_counter.queue)
        )

        return new_counter

    def ge_lower_bound(self) -> bool:
        CountingSet._data_collection.full_update(Operation.GE)
        CountingSet._data_collection.half_update(1, Operation.GE)

        # Check overflow
        if self.upper_bound == -1 and self.overflow:
            return True

        if len(self.queue) == 0:
            return False

        # get rightmost element
        return self.queue[0] + self.offset >= self.lower_bound

    def le_upper_bound(self):
        CountingSet._data_collection.full_update(Operation.GE)
        CountingSet._data_collection.half_update(1, Operation.GE)

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
