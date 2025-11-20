from collections import deque

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.data_collection import DataCollection

"""
    Counting-set counter method.
"""


class CountingSet(CounterBase):

    _data_collection = DataCollection()

    def __init__(self, lower_bound: int, upper_bound: int) -> "CountingSet":

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.queue = deque()
        self.queue.append(1)
        self.offset = 0
        self.overflow = False

        CountingSet._data_collection.max_density_update(len(self.queue))

    def inc(self):
        if len(self.queue) != 0:

            self.offset += 1

            # Unbounded case AND largest element > n
            if (
                self.upper_bound == -1
                and self.queue[-1] + self.offset >= self.lower_bound
            ):
                self.overflow = True
                self.queue.pop()

            # Bounded case AND largest element > m
            elif (
                self.upper_bound != -1
                and self.queue[-1] + self.offset > self.upper_bound
            ):
                self.queue.pop()

            CountingSet._data_collection.max_density_update(len(self.queue))

    @staticmethod
    def union(
        counter_1: "CountingSet", counter_2: "CountingSet"
    ) -> "CountingSet":

        aux = deque()
        while len(counter_1.queue) > 0 or len(counter_2.queue) > 0:

            # Get next element or -1
            max_1 = (
                counter_1.queue[-1] + counter_1.offset
                if len(counter_1.queue) > 0
                else -1
            )
            max_2 = (
                counter_2.queue[-1] + counter_2.offset
                if len(counter_2.queue) > 0
                else -1
            )

            # Apppend max to aux
            if max_1 > max_2:
                aux.appendleft(max_1)
                counter_1.queue.pop()
            elif max_1 < max_2:
                aux.appendleft(max_2)
                counter_2.queue.pop()
            else:
                aux.appendleft(max_1)
                counter_1.queue.pop()
                counter_2.queue.pop()

        new_counter = CountingSet(counter_1.lower_bound, counter_1.upper_bound)
        new_counter.queue = aux
        new_counter.overflow = counter_1.overflow | counter_2.overflow

        CountingSet._data_collection.max_density_update(len(new_counter.queue))

        return new_counter

    def ge_lower_bound(self):
        # Check overflow
        if self.upper_bound == -1:
            return self.overflow
        elif len(self.queue) == 0:
            return False
        else:
            return self.queue[-1] + self.offset >= self.lower_bound

    def le_upper_bound(self):
        # Check overflow
        if self.upper_bound == -1:
            return True
        elif len(self.queue) == 0:
            return False
        else:
            return self.queue[0] + self.offset <= self.upper_bound

    def __str__(self):
        vals = [self.queue[i] + self.offset for i in range(len(self.queue))]

        star = ""
        if self.overflow:
            star = "*"

        return vals.__str__() + star

    def __repr__(self):
        return self.__str__()

    def __bool__(self) -> bool:

        if self.upper_bound == -1:
            return len(self.queue) > 0 or self.overflow
        else:
            return len(self.queue) > 0

    @classmethod
    def data_collection_details(cls):
        if cls._data_collection is None:
            return None
        else:
            data = cls._data_collection.get_data()
            cls._data_collection.reset_data()
            return data
