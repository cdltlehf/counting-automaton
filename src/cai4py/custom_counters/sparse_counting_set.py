from collections import deque

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.data_collection import DataCollection

"""
    Sicheol's sparse counting-set.
"""


class SparseCountingSet(CounterBase):

    _data_collection = DataCollection()

    def __init__(
        self, lower_bound: int, upper_bound: int
    ) -> "SparseCountingSet":

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.queue = deque()
        self.queue.append(1)  # append at right end
        self.offset = 0

        SparseCountingSet._data_collection.max_density_update(len(self.queue))

    def inc(self):
        if len(self.queue) != 0:

            self.offset += 1

            # Bounded case AND largest element > m
            if (
                self.upper_bound != -1
                and self.queue[-1] + self.offset > self.upper_bound
            ):
                self.queue.pop()

            SparseCountingSet._data_collection.max_density_update(
                len(self.queue)
            )

    # Normal union step (same as in counting-set)
    @staticmethod
    def _combine_sets(
        counter_1: "SparseCountingSet", counter_2: "SparseCountingSet"
    ) -> list:
        aux = deque()
        while len(counter_1.queue) > 0 or len(counter_2.queue) > 0:

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

        return aux

    @staticmethod
    def union(
        counter_1: "SparseCountingSet", counter_2: "SparseCountingSet"
    ) -> "SparseCountingSet":

        # combine both (all values are out of range)
        comb = SparseCountingSet._combine_sets(counter_1, counter_2)
        aux = deque()

        if counter_1.upper_bound == -1:  # Unbounded case
            aux.append(comb[-1])

        elif (
            counter_1.upper_bound == counter_1.lower_bound
            or counter_1.upper_bound - 1 == counter_1.lower_bound
        ):  # The {m,m} and {m-1, m} cases cannot gain any decrease in set size
            aux = comb

        elif (
            counter_1.lower_bound == 1 or counter_1.lower_bound == 0
        ):  # The {1,m} and {0,m} case
            aux.append(comb[0])

        else:
            """
            Set reduction step.
            """

            # Get smallest value within range
            milestone = -1
            if len(comb) > 0:
                if comb[-1] <= counter_1.lower_bound:
                    milestone = comb.pop()
                    aux.append(milestone)
                else:
                    while len(comb) > 0 and comb[-1] >= counter_1.lower_bound:
                        milestone = comb.pop()
                    aux.append(milestone)

            delta = counter_1.upper_bound - counter_1.lower_bound
            if len(comb) > 0:
                for element in reversed(comb):

                    # Previous element should be removed and new element added
                    if milestone - element <= delta:
                        if aux[0] != milestone:
                            aux.popleft()
                        aux.appendleft(element)

                    else:  # Gap between element and previous milestone means current value must be milestone
                        milestone = element
                        aux.appendleft(milestone)

        new_counter = SparseCountingSet(
            counter_1.lower_bound, counter_1.upper_bound
        )
        new_counter.queue = aux

        SparseCountingSet._data_collection.max_density_update(
            len(new_counter.queue)
        )

        return new_counter

    def ge_lower_bound(self):
        if len(self.queue) == 0:
            return False
        else:
            return self.queue[-1] + self.offset >= self.lower_bound

    def le_upper_bound(self):
        if len(self.queue) == 0:
            return False
        else:
            return self.queue[0] + self.offset <= self.upper_bound

    def __str__(self):
        vals = [self.queue[i] + self.offset for i in range(len(self.queue))]
        return vals.__str__()

    def __repr__(self):
        return self.__str__()

    def __bool__(self) -> bool:
        return len(self.queue) > 0

    @classmethod
    def data_collection_details(cls):
        if cls._data_collection is None:
            return None
        else:
            data = cls._data_collection.get_data()
            cls._data_collection.reset_data()
            return data
