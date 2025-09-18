import sys

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.data_collection import DataCollection
from cai4py.utils.data_collection import Operation


class LazyCountingSet(CounterBase):

    _data_collection = DataCollection(
        {
            Operation.INIT: 1,
            Operation.INC: 2,
            Operation.UNION: 2,
            Operation.GE: 2,
            Operation.LE: 2,
        }
    )

    def __init__(self, lower_bound: int, upper_bound: int) -> None:

        LazyCountingSet._data_collection.full_update(Operation.INIT)
        LazyCountingSet._data_collection.half_update(1, Operation.INIT)

        # Set parameters
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.list = [1]
        self.overflow = False

        # Default (~<60K) Python int object is 28 bytes, however, np.array will be slower
        LazyCountingSet._data_collection.max_memory_update(
            sys.getsizeof(self.list)
        )

    def inc(self):

        LazyCountingSet._data_collection.full_update(Operation.INC)
        LazyCountingSet._data_collection.half_update(
            len(self.list) // 2, Operation.INC
        )  # Account for lambda function

        # Infinity indicator
        if self.upper_bound == -1:

            # Increment and set overflow
            for i in range(0, len(self.list)):
                if self.list[i] + 1 >= self.lower_bound:
                    self.overflow = True

                self.list[i] += 1

            self.list = list(filter(lambda x: x < self.lower_bound, self.list))

        else:

            # Increment
            for i in range(0, len(self.list)):
                self.list[i] += 1

            self.list = list(filter(lambda x: x <= self.upper_bound, self.list))

    @staticmethod
    def union(
        set1: "LazyCountingSet", set2: "LazyCountingSet"
    ) -> "LazyCountingSet":

        LazyCountingSet._data_collection.full_update(Operation.UNION)
        LazyCountingSet._data_collection.half_update(
            3 * len(set1.list), Operation.UNION
        )

        setUnion = set1.list

        # Perform union
        for element in set2.list:
            if element not in setUnion:
                setUnion.append(element)

        # Infinity indicator
        if set1.upper_bound == -1:
            LazyCountingSet._data_collection.half_update(
                2 * len(setUnion), Operation.UNION
            )
            setUnion = list(filter(lambda x: x < set1.lower_bound, setUnion))

            # Set parameters
            inst = LazyCountingSet(set1.lower_bound, set1.upper_bound)
            inst.list = setUnion
            inst.overflow = set1.overflow | set2.overflow
        else:
            LazyCountingSet._data_collection.half_update(
                2 * len(setUnion), Operation.UNION
            )
            setUnion = list(filter(lambda x: x <= set1.upper_bound, setUnion))

            inst = LazyCountingSet(set1.lower_bound, set1.upper_bound)
            inst.list = setUnion

        LazyCountingSet._data_collection.max_memory_update(
            sys.getsizeof(inst.list)
        )

        return inst

    def ge_lower_bound(self) -> bool:

        LazyCountingSet._data_collection.full_update(Operation.GE)
        LazyCountingSet._data_collection.half_update(
            len(self.list), Operation.GE
        )

        # Check overflow
        if self.upper_bound == -1 and self.overflow:
            return True

        # Check elements
        for elem in self.list:

            if elem >= self.lower_bound:
                return True
        return False

    def le_upper_bound(self) -> bool:

        LazyCountingSet._data_collection.full_update(Operation.LE)
        LazyCountingSet._data_collection.half_update(
            len(self.list), Operation.LE
        )

        # Check elements
        for elem in self.list:
            if elem <= self.upper_bound:
                return True
        return False

    def __str__(self):
        s = "["
        for i in range(0, len(self.list)):
            element = self.list[i]

            s += str(element)

            if i != len(self.list) - 1:
                s += ", "

        star = ""
        if self.overflow:
            star = "*"

        return s + "]" + star

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
