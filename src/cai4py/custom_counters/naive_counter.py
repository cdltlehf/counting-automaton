from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.data_collection import DataCollection

"""
    Naive counter method.
"""


class NaiveCounter(CounterBase):

    _data_collection = DataCollection()

    def __init__(self, lower_bound: int, upper_bound: int) -> "NaiveCounter":

        # Set parameters
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.list = [1]
        self.overflow = False

        NaiveCounter._data_collection.max_density_update(len(self.list))

    def inc(self):
        """
        Iterate over all elements to increment
        """

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

        NaiveCounter._data_collection.max_density_update(len(self.list))

    @staticmethod
    def union(set1: "NaiveCounter", set2: "NaiveCounter") -> "NaiveCounter":

        setUnion = set1.list

        # Perform union
        for element in set2.list:
            if element not in setUnion:
                setUnion.append(element)

        # Infinity indicator
        if set1.upper_bound == -1:
            setUnion = list(filter(lambda x: x < set1.lower_bound, setUnion))

            # Set parameters
            inst = NaiveCounter(set1.lower_bound, set1.upper_bound)
            inst.list = setUnion
            inst.overflow = set1.overflow | set2.overflow
        else:
            setUnion = list(filter(lambda x: x <= set1.upper_bound, setUnion))

            inst = NaiveCounter(set1.lower_bound, set1.upper_bound)
            inst.list = setUnion
        NaiveCounter._data_collection.max_density_update(len(inst.list))

        return inst

    def ge_lower_bound(self) -> bool:

        # Check overflow
        if self.upper_bound == -1 and self.overflow:
            return True

        # Check elements
        for elem in self.list:
            if elem >= self.lower_bound:
                return True
        return False

    def le_upper_bound(self) -> bool:

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

    def __bool__(self) -> bool:

        if self.upper_bound == -1:
            return len(self.list) > 0 or self.overflow
        else:
            return len(self.list) > 0

    @classmethod
    def data_collection_details(cls):
        if cls._data_collection is None:
            return None
        else:
            data = cls._data_collection.get_data()
            cls._data_collection.reset_data()
            return data
