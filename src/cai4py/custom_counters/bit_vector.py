import numpy as np

from cai4py.custom_counters.counter_base import CounterBase
from cai4py.utils.data_collection import DataCollection

"""
    Bit vector counter method
"""


class BitVector(CounterBase):

    _data_collection = DataCollection()

    def __init__(self, lower_bound: int, upper_bound: int) -> "BitVector":

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if (
            self.upper_bound != -1 and self.upper_bound <= 64
        ):  # Hardware implementation (bounded)

            self.bits = np.uint64(1) << np.uint64(63)
            num_bits = (
                self.upper_bound - max(self.lower_bound, 1) + 1
            )  # Have to handle the case {0,64}
            self.mask = np.uint64(2**num_bits - 1) << np.uint64(
                64 - upper_bound
            )

        elif (
            self.upper_bound == -1 and self.lower_bound <= 64
        ):  # Hardware implementation (unbounded)

            self.bits = np.uint64(1) << np.uint64(63)
            self.mask = np.uint64(1) << np.uint64(64 - lower_bound + 1)
            self.overflow = False

        else:  # Software implementation

            """
            NOTE: the regex parser does the following conversions
                - {1,} -> +
                - {0,} -> *
            If these conversions are ever removed then the bit vector (and other counters)
            will break for those cases. So like... don't remove it lol.
            """

            if upper_bound == -1:
                self.vector = np.zeros(self.lower_bound - 1, dtype=np.uint8)
                self.overflow = False
            else:
                self.vector = np.zeros(self.upper_bound, dtype=np.uint8)

                # Lowest in range is only initially valid if lower bound is 0 or 1
                if self.lower_bound == 1 or self.lower_bound == 0:
                    self.lowest_in_range = 1
                else:
                    self.lowest_in_range = -1

            self.vector[0] = 1
            self.size = 1  # Vector has value 0 at all positions i >= size
            self.start = 0
            self.density = 1

            BitVector._data_collection.max_density_update(self.density)

    def inc(self):

        if (
            self.upper_bound != -1 and self.upper_bound <= 64
        ):  # Hardware implementation (bounded)
            self.bits = self.bits >> np.uint64(1)

        elif (
            self.upper_bound == -1 and self.lower_bound <= 64
        ):  # Hardware implementation (unbounded)

            if self.mask & self.bits > 0:
                self.overflow = True
            self.bits = self.bits >> np.uint64(1)

        else:  # Software implementation

            # Will increment send the value over the lower bound (unbounded case)
            if (
                self.upper_bound == -1
                and self.vector[(self.start - 1) % len(self.vector)] == 1
            ):
                self.overflow = True

            # Move pointer left
            self.start = (self.start - 1) % len(self.vector)
            if self.vector[self.start] == 1:
                self.density -= 1

            self.vector[self.start] = 0

            if self.size < len(self.vector):
                self.size += 1

            if self.upper_bound != -1:

                # Set minimum value within range
                if (
                    self.lowest_in_range >= self.lower_bound
                    and self.lowest_in_range < self.upper_bound
                ):
                    self.lowest_in_range += 1
                elif (
                    self.vector[
                        (self.start + self.lower_bound - 1) % self.upper_bound
                    ]
                    == 1
                ):
                    self.lowest_in_range = self.lower_bound
                else:
                    self.lowest_in_range = -1

    @staticmethod
    def union(counter_1: "BitVector", counter_2: "BitVector") -> "BitVector":

        # Normal bit-wise OR for hardware bit vectors
        if (
            counter_1.upper_bound != -1 and counter_1.upper_bound <= 64
        ):  # Hardware implementation (bounded)
            new_counter = BitVector(
                counter_1.lower_bound, counter_1.upper_bound
            )
            new_counter.bits = counter_1.bits | counter_2.bits

        elif (
            counter_1.upper_bound == -1 and counter_1.lower_bound <= 64
        ):  # Hardware implementation (unbounded)

            new_counter = BitVector(
                counter_1.lower_bound, counter_1.upper_bound
            )
            new_counter.bits = counter_1.bits | counter_2.bits
            new_counter.overflow = counter_1.overflow | counter_2.overflow

        else:
            assert (
                counter_1.upper_bound == counter_2.upper_bound
            ), "Vectors not of equal length!"

            # Quick return option for union
            if counter_1.size == 0:
                return counter_2
            elif counter_2.size == 0:
                return counter_1

            # Data collection
            density = 0

            # Perform union
            aux = []
            for i in range(len(counter_1.vector)):
                pos_1 = (counter_1.start + i) % len(counter_1.vector)
                pos_2 = (counter_2.start + i) % len(counter_1.vector)

                if pos_1 >= counter_1.size:
                    aux_1 = 0
                else:
                    aux_1 = counter_1.vector[pos_1]

                if pos_2 >= counter_2.size:
                    aux_2 = 0
                else:
                    aux_2 = counter_2.vector[pos_2]

                # Data collection
                if aux_1 | aux_2:
                    density += 1

                aux.append(aux_1 | aux_2)

            # Set new parameters
            new_counter = BitVector(
                counter_1.lower_bound, counter_1.upper_bound
            )
            new_counter.vector = aux
            new_counter.size = max(counter_1.size, counter_2.size)
            new_counter.density = density

            BitVector._data_collection.max_density_update(new_counter.density)

            if counter_1.upper_bound == -1:
                new_counter.overflow = counter_1.overflow | counter_2.overflow

            elif (
                counter_1.lowest_in_range == -1
                or counter_2.lowest_in_range == -1
            ):
                new_counter.lowest_in_range = max(
                    counter_1.lowest_in_range, counter_2.lowest_in_range
                )

            else:
                new_counter.lowest_in_range = min(
                    counter_1.lowest_in_range, counter_2.lowest_in_range
                )

        return new_counter

    def ge_lower_bound(self) -> bool:

        if (
            self.upper_bound != -1 and self.upper_bound <= 64
        ):  # Hardware implementation (bounded)
            return self.bits & self.mask > 0  # Check if the bits are present

        elif (
            self.upper_bound == -1 and self.lower_bound <= 64
        ):  # Hardware implementation (unbounded)
            return self.overflow

        else:  # Software implementation
            if self.upper_bound == -1:  # Unbounded
                return self.overflow
            else:
                return self.lowest_in_range != -1  # Check lowest in range

    def le_upper_bound(self) -> bool:

        if (
            self.upper_bound != -1 and self.upper_bound <= 64
        ):  # Hardware implementation (bounded)
            return (
                self.bits & self.mask > 0
            )  # Technically incorrect but will never affect outcome

        elif (
            self.upper_bound == -1 and self.lower_bound <= 64
        ):  # Hardware implementation (unbounded)
            return True  # Doesn't matter: will never be called

        else:  # Software implementation (bounded)
            if (
                self.upper_bound == -1
            ):  # Unbounded - doesn't matter: will never be called
                return True
            else:
                return (
                    self.lowest_in_range != -1
                )  # Technically incorrect but will never affect outcome

    def __str__(self):

        if (
            self.upper_bound != -1 and self.upper_bound <= 64
        ):  # Hardware implementation (bounded)
            vals = []
            for i in range(self.upper_bound):
                if self.bits & (np.uint64(1) << np.uint64(63 - i)) > 0:
                    vals.append(i + 1)

            return vals.__str__()

        elif (
            self.upper_bound == -1 and self.lower_bound <= 64
        ):  # Hardware implementation (unbounded)
            vals = []
            for i in range(self.lower_bound):
                if self.bits & (np.uint64(1) << np.uint64(63 - i)) > 0:
                    vals.append(i + 1)

            star = ""
            if self.overflow:
                star = "*"

            return vals.__str__() + star

        else:
            vals = []
            for i in range(len(self.vector)):
                pos = (self.start + i) % len(self.vector)

                if self.vector[pos]:
                    vals.append(i + 1)

            star = ""
            if self.upper_bound == -1 and self.overflow:
                star = "*"

            return vals.__str__() + star

    def __repr__(self):
        return self.__str__()

    def __bool__(self) -> bool:

        if (
            self.upper_bound != -1 and self.upper_bound <= 64
        ):  # Hardware implementation (bounded)
            if self.bits > 0:
                return True
            else:
                return False

        elif (
            self.upper_bound == -1 and self.lower_bound <= 64
        ):  # Hardware implementation (unbounded)
            if self.bits > 0:
                return True
            else:
                return self.overflow

        elif self.upper_bound == -1:  # Software implementation (unbounded)
            return self.density > 0 or self.overflow

        else:  # Software implementation (bounded)
            return self.density > 0

    @classmethod
    def data_collection_details(cls):
        if cls._data_collection is None:
            return None
        else:
            data = cls._data_collection.get_data()
            cls._data_collection.reset_data()
            return data
