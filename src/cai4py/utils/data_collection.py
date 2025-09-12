from copy import deepcopy
from enum import StrEnum

class Operation(StrEnum):
    INIT = "INIT"
    INC = "INC"
    UNION = "UNION"
    GE = "GE"
    LE = "LE"

class DataCollection():

    def __init__(self, weights: dict):
        """
            The half abstraction completely abstracts away the actual computations and assigns a weight
            to each operation, which depends on the time complexity of the operation for the particular counter
            implementation. For example, a constant time operation has weight 1 and a linear time operation a
            weight of 2. 
        """
        self.full_abstraction = {}
        for operation in Operation:
            self.full_abstraction[operation] = (0,0)

        """
            The full abstraction takes into account the actual computation required for 
            performing a match. For example, the init of the counter may take linear time in 
            the quantifier upper bound. Thus, if the quantifier is {n,m} then the full abstraction
            will contain the value m to indicate that the counter was initialised once with an upper bound
            of m.
        """

        self.half_abstraction = {}
        for operation in Operation:
            self.half_abstraction[operation] = 0

        for k, _ in self.full_abstraction.items():
            if k in weights:
                self.full_abstraction[k] = (weights[k], 0)
            else:
                raise f"Dictionary supplied does not contain the key: {k}"
            

        """
            Want a way to count the number of operations that occured.
        """
        self.operations_count = {}
        for operation in Operation:
            self.operations_count[operation] = 0

        """
            'Abstract' maximum memory that a counter requires
        """
        self.max_memory = 0


    def full_update(self, operation):
        weight, count = self.full_abstraction[operation]
        self.full_abstraction[operation] = (weight, count + 1)

    def half_update(self, size: int, operation):
        self.half_abstraction[operation] += size

    def count_update(self, operation):
        self.operations_count[operation] += 1

    def max_memory_update(self, memory: int):
        self.max_memory = max(self.max_memory, memory)

    def get_data(self):
        return {
            "Full Abstraction": deepcopy(self.full_abstraction),
            "Half Abstraction": deepcopy(self.half_abstraction),
            "Operations Count": deepcopy(self.operations_count),
            "Maximum Memory": self.max_memory
            }

    def reset_data(self):
        self.max_memory = 0

        for operation in Operation:
            w, _ = self.full_abstraction[operation]
            self.full_abstraction[operation] = (w, 0)

            self.half_abstraction[operation] = 0

            self.operations_count[operation] = 0
