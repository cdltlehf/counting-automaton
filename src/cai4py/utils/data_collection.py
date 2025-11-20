from enum import StrEnum

class Operation(StrEnum):
    INIT = "INIT"
    INC = "INC"
    UNION = "UNION"
    GE = "GE"
    LE = "LE"

"""
    Allows us to collect different data measurements from within the counters. Use
    to contain a lot more but we scaled it back.
"""
class DataCollection():

    def __init__(self):
        """
            Density of the counter i.e. the number of values within the counter
        """
        self.max_density = 0


    def max_density_update(self, density: int):
        self.max_density = max(self.max_density, density)

    def get_data(self):
        return {"Maximum Density": self.max_density}

    def reset_data(self):
        self.max_density = 0
