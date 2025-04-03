"""BoundedSuperConfig"""

from ..position_counting_automaton import PositionCountingAutomaton
from .super_config import SuperConfig


class BoundedSuperConfig(SuperConfig):
    """Class for super-configurations using a set of configurations with
    bounds"""

    def __init__(self, automaton: PositionCountingAutomaton):
        super().__init__(automaton)
        self.automaton = automaton

    def update(self, symbol: str) -> "BoundedSuperConfig":
        super().update(symbol)
        for state, counter_vectors in self._configs.items():
            for counter_vector in counter_vectors:
                flag = False
                for counter_variable, counter_value in counter_vector.items():
                    _, high = self.automaton.counters[counter_variable]
                    if high is not None and counter_value > high:
                        flag = True
                        break
                if flag:
                    self._configs[state].remove(counter_vector)

        empty_states = []
        for state in self._configs:
            if not self._configs[state]:
                empty_states.append(state)

        for state in empty_states:
            del self._configs[state]
        return self
