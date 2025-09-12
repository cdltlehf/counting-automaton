"""SuperConfig"""

from collections import defaultdict as dd
from typing import Collection, Iterator, Optional

from cai4py.custom_counters.bit_vector import BitVector
from cai4py.custom_counters.counter_type import CounterType

from ...utils.util_logging import setup_debugger
logger = setup_debugger(__name__)

from cai4py.more_collections import OrderedSet

from ..counter_vector import CounterVector
from ..position_counting_automaton import Config
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import FINAL_STATE
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State

ConfigDictType = dd[State, OrderedSet[CounterVector[CounterVariable]]]

class SuperConfig(Collection[Config]):
    """Class for super-configurations using a set of configurations"""

    def __init__(self, automaton: PositionCountingAutomaton, counter_type: CounterType):
        self.automaton = automaton

        self.counter_type = counter_type

        self._configs = dict()

        initial_config = automaton.get_initial_config()
        initial_state, initial_counter = initial_config

        self._configs.update({initial_state : initial_counter})

    @classmethod
    def get_initial(cls, automaton: PositionCountingAutomaton, counter_type: CounterType) -> "SuperConfig":
        return cls(automaton, counter_type)

    """
        Resets all configs of the automaton. Allows multiple string to be run on an automaton
        created for a single regex without having to rebuild the automaton.
    """
    def _internal_config_reset(self):
        self._configs = dict()

        initial_config = self.automaton.get_initial_config()
        initial_state, initial_counter = initial_config

        self._configs.update({initial_state : initial_counter})

    def __iter__(self) -> Iterator[Config]:
        for state, counter in self._configs.items():
            if state == FINAL_STATE:
                continue
            yield (state, counter)

    def to_json(self) -> list[tuple[int, list[Optional[int]]]]:
        return [
            (state, counter.to_list())
            for (state, counter) in self
        ]
    
    def __str__(self):
        s = "["
        for (state, counter) in self:
            s += str(state )+ ": " + str(counter) + ", "

        s = s.strip(", ")
        s += "]"
        return s

    def __len__(self) -> int:
        return sum(map(len, self._configs.values()))

    def __contains__(self, config: object) -> bool:
        if not isinstance(config, tuple):
            return False
        state, counter = config
        return counter in self._configs[state]

    """
        Get superconfigs.
    """
    def get_computation(self, w: str) -> Iterator["SuperConfig"]:
    
        yield self.automaton.get_initial_config()

        for symbol in w:
            super_config = self.update(symbol)
            yield super_config

    """
        Match given word using the automaton.
    """
    def match(self, w: str):

        self._internal_config_reset()

        if len(w) == 0:
            return (self.is_final(), self.counter_type.get_data_collection())
        last_super_config = None

        # Super config: (state, counter)
        for super_config in self.get_computation(w):
            logger.debug(f"Super Config: {super_config}")
            last_super_config = super_config

        assert last_super_config is not None

        # Match only occurs if last super config has final state
        return (last_super_config.is_final(), self.counter_type.get_data_collection())

    """
        Use given symbol to traverse edges of automaton 
        to create a superconfig.
    """
    def update(self, symbol: str) -> "SuperConfig":
        assert len(symbol) == 1

        union = self.counter_type.get_union()

        next_super_config = dict()
        for config in self:
            # Get a config
            next_configs = self.automaton.get_next_configs(config, symbol, self.counter_type)

            logger.debug(f"\t\t\tNext Configs: {next_configs}")
            for state, counter in next_configs:
                
                if state in next_super_config:
                    old_counter = next_super_config[state]
                    
                    # Perform union of counters
                    if counter is not None:
                        next_super_config[state] = union(old_counter, counter)

                else:
                    next_super_config[state] = counter

        logger.debug(f"\nMatching the symbol: {symbol}\n{next_super_config}")     

        self._configs = next_super_config
        return self

    def is_final(self) -> bool:
        return any(map(self.automaton.check_final, self))