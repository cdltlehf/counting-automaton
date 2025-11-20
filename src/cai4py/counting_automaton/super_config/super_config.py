"""SuperConfig"""

from collections import defaultdict
from typing import Collection, Iterator, Optional

from cai4py.custom_counters.bit_vector import BitVector
from cai4py.custom_counters.counter_base import CounterBase
from cai4py.custom_counters.counter_type import CounterType
from cai4py.custom_counters.counting_set import CountingSet
from cai4py.custom_counters.naive_counter import NaiveCounter

from ...utils.util_logging import setup_debugger

logger = setup_debugger(__name__)

from cai4py.more_collections import OrderedSet

from ..counter_map import CounterMap
from ..position_counting_automaton import Config
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import FINAL_STATE
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State
from cai4py.counting_automaton.super_config.super_config_base import (
    SuperConfigBase,
)

ConfigDictType = defaultdict[State, OrderedSet[CounterMap[CounterVariable]]]


class SuperConfig(SuperConfigBase, Collection[Config]):
    """Class for super-configurations using a set of configurations"""

    def __init__(
        self, automaton: PositionCountingAutomaton, counter_type: CounterType
    ):
        super().__init__(automaton, counter_type)

        self._configs: dict[
            State, OrderedSet[dict[CounterVariable, CounterBase]]
        ] = defaultdict(OrderedSet[dict[CounterVariable, CounterBase]])

        initial_config = automaton.get_initial_config()
        initial_state, counting_state = initial_config

        self._configs.update({initial_state: OrderedSet([counting_state])})

    @classmethod
    def get_initial(
        cls, automaton: PositionCountingAutomaton, counter_type: CounterType
    ) -> "SuperConfig":
        return cls(automaton, counter_type)

    """
        Resets all configs of the automaton. Allows multiple strings to be run on an automaton
        created for a single regex without having to rebuild the automaton.
    """

    def _internal_config_reset(self):
        self._configs = dict()

        initial_config = self.automaton.get_initial_config()
        initial_state, initial_counters = initial_config

        self._configs.update({initial_state: OrderedSet([initial_counters])})

        self.counter_type.get_data_collection()

    def __iter__(self) -> Iterator[Config]:
        for state, set_of_counter_dicts in self._configs.items():
            if state == FINAL_STATE:
                continue
            for counters in set_of_counter_dicts:
                yield (state, counters)

    def to_json(self) -> list[tuple[int, list[Optional[int]]]]:
        return [(state, list(counters)) for (state, counters) in self]

    def __str__(self):
        s = "["
        for state, counter in self:
            s += str(state) + ": " + str(counter) + ", "

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

    @classmethod
    def get_computation(
        cls,
        automaton: PositionCountingAutomaton,
        w: str,
        counter_type: CounterType,
    ) -> Iterator["SuperConfig"]:
        super_config = cls.get_initial(automaton, counter_type)
        yield super_config

        for symbol in w:
            super_config = super_config.update(symbol)
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
        for super_config in self.__class__.get_computation(
            self.automaton, w, self.counter_type
        ):
            if len(self._configs) <= 0:
                return (False, self.counter_type.get_data_collection())

            logger.debug(f"Super Config: {super_config}")
            last_super_config = super_config

        assert last_super_config is not None

        # Match only occurs if last super config has final state
        return (
            last_super_config.is_final(),
            self.counter_type.get_data_collection(),
        )

    """
        Use given symbol to traverse edges of automaton 
        to create a superconfig.
    """

    def update(self, symbol: str) -> "SuperConfig":
        assert len(symbol) == 1

        next_super_config = dict()
        for config in self:
            # Get a config
            next_configs = self.automaton.get_next_configs(
                config, symbol, self.counter_type
            )

            logger.debug(f"\t\t\tNext Configs: {next_configs}")
            for state, counters in next_configs:

                if state in next_super_config:
                    old_counters = next_super_config[state]

                    # Perform union of counters
                    if counters is not None:
                        match self.counter_type:
                            case CounterType.BIT_VECTOR:
                                assert isinstance(counters, BitVector)
                                assert isinstance(old_counters, BitVector)
                                next_super_config[state] = BitVector.union(
                                    old_counters, counters
                                )
                            case CounterType.NAIVE_COUNTER:
                                assert isinstance(counters, NaiveCounter)
                                assert isinstance(old_counters, NaiveCounter)
                                next_super_config[state] = NaiveCounter.union(
                                    old_counters, counters
                                )
                            case CounterType.COUNTING_SET:
                                assert isinstance(counters, CountingSet)
                                assert isinstance(old_counters, CountingSet)
                                next_super_config[state] = CountingSet.union(
                                    old_counters, counters
                                )
                            case _:
                                raise RuntimeError("Unknown Counter Type!")

                else:
                    next_super_config[state] = counters

        if len(next_super_config) <= 0:
            self._configs = next_super_config
            return self

        logger.debug(f"\nMatching the symbol: {symbol}\n{next_super_config}")

        self._configs = next_super_config
        return self

    def is_final(self) -> bool:
        return any(map(self.automaton.check_final, self))
