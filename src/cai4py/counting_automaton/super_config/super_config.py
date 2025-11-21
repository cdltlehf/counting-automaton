"""SuperConfig"""

from collections import defaultdict
from typing import Collection, Iterator, Optional

from cai4py.counting_automaton.super_config.super_config_base import (
    SuperConfigBase,
)
from cai4py.counting_automaton.position_counting_automaton import (
    Config,
    CounterVariable,
)
from cai4py.custom_counters.bit_vector import BitVector
from cai4py.custom_counters.counter_base import CounterBase
from cai4py.custom_counters.counter_type import CounterType
from cai4py.custom_counters.counting_set import CountingSet
from cai4py.custom_counters.sparse_counting_set import SparseCountingSet
from cai4py.custom_counters.naive_counter import NaiveCounter
from cai4py.more_collections import OrderedSet

from ...custom_counters.counter_map import CounterMap
from ...utils.util_logging import setup_debugger
from ..position_counting_automaton import CounterVariable
from ..position_counting_automaton import FINAL_STATE
from ..position_counting_automaton import PositionCountingAutomaton
from ..position_counting_automaton import State

logger = setup_debugger(__name__)

ConfigDictType = defaultdict[State, OrderedSet[CounterMap]]


class SuperConfig(SuperConfigBase, Collection[Config]):
    """Class for super-configurations using a set of configurations"""

    def __init__(
        self, automaton: PositionCountingAutomaton, counter_type: CounterType
    ):
        super().__init__(automaton, counter_type)

        # Internal storage: list of counter dicts per state (dict[CounterVariable, CounterBase]).
        # Using a list avoids the need for hashability (dict is unhashable) while preserving order.
        self._configs: dict[State, list[dict[CounterVariable, CounterBase]]] = (
            defaultdict(list)
        )

        initial_state, counting_state = automaton.get_initial_config()
        # Pre-populate counters dict with inactive placeholders for each automaton counter variable.
        for counter_var in automaton.counters.keys():
            counting_state[counter_var] = counter_type.create_counter(0, 0)
        self._configs[initial_state] = [counting_state]

    @classmethod
    def get_initial(
        cls, automaton: PositionCountingAutomaton, counter_type: CounterType
    ) -> "SuperConfig":
        return cls(automaton, counter_type)

    # Reset all configs of the automaton so multiple strings can be matched without rebuilding.

    def _internal_config_reset(self):
        self._configs = {}
        initial_state, initial_counters = self.automaton.get_initial_config()
        for counter_var in self.automaton.counters.keys():
            initial_counters[counter_var] = self.counter_type.create_counter(
                0, 0
            )
        self._configs[initial_state] = [initial_counters]
        self.counter_type.get_data_collection()

    def __iter__(self) -> Iterator[Config]:
        for state, counters_list in self._configs.items():
            if state == FINAL_STATE:
                continue
            for counters in counters_list:
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
        return sum(len(v) for v in self._configs.values())

    def __contains__(self, config: object) -> bool:
        if not isinstance(config, tuple):
            return False
        state, counter = config
        return state in self._configs and any(
            counter is c for c in self._configs[state]
        )

    # Get superconfigs.

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

    # Match given word using the automaton.

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

            logger.debug("Super Config: %s", super_config)
            last_super_config = super_config

        assert last_super_config is not None

        # Match only occurs if last super config has final state
        return (
            last_super_config.is_final(),
            self.counter_type.get_data_collection(),
        )

    # Use given symbol to traverse edges of automaton and create a new superconfig.

    def update(self, symbol: str) -> "SuperConfig":
        assert len(symbol) == 1

        next_super_config: dict[State, dict[CounterVariable, CounterBase]] = {}
        for config in self:
            # Get a config
            next_configs = self.automaton.get_next_configs(
                config, symbol, self.counter_type
            )

            logger.debug("\t\t\tNext Configs: %s", next_configs)
            for state, counters in next_configs:

                if state in next_super_config:
                    # Merge counters dict by variable; perform value-level unions when needed.
                    merged: dict[CounterVariable, CounterBase] = {}
                    old_counters = next_super_config[state]
                    all_vars = (
                        set(old_counters.keys()) | set(counters.keys())
                        if counters is not None
                        else set(old_counters.keys())
                    )
                    for var in all_vars:
                        c_old = old_counters.get(var)
                        c_new = (
                            counters.get(var) if counters is not None else None
                        )
                        if c_old is not None and c_new is not None:
                            if self.counter_type == CounterType.BIT_VECTOR:
                                assert isinstance(
                                    c_old, BitVector
                                ) and isinstance(c_new, BitVector)
                                merged[var] = BitVector.union(c_old, c_new)
                            elif self.counter_type == CounterType.NAIVE_COUNTER:
                                assert isinstance(
                                    c_old, NaiveCounter
                                ) and isinstance(c_new, NaiveCounter)
                                merged[var] = NaiveCounter.union(c_old, c_new)
                            elif self.counter_type == CounterType.COUNTING_SET:
                                assert isinstance(
                                    c_old, CountingSet
                                ) and isinstance(c_new, CountingSet)
                                merged[var] = CountingSet.union(c_old, c_new)
                            elif (
                                self.counter_type
                                == CounterType.SPARSE_COUNTING_SET
                            ):
                                assert isinstance(
                                    c_old, SparseCountingSet
                                ) and isinstance(c_new, SparseCountingSet)
                                merged[var] = SparseCountingSet.union(
                                    c_old, c_new
                                )
                            else:
                                raise RuntimeError("Unknown Counter Type!")
                        else:
                            merged[var] = c_old if c_old is not None else c_new  # type: ignore
                    next_super_config[state] = merged
                else:
                    next_super_config[state] = (
                        counters if counters is not None else {}
                    )

        if len(next_super_config) == 0:
            self._configs = {}
            return self

        logger.debug("\nMatching the symbol: %s\n%s", symbol, next_super_config)

        self._configs = {
            state: [counters] for state, counters in next_super_config.items()
        }
        return self

    def is_final(self) -> bool:
        return any(map(self.automaton.check_final, self))
