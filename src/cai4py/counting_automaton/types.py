from cai4py.counting_automaton.counter_map import Action, Guard
from cai4py.custom_counters.counter_base import CounterBase
from typing import Any, NewType

from cai4py.more_collections.ordered_set import OrderedSet

SymbolPredicate = Any
State = NewType("State", int)
Arc = tuple[Guard, Action, State]
Follow = dict[State, OrderedSet[Arc]]
CounterVariable = NewType("CounterVariable", int)
Config = tuple[State, dict[CounterVariable, CounterBase]]
Config = tuple[State, CounterBase | None]
