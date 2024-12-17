"""Position counter automaton."""

from collections import defaultdict as dd
from copy import copy
from enum import StrEnum
from functools import reduce
from itertools import chain
import logging
import re._compiler as compiler  # type: ignore[import-untyped]
from re._constants import _NamedIntConstant as NamedIntConstant  # type: ignore[import-untyped]
from re._constants import AT
from re._constants import ATOMIC_GROUP
from re._constants import BRANCH
from re._constants import MAXREPEAT
from re._constants import SUBPATTERN
from re._parser import SubPattern  # type: ignore[import-untyped]
from typing import Any, Hashable, Iterable, Iterator, Mapping, Optional

from .more_collections import OrderedSet
from .parser_tools import fold
from .parser_tools import MAX_REPEAT
from .parser_tools import MIN_REPEAT
from .parser_tools import parse
from .parser_tools import PREDICATE_OPCODES


class CounterOperationComponent(StrEnum):
    INACTIVATE = " = None"
    INCREMENT = "++"
    RESET_OR_ACTIVATE = " = 1"


class CounterPredicateType(StrEnum):
    NOT_LESS_THAN = " >= "
    NOT_GREATER_THAN = " <= "
    LESS_THAN = " < "


CounterVariable = int
CounterValue = int

CounterPredicate = tuple[CounterPredicateType, int]
# TODO: Redefine guard and action
# Guard = dd[CounterVariable, CounterPredicate]
# Action = dd[CounterVariable, CounterOperationComponent]
Guard = tuple[tuple[CounterVariable, CounterPredicate], ...]
Action = tuple[tuple[CounterVariable, CounterOperationComponent], ...]


class CounterVector(Mapping[CounterVariable, CounterValue], Hashable):
    """Counter vector."""

    def __init__(self, counters: dict[CounterVariable, int]) -> None:
        """Initialize a counter vector."""
        self._values: tuple[Optional[CounterValue], ...]
        self._values = tuple([None] * len(counters))
        self._upper_bound = counters

    def upper_bound(self, counter_variable: CounterVariable) -> int:
        return self._upper_bound[counter_variable]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CounterVector):
            return NotImplemented
        return self._values == other._values

    def __str__(self) -> str:
        return str(self._values)

    def __iter__(self) -> Iterator[CounterVariable]:
        return iter(self._upper_bound)

    def __setitem__(self, key: CounterVariable, value: CounterValue) -> None:
        self._values = tuple(
            value if i == key else self._values[i]
            for i in range(len(self._values))
        )

    def __getitem__(self, key: CounterVariable) -> CounterValue:
        value = self._values[key]
        if value is None:
            raise KeyError(key)
        return value

    def __delitem__(self, key: CounterVariable) -> None:
        self._values = tuple(
            None if i == key else self._values[i]
            for i in range(len(self._values))
        )

    def __len__(self) -> int:
        return len(self._values)

    def __copy__(self) -> "CounterVector":
        new = CounterVector(self._upper_bound)
        new._values = self._values
        return new

    def __hash__(self) -> int:
        return hash(self._values)

    def evalutate_guard(self, guard: Guard) -> bool:
        def eval_predicate(
            predicate: CounterPredicate,
            counter_value: CounterValue
        ) -> bool:
            (predicate_type, value) = predicate
            if predicate_type is CounterPredicateType.NOT_LESS_THAN:
                return counter_value >= value
            elif predicate_type is CounterPredicateType.NOT_GREATER_THAN:
                return counter_value <= value
            elif predicate_type is CounterPredicateType.LESS_THAN:
                return counter_value < value
            else:
                assert False, predicate_type

        return all(
            eval_predicate(predicate, self[counter_variable])
            for counter_variable, predicate in guard
        )

    def clone(self) -> "CounterVector":
        return copy(self)

    def apply_action(self, action: Action) -> "CounterVector":
        for counter_variable, operation in action:
            if operation is CounterOperationComponent.INCREMENT:
                assert counter_variable in self, counter_variable
                counter_value = self[counter_variable]
                upper_bound = self.upper_bound(counter_variable)
                self[counter_variable] = min(counter_value + 1, upper_bound)
            elif operation is CounterOperationComponent.RESET_OR_ACTIVATE:
                assert counter_variable not in self, counter_variable
                self[counter_variable] = 1
            elif operation is CounterOperationComponent.INACTIVATE:
                assert counter_variable in self, counter_variable
                del self[counter_variable]
            else:
                assert False, operation
        return self

State = int
SymbolPredicate = Any
Arc = tuple[Guard, Action, State]
Follow = dd[State, OrderedSet[Arc]]
Config = tuple[State, CounterVector]
SuperConfig = OrderedSet[Config]

INITIAL_STATE: State = 0
FINAL_STATE: State = -1

class PositionCounterAutomaton:
    """Position counter automaton."""

    def __init__(
        self,
        states: dict[State, SymbolPredicate],
        counters: dict[CounterVariable, int],
        follow: Follow,
    ) -> None:
        self.states = states
        self.counters = counters
        self.follow = follow

    @classmethod
    def create(cls, pattern: str) -> "PositionCounterAutomaton":
        tree = parse(pattern)
        logging.debug(tree)
        callback_object = _PositionConstructionCallback()

        def callback(
            x: Optional[tuple[NamedIntConstant, Any]],
            ys: Iterable[PositionCounterAutomaton],
        ) -> PositionCounterAutomaton:
            automaton = callback_object(x, ys)
            logging.debug(x)
            logging.debug(automaton)
            logging.debug("\n")
            return automaton

        return fold(callback, tree)

    def eval_state(self, state: State, symbol: str) -> bool:
        assert len(symbol) == 1
        if isinstance(self.states[state], str):
            return bool(self.states[state] == symbol)
        elif isinstance(self.states[state], SubPattern):
            compiled = compiler.compile(self.states[state])
            return compiled.fullmatch(symbol) is not None
        assert False, type(self.states[state])

    def __str__(self) -> str:

        def guard_to_str(guard: Guard) -> str:
            return ", ".join(
                f"c[{counter}]{predicate_type}{operand}"
                for counter, (predicate_type, operand) in guard
            )

        def action_to_str(action: Action) -> str:
            return ", ".join(
                f"c[{counter}]{operation}" for counter, operation in action
            )

        def arc_to_str(arc: Arc) -> str:
            guard, action, adjacent_state = arc
            return (
                f"--{{{guard_to_str(guard)}; {action_to_str(action)}}}"
                + f"-> {adjacent_state}"
            )

        follow_string = (
            "\n".join(
                "\n".join(
                    f"- {state} {arc_to_str(arc)}"
                    for arc in follow
                )
                for state, follow in self.follow.items()
            )
        )
        return "\n".join(
            [
                f"states: {self.states}",
                f"follow:\n{follow_string}",
                f"counter: {self.counters}",
            ]
        )

    def check_final(self, config: Config) -> bool:
        cur_state, counter_vector = config
        logging.debug(self.config_to_str(config))
        for guard, _, adjacent_state in self.follow[cur_state]:
            if adjacent_state is not FINAL_STATE:
                continue
            if not counter_vector.evalutate_guard(guard):
                continue
            return True
        return False

    def is_nullable(self) -> bool:
        for _, _, adjacent_state in self.follow[INITIAL_STATE]:
            if adjacent_state is FINAL_STATE:
                return True
        return False

    def get_next_config(
        self,
        config: Config,
        symbol: str
    ) -> SuperConfig:
        cur_state, counter_vector = config
        next_configs: SuperConfig = OrderedSet()

        for guard, action, adjacent_state in self.follow[cur_state]:
            if not counter_vector.evalutate_guard(guard):
                continue

            if adjacent_state is not FINAL_STATE:
                if not self.eval_state(adjacent_state, symbol):
                    continue

            # TODO: Optimization. If there is only one possible action,
            # we can avoid cloning the counter vector.
            next_counter_vector = counter_vector.clone()
            next_counter_vector.apply_action(action)
            next_configs.append((adjacent_state, next_counter_vector))
        return next_configs

    def get_next_super_config(
        self,
        super_config: SuperConfig,
        symbol: str
    ) -> SuperConfig:
        assert len(symbol) == 1

        next_configs = OrderedSet(
            chain.from_iterable(
                self.get_next_config(config, symbol)
                for config in super_config
            )
        )
        return next_configs

    @staticmethod
    def config_to_str(config: Config) -> str:
        state, counter_vector = config
        return f"({state}, {counter_vector})"

    @staticmethod
    def super_config_to_str(super_config: SuperConfig) -> str:
        return "{" + ", ".join(
            map(PositionCounterAutomaton.config_to_str, super_config)
        ) + "}"

    def iterate_super_configs(self, w: str) -> Iterable[SuperConfig]:
        initial_config: Config = (INITIAL_STATE, CounterVector(self.counters))
        yield OrderedSet({initial_config})

        current_super_config = OrderedSet([initial_config])
        for index, symbol in enumerate(w):
            logging.debug("%s", w)
            logging.debug("%s", " " * index + "^" + symbol)
            logging.debug(self.super_config_to_str(current_super_config))
            current_super_config = self.get_next_super_config(
                current_super_config,
                symbol
            )
            yield current_super_config

    def match(self, w: str) -> bool:
        logging.debug("Matching")

        if not w:
            return self.is_nullable()

        super_config: SuperConfig
        for super_config in self.iterate_super_configs(w):
            if not super_config:
                return False

        logging.debug("%s", w)
        logging.debug("%s", " " * len(w) + "^")
        logging.debug(self.super_config_to_str(super_config))
        logging.debug("Matching end")
        return any(self.check_final(config) for config in super_config)

    def __call__(self, w: str) -> bool:
        return self.match(w)

    def backtrack(self, w: str) -> bool:
        logging.debug("Backtrack matching")

        def _backtrack(w: str, config: Config, index: int) -> bool:
            logging.debug("%s", w)
            logging.debug("%s", " " * index + "^")
            logging.debug("%d %s", index, str(config))
            if len(w) == index:
                return self.check_final(config)

            next_configs = self.get_next_config(config, w[index])
            return any(
                _backtrack(w, config, index + 1) for config in next_configs
            )

        initial_counter = CounterVector(self.counters)
        return _backtrack(w, (0, initial_counter), 0)


class _PositionConstructionCallback:
    """Callback function for constructing position automata.
    It is a stateful object."""

    def __init__(self) -> None:
        self.state = 0
        self.counter = -1

    @staticmethod
    def get_final_arcs(follow: Follow) -> list[tuple[State, Arc]]:
        final_arcs: list[tuple[State, Arc]] = []
        for state, arcs in follow.items():
            for arc in arcs:
                _, _, adjacent_state = arc
                if adjacent_state is FINAL_STATE:
                    final_arcs.append((state, arc))
        return final_arcs

    def call_empty(self) -> PositionCounterAutomaton:
        follow: Follow = dd(OrderedSet)
        follow[INITIAL_STATE].append((tuple(), tuple(), FINAL_STATE))
        return PositionCounterAutomaton({}, {}, follow)

    def call_predicate(self, x: tuple[str, Any]) -> PositionCounterAutomaton:
        _, operand = x
        self.state += 1

        follow: Follow = dd(OrderedSet)
        follow[INITIAL_STATE].append((tuple(), tuple(), self.state))
        follow[self.state].append((tuple(), tuple(), FINAL_STATE))

        return PositionCounterAutomaton({self.state: operand}, {}, follow)

    def call_at(self, x: tuple[str, Any]) -> PositionCounterAutomaton:
        raise NotImplementedError("Anchor is not supported")

    def call_catenation(
        self, y1: PositionCounterAutomaton, y2: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        assert y1.states.keys().isdisjoint(y2.states.keys())

        for final_state, final_arc in self.get_final_arcs(y1.follow):
            guard, action, _ = final_arc

            arcs: list[Arc] = []
            for initial_arc in y2.follow[INITIAL_STATE]:
                initial_guard, initial_action, initial_state = initial_arc
                new_guard = (*guard, *initial_guard)
                new_action = (*action, *initial_action)
                arcs.append((new_guard, new_action, initial_state))
            y1.follow[final_state].substitute(final_arc, arcs)

        for state in y2.states:
            if state == INITIAL_STATE:
                continue
            assert state not in y1.follow, str(y1.follow.keys())
            y1.follow[state] = y2.follow[state]

        # NOTE: we can optimize this, but it is not necessary.
        y1.states.update(y2.states)
        y1.counters.update(y2.counters)
        return y1

    def call_union(
        self, y1: PositionCounterAutomaton, y2: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        assert y1.states.keys().isdisjoint(y2.states.keys())

        y1.follow[INITIAL_STATE].append_iterable(y2.follow[INITIAL_STATE])
        for state in y2.states:
            assert state not in y1.follow
            y1.follow[state] = y2.follow[state]

        # NOTE: we can optimize this, but it is not necessary.
        y1.states.update(y2.states)
        y1.counters.update(y2.counters)
        return y1

    def call_star(
        self, y: PositionCounterAutomaton, lazy: bool
    ) -> PositionCounterAutomaton:
        y = self.call_plus(y, lazy)
        y = self.call_question(y, lazy)
        return y

    def call_plus(
        self, y: PositionCounterAutomaton, lazy: bool
    ) -> PositionCounterAutomaton:
        for final_state, final_arc in self.get_final_arcs(y.follow):
            guard, action, _ = final_arc

            arcs: list[Arc] = []
            for initial_arc in y.follow[INITIAL_STATE]:
                initial_guard, initial_action, initial_state = initial_arc
                new_guard = (*guard, *initial_guard)
                new_action = (*action, *initial_action)
                arcs.append((new_guard, new_action, initial_state))
            y.follow[final_state].substitute(final_arc, arcs)
            if lazy:
                y.follow[final_state].prepend(final_arc)
            else:
                y.follow[final_state].append(final_arc)
        return y

    def call_question(
        self, y: PositionCounterAutomaton, lazy: bool
    ) -> PositionCounterAutomaton:
        if lazy:
            y.follow[INITIAL_STATE].prepend((tuple(), tuple(), FINAL_STATE))
        else:
            y.follow[INITIAL_STATE].append((tuple(), tuple(), FINAL_STATE))
        return y

    def call_repeat(
        self,
        y: PositionCounterAutomaton,
        lower_bound: int,
        upper_bound: Optional[int],
        lazy: bool,
    ) -> PositionCounterAutomaton:
        self.counter += 1

        final_arcs = self.get_final_arcs(y.follow)
        initial_arcs = y.follow[INITIAL_STATE]

        for last_state, final_arc in final_arcs:
            guard, action, _ = final_arc

            if last_state == INITIAL_STATE:
                continue

            # Back-edge: last_state -> first_state
            # where Initial -> first_state and last_state -> Final
            repeat_arcs: list[Arc] = []
            for initial_arc in initial_arcs:
                initial_guard, initial_action, initial_state = initial_arc
                if initial_state == FINAL_STATE:
                    continue

                repeat_guard = guard
                if upper_bound is not None:
                    upper_predicate = (
                        CounterPredicateType.LESS_THAN, upper_bound)
                    repeat_guard = (
                        *repeat_guard,
                        (self.counter, upper_predicate),
                    )
                repeat_guard = (*repeat_guard, *initial_guard)

                operation = (self.counter, CounterOperationComponent.INCREMENT)
                repeat_action = (*action, operation, *initial_action)
                repeat_arc = (repeat_guard, repeat_action, initial_state)

                repeat_arcs.append(repeat_arc)
            y.follow[last_state].substitute(final_arc, repeat_arcs)

            # Final edge: last_state -> Final
            lower_predicate = (CounterPredicateType.NOT_LESS_THAN, lower_bound)
            final_guard = (*guard, (self.counter, lower_predicate))
            if upper_bound is not None:
                upper_predicate = (
                    CounterPredicateType.NOT_GREATER_THAN, upper_bound)
                final_guard = (*final_guard, (self.counter, upper_predicate))

            final_operation = (
                self.counter, CounterOperationComponent.INACTIVATE)
            final_action = (*action, final_operation)
            final_arc = (final_guard, final_action, FINAL_STATE)
            if lazy:
                y.follow[last_state].prepend(final_arc)
            else:
                y.follow[last_state].append(final_arc)

        # Initial-edge: Initial -> first_state
        new_initial_arcs: list[Arc] = []
        for initial_arc in initial_arcs:
            initial_guard, initial_action, first_state = initial_arc

            if first_state != FINAL_STATE:
                initial_operation = (
                    self.counter, CounterOperationComponent.RESET_OR_ACTIVATE)
                initial_action = (*initial_action, initial_operation)
            initial_arc = (tuple(), initial_action, first_state)
            new_initial_arcs.append(initial_arc)

        y.follow[INITIAL_STATE] = OrderedSet(new_initial_arcs)

        if lower_bound == 0:
            nullable_arc: Arc = (tuple(), tuple(), FINAL_STATE)
            if lazy:
                y.follow[INITIAL_STATE].prepend(nullable_arc)
            else:
                y.follow[INITIAL_STATE].append(nullable_arc)

        if upper_bound is None:
            # Special
            y.counters[self.counter] = lower_bound
        else:
            y.counters[self.counter] = upper_bound
        return y

    def __call__(
        self,
        x: Optional[tuple[NamedIntConstant, Any]],
        ys: Iterable[PositionCounterAutomaton],
    ) -> PositionCounterAutomaton:

        if x is None:
            return reduce(self.call_catenation, ys, self.call_empty())

        opcode, operand = x
        if opcode in PREDICATE_OPCODES:
            return self.call_predicate(x)
        elif opcode is AT:
            return self.call_at(x)
        elif opcode == BRANCH:
            return reduce(self.call_union, ys)
        elif opcode in {MIN_REPEAT, MAX_REPEAT}:
            y = next(iter(ys))
            lazy = opcode == MIN_REPEAT

            m, n = operand
            if m == 0 and n is MAXREPEAT:
                return self.call_star(y, lazy)
            if m == 1 and n is MAXREPEAT:
                return self.call_plus(y, lazy)
            if m == 0 and n == 1:
                return self.call_question(y, lazy)

            if n is MAXREPEAT:
                return self.call_repeat(y, m, None, lazy)
            else:
                return self.call_repeat(y, m, n, lazy)
        elif opcode in {ATOMIC_GROUP, SUBPATTERN}:
            return next(iter(ys))
        else:
            assert False, f"Unknown opcode: {opcode}"
