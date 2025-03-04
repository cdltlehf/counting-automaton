"""Position counter automaton."""

# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import

from collections import defaultdict as dd
from copy import copy
from functools import reduce
from json import dumps
import logging
import re._compiler as compiler  # type: ignore[import-untyped]
from re._parser import SubPattern  # type: ignore[import-untyped]
from typing import Any, Iterable, Optional

from .counter_cartesian_super_config import CounterCartesianSuperConfig
from .counter_cartesian_super_config import StateMap
from .counter_vector import Action
from .counter_vector import CounterVariable
from .counter_vector import CounterVector
from .counter_vector import Guard
from .more_collections import OrderedSet
from .parser_tools import fold
from .parser_tools import MAX_REPEAT
from .parser_tools import MIN_REPEAT
from .parser_tools import parse
from .parser_tools.constants import *

State = int
SymbolPredicate = Any
Arc = tuple[Guard, Action, State]
Follow = dd[State, OrderedSet[Arc]]
Config = tuple[State, CounterVector]
# SuperConfig = OrderedSet[Config]
SuperConfig = dd[State, OrderedSet[CounterVector]]

INITIAL_STATE: State = 0
FINAL_STATE: State = -1


def arc_to_str(arc: Arc) -> str:
    guard, action, adjacent_state = arc
    return f"-{{{guard}; {action}}}-> {adjacent_state}"


def iterate_super_config(super_config: SuperConfig) -> Iterable[Config]:
    for state, counter_vectors in super_config.items():
        for counter_vector in counter_vectors:
            yield (state, counter_vector)


def counter_vector_to_json(
    counter_vector: CounterVector,
) -> list[Optional[int]]:
    return counter_vector.to_list()


def config_to_json(config: Config) -> tuple[State, list[Optional[int]]]:
    state, counter_vector = config
    return (state, counter_vector_to_json(counter_vector))


def super_config_to_json(
    super_config: SuperConfig,
) -> dict[State, list[list[Optional[int]]]]:
    return {
        state: [counter_vector_to_json(config) for config in configs]
        for state, configs in super_config.items()
    }


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

        follow_string = "\n".join(
            "\n".join(f"- {state} {arc_to_str(arc)}" for arc in follow)
            for state, follow in self.follow.items()
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
        logging.debug(dumps(config_to_json(config)))
        for guard, _, adjacent_state in self.follow[cur_state]:
            if adjacent_state is not FINAL_STATE:
                continue
            if guard(counter_vector):
                return True
        return False

    def is_nullable(self) -> bool:
        for _, _, adjacent_state in self.follow[INITIAL_STATE]:
            if adjacent_state is FINAL_STATE:
                return True
        return False

    def get_next_configs(self, config: Config, symbol: str) -> list[Config]:
        cur_state, counter_vector = config
        next_configs: list[Config] = []

        for guard, action, adjacent_state in self.follow[cur_state]:
            if not guard(counter_vector):
                continue

            if adjacent_state is not FINAL_STATE:
                if not self.eval_state(adjacent_state, symbol):
                    continue

            # TODO: Optimization. If there is only one possible action,
            # we can avoid cloning the counter vector.
            next_counter_vector = copy(counter_vector)
            action.move_and_apply(next_counter_vector)
            next_configs.append((adjacent_state, next_counter_vector))
        return next_configs

    def get_next_super_config(
        self, super_config: SuperConfig, symbol: str
    ) -> SuperConfig:
        assert len(symbol) == 1

        next_super_config: SuperConfig = dd(OrderedSet)
        for config in iterate_super_config(super_config):
            next_configs = self.get_next_configs(config, symbol)
            for state, counter_vector in next_configs:
                next_super_config[state].append(counter_vector)
        return next_super_config

    def get_initial_config(self) -> Config:
        return (INITIAL_STATE, CounterVector(self.counters))

    def get_initial_super_config(self) -> SuperConfig:
        initial_config = self.get_initial_config()
        initial_state, initial_counter_vector = initial_config

        super_config: SuperConfig = dd(OrderedSet)
        super_config[initial_state].append(initial_counter_vector)
        return super_config

    def iterate_super_configs(self, w: str) -> Iterable[SuperConfig]:
        super_config = self.get_initial_super_config()
        yield super_config

        for index, symbol in enumerate(w):
            logging.debug("%s", w)
            logging.debug("%s", " " * index + "^" + symbol)
            logging.debug(dumps(super_config_to_json(super_config)))
            super_config = self.get_next_super_config(super_config, symbol)
            yield super_config

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
        logging.debug(dumps(super_config_to_json(super_config)))
        logging.debug("Matching end")
        return any(
            self.check_final(config)
            for config in iterate_super_config(super_config)
        )

    def __call__(self, w: str) -> bool:
        return self.match(w)

    def get_next_counter_cartesian_super_config(
        self, super_config: CounterCartesianSuperConfig, symbol: str
    ) -> CounterCartesianSuperConfig:

        next_counter_set_vectors: StateMap = {}
        for (
            state,
            counter_set_vector,
        ) in super_config.counter_set_vectors.items():
            for arc in self.follow[state]:
                result = super_config.apply_arc_to_counter_set_vector(
                    counter_set_vector, arc
                )

                if result is None:
                    logging.warning(
                        "(%s, %s) %s (failed)",
                        state,
                        dumps(counter_set_vector),
                        arc_to_str(arc),
                    )
                    continue

                adjacent_state, adjacent_counter_set_vector = result
                logging.warning(
                    "(%s, %s) %s; %s",
                    state,
                    dumps(counter_set_vector),
                    arc_to_str(arc),
                    dumps(adjacent_counter_set_vector),
                )

                if adjacent_state is FINAL_STATE:
                    continue

                if not self.eval_state(adjacent_state, symbol):
                    continue

                if adjacent_state not in next_counter_set_vectors:
                    next_counter_set_vectors[adjacent_state] = (
                        adjacent_counter_set_vector
                    )
                else:
                    next_counter_set_vectors[adjacent_state] = (
                        super_config.union_counter_set_vector(
                            next_counter_set_vectors[adjacent_state],
                            adjacent_counter_set_vector,
                        )
                    )
            super_config.free_counter_set_vector(counter_set_vector)
        super_config.counter_set_vectors = next_counter_set_vectors
        return super_config

    def iterate_counter_cartesian_super_configs(
        self, w: str
    ) -> Iterable[CounterCartesianSuperConfig]:
        super_config = CounterCartesianSuperConfig(INITIAL_STATE, self.counters)
        yield super_config
        for index, symbol in enumerate(w):
            logging.debug("%s", w)
            logging.debug("%s", " " * index + "^" + symbol)
            logging.debug(dumps(super_config.to_json()))
            super_config = self.get_next_counter_cartesian_super_config(
                super_config, symbol
            )
            yield super_config

    def counter_cartesian_check_final(
        self, super_config: CounterCartesianSuperConfig
    ) -> bool:
        for state, counter_set_vector in super_config.items():
            for guard, _, adjacent_state in self.follow[state]:
                if adjacent_state is not FINAL_STATE:
                    continue
                if super_config.evaluate_guard(guard, counter_set_vector):
                    return True
        return False

    def counter_cartesian_match(self, w: str) -> bool:
        logging.debug("Counter Cartesian matching")

        if not w:
            return self.is_nullable()

        super_config: CounterCartesianSuperConfig
        for super_config in self.iterate_counter_cartesian_super_configs(w):
            if not super_config:
                return False

        logging.debug("%s", w)
        logging.debug("%s", " " * len(w) + "^")
        # logging.debug(dumps(super_config_to_json(super_config)))
        logging.debug("Matching end")
        return self.counter_cartesian_check_final(super_config)

    def backtrack(self, w: str) -> bool:
        logging.debug("Backtrack matching")

        def _backtrack(w: str, config: Config, index: int) -> bool:
            logging.debug("%s", w)
            logging.debug("%s", " " * index + "^")
            logging.debug("%d %s", index, str(config))
            if len(w) == index:
                return self.check_final(config)

            next_configs = self.get_next_configs(config, w[index])
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

    @staticmethod
    def create_simple_arc(state: State) -> Arc:
        return (Guard(), Action(), state)

    def call_empty(self) -> PositionCounterAutomaton:
        follow: Follow = dd(OrderedSet)
        follow[INITIAL_STATE].append(self.create_simple_arc(FINAL_STATE))
        return PositionCounterAutomaton({}, {}, follow)

    def call_predicate(self, x: tuple[str, Any]) -> PositionCounterAutomaton:
        _, operand = x
        self.state += 1

        follow: Follow = dd(OrderedSet)
        follow[INITIAL_STATE].append(self.create_simple_arc(self.state))
        follow[self.state].append(self.create_simple_arc(FINAL_STATE))
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
                new_guard = guard + initial_guard
                new_action = action + initial_action
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
                new_guard = guard + initial_guard
                new_action = action + initial_action
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
            y.follow[INITIAL_STATE].prepend(self.create_simple_arc(FINAL_STATE))
        else:
            y.follow[INITIAL_STATE].append(self.create_simple_arc(FINAL_STATE))
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

                repeat_guard = copy(guard)
                # NOTE: We can choose to add this guard or not.
                # if upper_bound is not None:
                #     repeat_guard += Guard.less_than(self.counter, upper_bound)
                repeat_guard += initial_guard

                repeat_action = action + Action.increase(self.counter)
                repeat_action += initial_action
                repeat_arc = (repeat_guard, repeat_action, initial_state)

                repeat_arcs.append(repeat_arc)
            y.follow[last_state].substitute(final_arc, repeat_arcs)

            # Final edge: last_state -> Final
            final_guard = guard + Guard.not_less_than(self.counter, lower_bound)
            if upper_bound is not None:
                final_guard += Guard.not_greater_than(self.counter, upper_bound)

            final_action = action + Action.inactivate(self.counter)
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
                initial_action = initial_action + Action.activate(self.counter)
            initial_arc = (Guard(), initial_action, first_state)
            new_initial_arcs.append(initial_arc)

        y.follow[INITIAL_STATE] = OrderedSet(new_initial_arcs)

        if lower_bound == 0:
            nullable_arc: Arc = self.create_simple_arc(FINAL_STATE)
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
        if opcode in {LITERAL, ANY, NOT_LITERAL, IN}:
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
