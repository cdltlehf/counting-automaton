"""Position counting automaton."""

from copy import copy
from functools import reduce
from json import dumps
import logging
from typing import Any, Iterable, NewType, Optional

from cai4py.more_collections import OrderedSet
from cai4py.parser_tools import fold
from cai4py.parser_tools import MAX_PLUS
from cai4py.parser_tools import MAX_QUESTION
from cai4py.parser_tools import MAX_REPEAT
from cai4py.parser_tools import MAX_STAR
from cai4py.parser_tools import MIN_PLUS
from cai4py.parser_tools import MIN_QUESTION
from cai4py.parser_tools import MIN_REPEAT
from cai4py.parser_tools import MIN_STAR
from cai4py.parser_tools import parse
from cai4py.parser_tools.constants import *  # pylint: disable=wildcard-import,unused-wildcard-import
from cai4py.parser_tools.re import _compile
from cai4py.parser_tools.re import SubPattern

from .counter_vector import Action
from .counter_vector import CounterVector
from .counter_vector import Guard
from .logging import ComputationStep
from .logging import VERBOSE

logger = logging.getLogger(__name__)

State = NewType("State", int)
CounterVariable = NewType("CounterVariable", int)
SymbolPredicate = Any
Arc = tuple[Guard[CounterVariable], Action[CounterVariable], State]
Follow = dict[State, OrderedSet[Arc]]
Config = tuple[State, CounterVector[CounterVariable]]
Range = tuple[int, Optional[int]]

INITIAL_STATE = State(0)
FINAL_STATE = State(-1)
GLOBAL_COUNTER = CounterVariable(0)


def arc_to_str(arc: Arc) -> str:
    guard, action, adjacent_state = arc
    return f"-{{{guard}; {action}}}-> {adjacent_state}"


def counter_vector_to_json(
    counter_vector: CounterVector[CounterVariable],
) -> list[Optional[int]]:
    return counter_vector.to_list()


def config_to_json(config: Config) -> tuple[State, list[Optional[int]]]:
    state, counter_vector = config
    return (state, counter_vector_to_json(counter_vector))


class PositionCountingAutomaton:
    """Position counter automaton."""

    def __init__(
        self,
        states: dict[State, SymbolPredicate],
        follow: Follow,
        counters: Optional[dict[CounterVariable, Range]] = None,
        counter_scopes: Optional[dict[CounterVariable, set[State]]] = None,
    ) -> None:
        self.states = states
        self.follow = follow
        self.counters = counters if counters is not None else {}
        if counter_scopes is None:
            counter_scopes = {}
        self.counter_scopes = counter_scopes
        self._state_scopes: Optional[dict[State, set[CounterVariable]]] = None

    @property
    def state_scopes(self) -> dict[State, set[CounterVariable]]:
        if self._state_scopes is None:
            self._state_scopes = {}
            for state in self.states:
                self._state_scopes[state] = set()
            for counter, states in self.counter_scopes.items():
                for state in states:
                    self._state_scopes[state].add(counter)
        return self._state_scopes

    @classmethod
    def create(cls, pattern: str) -> "PositionCountingAutomaton":
        tree = parse(pattern)
        logger.debug(tree)
        callback_object = _PositionConstructionCallback()

        def callback(
            x: Optional[tuple[NamedIntConstant, Any]],
            ys: Iterable[PositionCountingAutomaton],
        ) -> PositionCountingAutomaton:
            automaton = callback_object(x, ys)
            logger.debug(x)
            logger.debug(automaton)
            logger.debug("\n")
            return automaton

        return fold(callback, tree)

    def is_flat(self) -> bool:
        return all(
            len(state_scope) <= 1 for state_scope in self.state_scopes.values()
        )

    def eval_state(self, state: State, symbol: str) -> bool:
        assert len(symbol) == 1
        logger.log(VERBOSE, ComputationStep.EVAL_SYMBOL.value)
        if isinstance(self.states[state], str):
            return bool(self.states[state] == symbol)
        elif isinstance(self.states[state], SubPattern):
            compiled = _compile(self.states[state])
            return (
                compiled.fullmatch(symbol) is not None
            )  # NOTE: this is only used to check for character class matches.
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

    def is_nullable(self) -> bool:
        for _, _, adjacent_state in self.follow[INITIAL_STATE]:
            if adjacent_state is FINAL_STATE:
                return True
        return False

    def get_next_configs(self, config: Config, symbol: str) -> list[Config]:
        current_state, counter_vector = config
        next_configs: list[Config] = []

        if current_state == FINAL_STATE:
            return next_configs

        for guard, action, adjacent_state in self.follow[current_state]:
            if adjacent_state is FINAL_STATE:
                continue

            if not guard(counter_vector):
                continue

            if not self.eval_state(adjacent_state, symbol):
                continue

            next_counter_vector = copy(counter_vector)
            action.move_and_apply(next_counter_vector)
            next_configs.append((adjacent_state, next_counter_vector))
        return next_configs

    def check_final(self, config: Config) -> bool:
        current_state, counter_vector = config
        logger.debug(dumps(config_to_json(config)))
        for guard, _, adjacent_state in self.follow[current_state]:
            if adjacent_state is not FINAL_STATE:
                continue
            if guard(counter_vector):
                return True
        return False

    def get_initial_config(self) -> Config:
        initial_counter = CounterVector(self.counters.keys())
        initial_config = (INITIAL_STATE, initial_counter)
        return initial_config

    def backtrack(self, w: str, config: Config, index: int) -> bool:
        logger.debug("%s", w)
        logger.debug("%s", " " * index + "^")
        logger.debug("%d %s", index, config)

        if len(w) == index:
            return self.check_final(config)

        next_configs = self.get_next_configs(config, w[index])
        return any(
            self.backtrack(w, config, index + 1) for config in next_configs
        )

    def __call__(self, w: str) -> bool:
        logger.debug("Backtrack matching")
        initial_config = self.get_initial_config()
        return self.backtrack(w, initial_config, 0)


class _PositionConstructionCallback:
    """Callback function for constructing position automata.
    It is a stateful object."""

    def __init__(self) -> None:
        self.state = 0
        self.counter = 0

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

    def call_empty(self) -> PositionCountingAutomaton:
        follow: Follow = {}
        follow.setdefault(INITIAL_STATE, OrderedSet()).append(
            self.create_simple_arc(FINAL_STATE)
        )
        return PositionCountingAutomaton({}, follow)

    def call_predicate(self, x: tuple[str, Any]) -> PositionCountingAutomaton:
        _, operand = x
        self.state += 1

        follow: Follow = {}
        follow.setdefault(
            INITIAL_STATE,
            OrderedSet([self.create_simple_arc(State(self.state))]),
        )
        follow.setdefault(
            State(self.state), OrderedSet([self.create_simple_arc(FINAL_STATE)])
        )
        return PositionCountingAutomaton({State(self.state): operand}, follow)

    def call_at(self, x: tuple[str, Any]) -> PositionCountingAutomaton:
        raise NotImplementedError("Anchor is not supported")

    def call_catenation(
        self, y1: PositionCountingAutomaton, y2: PositionCountingAutomaton
    ) -> PositionCountingAutomaton:
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
        for counter, scope in y2.counter_scopes.items():
            y1.counter_scopes.setdefault(counter, set()).update(scope)
        return y1

    def call_union(
        self, y1: PositionCountingAutomaton, y2: PositionCountingAutomaton
    ) -> PositionCountingAutomaton:
        assert y1.states.keys().isdisjoint(y2.states.keys())

        y1.follow[INITIAL_STATE].append_iterable(y2.follow[INITIAL_STATE])
        for state in y2.states:
            assert state not in y1.follow
            y1.follow[state] = y2.follow[state]

        # NOTE: we can optimize this, but it is not necessary.
        y1.states.update(y2.states)
        y1.counters.update(y2.counters)
        for counter, scope in y2.counter_scopes.items():
            y1.counter_scopes.setdefault(counter, set()).update(scope)
        return y1

    def call_star(
        self, y: PositionCountingAutomaton, lazy: bool
    ) -> PositionCountingAutomaton:
        y = self.call_plus(y, lazy)
        y = self.call_question(y, lazy)
        return y

    def call_plus(
        self, y: PositionCountingAutomaton, lazy: bool
    ) -> PositionCountingAutomaton:
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
        self, y: PositionCountingAutomaton, lazy: bool
    ) -> PositionCountingAutomaton:
        if lazy:
            y.follow[INITIAL_STATE].prepend(self.create_simple_arc(FINAL_STATE))
        else:
            y.follow[INITIAL_STATE].append(self.create_simple_arc(FINAL_STATE))
        return y

    def call_repeat(
        self,
        y: PositionCountingAutomaton,
        lower_bound: int,
        upper_bound: Optional[int],
        lazy: bool,
    ) -> PositionCountingAutomaton:
        self.counter += 1

        final_arcs = self.get_final_arcs(y.follow)
        initial_arcs = y.follow[INITIAL_STATE]
        counter_variable = CounterVariable(self.counter)
        if y.is_nullable():
            lower_bound = 0

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
                #     repeat_guard += Guard.less_than(
                #         counter_variable, upper_bound
                #     )
                # else:
                #     repeat_guard += Guard.less_than(
                #         counter_variable, upper_bound
                #     )
                repeat_guard += initial_guard

                repeat_action = action + Action.increase(counter_variable)
                repeat_action += initial_action
                repeat_arc = (repeat_guard, repeat_action, initial_state)

                repeat_arcs.append(repeat_arc)
            y.follow[last_state].substitute(final_arc, repeat_arcs)

            # Final edge: last_state -> Final
            final_guard = guard

            if lower_bound != 0:
                final_guard += Guard.not_less_than(
                    counter_variable, lower_bound
                )
            if upper_bound is not None:
                final_guard += Guard.not_greater_than(
                    counter_variable, upper_bound
                )

            final_action = action + Action.inactivate(counter_variable)
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
                initial_action = initial_action + Action.activate(
                    counter_variable
                )
            initial_arc = (Guard(), initial_action, first_state)
            new_initial_arcs.append(initial_arc)

        y.follow[INITIAL_STATE] = OrderedSet(new_initial_arcs)

        if lower_bound == 0:
            nullable_arc: Arc = self.create_simple_arc(FINAL_STATE)
            if lazy:
                y.follow[INITIAL_STATE].prepend(nullable_arc)
            else:
                y.follow[INITIAL_STATE].append(nullable_arc)

        y.counters[counter_variable] = (lower_bound, upper_bound)
        y.counter_scopes.setdefault(counter_variable, set()).update(
            y.states.keys()
        )
        return y

    def __call__(
        self,
        x: Optional[tuple[NamedIntConstant, Any]],
        ys: Iterable[PositionCountingAutomaton],
    ) -> PositionCountingAutomaton:

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
            if n is MAXREPEAT:
                return self.call_repeat(y, m, None, lazy)
            else:
                return self.call_repeat(y, m, n, lazy)
        elif opcode is MAX_STAR:
            return self.call_star(next(iter(ys)), False)
        elif opcode is MIN_STAR:
            return self.call_star(next(iter(ys)), True)
        elif opcode is MAX_PLUS:
            return self.call_plus(next(iter(ys)), False)
        elif opcode is MIN_PLUS:
            return self.call_plus(next(iter(ys)), True)
        elif opcode is MAX_QUESTION:
            return self.call_question(next(iter(ys)), False)
        elif opcode is MIN_QUESTION:
            return self.call_question(next(iter(ys)), True)
        elif opcode in {ATOMIC_GROUP, SUBPATTERN}:
            return next(iter(ys))
        else:
            raise ValueError(f"Unknown opcode: {opcode}")
