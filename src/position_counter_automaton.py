"""Position counter automaton."""

from collections import defaultdict
from enum import StrEnum
from functools import reduce
import logging
import re._compiler as compiler  # type: ignore[import-untyped]
from re._constants import _NamedIntConstant as NamedIntConstant  # type: ignore[import-untyped]
from re._constants import AT
from re._constants import ATOMIC_GROUP
from re._constants import BRANCH
from re._constants import MAXREPEAT
from re._constants import SUBPATTERN
from re._parser import SubPattern  # type: ignore[import-untyped]
from typing import Any, Iterable, Optional

from .more_collections import OrderedSet
from .parser_tools import fold
from .parser_tools import MAX_REPEAT
from .parser_tools import MIN_REPEAT
from .parser_tools import parse
from .parser_tools import PREDICATE_OPCODES


class CounterOperationComponent(StrEnum):
    INCREMENT = "++"
    RESET = " = 1"


class CounterPredicateType(StrEnum):
    GE = " >= "
    LE = " <= "
    LT = " < "


Counter = int
CounterVec = tuple[Counter, ...]
CounterPredicate = tuple[CounterPredicateType, int]
Guard = Iterable[tuple[Counter, CounterPredicate]]
Action = Iterable[tuple[Counter, CounterOperationComponent]]

Follow = defaultdict[int, OrderedSet[tuple[Guard, Action, int]]]


def eval_guard(guard: Guard, counter_vec: CounterVec) -> bool:
    for counter, (predicate_type, value) in guard:
        if predicate_type is CounterPredicateType.GE:
            if counter_vec[counter] < value:
                return False
        elif predicate_type is CounterPredicateType.LE:
            if counter_vec[counter] > value:
                return False
        elif predicate_type is CounterPredicateType.LT:
            if counter_vec[counter] >= value:
                return False
        else:
            assert False, predicate_type
    return True


def apply_action(action: Action, counter_vec: CounterVec) -> CounterVec:

    def apply(
        counter_vec: CounterVec,
        operation: tuple[int, CounterOperationComponent],
    ) -> CounterVec:
        counter, op_type = operation
        if op_type is CounterOperationComponent.INCREMENT:
            return tuple(
                e + 1 if i == counter else e for i, e in enumerate(counter_vec)
            )
        elif op_type is CounterOperationComponent.RESET:
            return tuple(
                1 if i == counter else e for i, e in enumerate(counter_vec)
            )
        assert False, operation

    return reduce(apply, action, counter_vec)


class PositionCounterAutomaton:
    """Position counter automaton."""

    def __init__(
        self,
        states: dict[int, Any],
        follow: Follow,
        counter: int,
    ) -> None:
        self.states = states
        self.follow = follow
        self.counter = counter

    @property
    def final(self) -> OrderedSet[int]:
        final_states: OrderedSet[int] = OrderedSet()
        for state, arcs in self.follow.items():
            if any(-1 == adjacent_state for _, _, adjacent_state in arcs):
                final_states.append(state)
        return final_states

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

    def eval_state(self, state: int, symbol: str) -> bool:
        assert len(symbol) == 1
        if state == -1:
            return True

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

        def arc_to_str(arc: tuple[Guard, Action, int]) -> str:
            guard, action, adjacent_state = arc
            return (
                f"{guard_to_str(guard)}; {action_to_str(action)}"
                + f" -> {adjacent_state}"
            )

        def follow_to_str(follow: OrderedSet[tuple[Guard, Action, int]]) -> str:
            return ",\n    ".join(arc_to_str(arc) for arc in follow)

        follow_string = (
            "{\n"
            + ",\n".join(
                f"  {state}:\n    {follow_to_str(follow)}"
                for state, follow in self.follow.items()
            )
            + "\n}"
        )
        return "\n".join(
            [
                f"states: {self.states}",
                f"follow: {follow_string}",
                f"counter: {self.counter}",
            ]
        )

    def get_next_configs(
        self, config: tuple[int, CounterVec], symbol: str
    ) -> OrderedSet[tuple[int, CounterVec]]:
        cur_state, counter_vec = config
        next_configs: OrderedSet[tuple[int, CounterVec]] = OrderedSet()

        for guard, action, adjacent_state in self.follow[cur_state]:
            if not eval_guard(guard, counter_vec):
                continue
            if not self.eval_state(adjacent_state, symbol):
                continue

            counter_vec = apply_action(action, counter_vec)
            next_configs.append((adjacent_state, counter_vec))
        return next_configs

    def __call__(self, w: str) -> bool:
        raise NotImplementedError("Use backtrack instead")

    def check_final(self, config: tuple[int, CounterVec]) -> bool:
        cur_state, counter_vec = config
        for guard, _, adjacent_state in self.follow[cur_state]:
            if adjacent_state != -1:
                continue
            if not eval_guard(guard, counter_vec):
                continue
            return True
        return False

    def backtrack(self, w: str) -> bool:
        logging.debug("Backtrack matching")

        def _backtrack(
            w: str, config: tuple[int, CounterVec], index: int
        ) -> bool:
            logging.debug("%s", w)
            logging.debug("%s", " " * index + "^")
            logging.debug("%d %s", index, str(config))
            if len(w) == index:
                return self.check_final(config)

            next_configs = self.get_next_configs(config, w[index])
            return any(
                _backtrack(w, config, index + 1) for config in next_configs
            )

        initial_counter = (1,) * self.counter
        return _backtrack(w, (0, initial_counter), 0)


class _PositionConstructionCallback:
    """Callback function for constructing position automata.
    It is a stateful object."""

    def __init__(self) -> None:
        self.state = 0
        self.counter = 0

    @staticmethod
    def find_final_arc(
        arcs: OrderedSet[tuple[Guard, Action, int]],
    ) -> tuple[Guard, Action, int]:
        for guard, action, adjacent_state in arcs:
            if adjacent_state == -1:
                return guard, action, adjacent_state
        assert False

    def call_empty(self) -> PositionCounterAutomaton:
        follow: Follow = defaultdict(OrderedSet)
        follow[0].append((tuple(), tuple(), -1))
        return PositionCounterAutomaton({}, follow, self.counter)

    def call_predicate(self, x: tuple[str, Any]) -> PositionCounterAutomaton:
        self.state += 1
        _, operand = x

        follow: Follow = defaultdict(OrderedSet)
        follow[0].append((tuple(), tuple(), self.state))
        follow[self.state].append((tuple(), tuple(), -1))
        return PositionCounterAutomaton(
            {self.state: operand}, follow, self.counter
        )

    def call_at(self, x: tuple[str, Any]) -> PositionCounterAutomaton:
        raise NotImplementedError("Anchor is not supported")

    def call_catenation(
        self, y1: PositionCounterAutomaton, y2: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        assert y1.states.keys().isdisjoint(y2.states.keys())

        for state in y1.final:
            final_arc = self.find_final_arc(y1.follow[state])
            guard, action, _ = final_arc

            arcs: list[tuple[Guard, Action, int]] = []
            for initial_guard, initial_action, initial_state in y2.follow[0]:
                new_guard = (*guard, *initial_guard)
                new_action = (*action, *initial_action)
                arcs.append((new_guard, new_action, initial_state))
            y1.follow[state].substitute(final_arc, arcs)

        for state in y2.states:
            assert state not in y1.follow, str(y1.follow.keys())
            y1.follow[state] = y2.follow[state]

        y1.states.update(y2.states)
        y1.counter = self.counter
        return y1

    def call_union(
        self, y1: PositionCounterAutomaton, y2: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        assert y1.states.keys().isdisjoint(y2.states.keys())

        y1.follow[0].append_iterable(y2.follow[0])
        for state in y2.states:
            assert state not in y1.follow
            y1.follow[state] = y2.follow[state]

        y1.states.update(y2.states)
        y1.counter = self.counter
        return y1

    def call_star(
        self, y: PositionCounterAutomaton, lazy: bool
    ) -> PositionCounterAutomaton:
        y = self.call_plus(y, lazy)
        y = self.call_question(y, lazy)
        y.counter = self.counter
        return y

    def call_plus(
        self, y: PositionCounterAutomaton, lazy: bool
    ) -> PositionCounterAutomaton:
        for state in y.final:
            final_arc = self.find_final_arc(y.follow[state])
            guard, action, _ = final_arc

            arcs: list[tuple[Guard, Action, int]] = []
            for initial_guard, initial_action, initial_state in y.follow[0]:
                new_guard = (*guard, *initial_guard)
                new_action = (*action, *initial_action)
                arcs.append((new_guard, new_action, initial_state))
            y.follow[state].substitute(final_arc, arcs)
            if lazy:
                y.follow[state].prepend(final_arc)
            else:
                y.follow[state].append(final_arc)
        y.counter = self.counter
        return y

    def call_question(
        self, y: PositionCounterAutomaton, lazy: bool
    ) -> PositionCounterAutomaton:
        if lazy:
            y.follow[0].prepend((tuple(), tuple(), -1))
        else:
            y.follow[0].append((tuple(), tuple(), -1))
        y.counter = self.counter
        return y

    def call_repeat(
        self,
        y: PositionCounterAutomaton,
        m: int,
        n: Optional[int],
        lazy: bool,
    ) -> PositionCounterAutomaton:
        for state in y.final:
            # final_arc: state --(_; _)-> -1
            final_arc = self.find_final_arc(y.follow[state])
            guard, action, _ = final_arc

            # Back-edge
            repeat_arcs: list[tuple[Guard, Action, int]] = []
            for initial_guard, initial_action, initial_state in y.follow[0]:
                repeat_guard = guard
                if n is not None:
                    upper_predicate = (CounterPredicateType.LT, n)
                    repeat_guard = (
                        *repeat_guard,
                        (self.counter, upper_predicate),
                    )
                repeat_guard = (*repeat_guard, *initial_guard)

                repeat_action = action
                operation = (self.counter, CounterOperationComponent.INCREMENT)
                repeat_action = (*repeat_action, operation, *initial_action)
                repeat_arc = (repeat_guard, repeat_action, initial_state)
                repeat_arcs.append(repeat_arc)
            y.follow[state].substitute(final_arc, repeat_arcs)

            # Final edge
            lower_predicate = (CounterPredicateType.GE, m)
            final_guard = (*guard, (self.counter, lower_predicate))
            if n is not None:
                upper_predicate = (CounterPredicateType.LE, n)
                final_guard = (*final_guard, (self.counter, upper_predicate))

            final_operation = (self.counter, CounterOperationComponent.RESET)
            final_action = (*action, final_operation)
            final_arc = (final_guard, final_action, -1)
            if lazy:
                y.follow[state].prepend(final_arc)
            else:
                y.follow[state].append(final_arc)

        if m == 0:
            if lazy:
                y.follow[0].prepend((tuple(), tuple(), -1))
            else:
                y.follow[0].append((tuple(), tuple(), -1))

        self.counter += 1
        y.counter = self.counter
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
