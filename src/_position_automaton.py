"""Position automaton"""

from collections import dd
from functools import reduce
from itertools import chain
from itertools import takewhile
import logging
import re._compiler as compiler  # type: ignore[import-untyped]
from re._constants import _NamedIntConstant as NamedIntConstant  # type: ignore[import-untyped]
from re._constants import AT
from re._constants import ATOMIC_GROUP
from re._constants import BRANCH
from re._constants import MAXREPEAT
from re._constants import SUBPATTERN
from re._parser import SubPattern  # type: ignore[import-untyped]
import string
from typing import Any, Iterable, Optional, Union

from .more_collections import OrderedSet
from .parser_tools import fold
from .parser_tools import MAX_REPEAT
from .parser_tools import MIN_REPEAT
from .parser_tools import parse
from .parser_tools import PREDICATE_OPCODES

Follow = dd[int, OrderedSet[int]]


class PositionAutomaton:
    """Position automaton representation."""

    def __init__(self, states: dict[int, Any], follow: Follow) -> None:
        self.states = states
        self.follow = follow

    @property
    def initial(self) -> OrderedSet[int]:
        return self.follow[0]

    @property
    def final(self) -> OrderedSet[int]:
        return OrderedSet(
            state
            for state, adjacent_states in self.follow.items()
            if -1 in adjacent_states
        )

    @property
    def nullable(self) -> bool:
        return -1 in self.initial

    def __str__(self) -> str:
        follow_string = (
            "{"
            + ", ".join(
                f"{state}: {follow}" for state, follow in self.follow.items()
            )
            + "}"
        )
        return "\n".join(
            [
                f"states: {self.states}",
                f"initial: {self.initial}",
                f"follow: {follow_string}",
                f"final: {self.final}",
                f"nullable: {self.nullable}",
            ]
        )

    @classmethod
    def create(cls, pattern: str) -> "PositionAutomaton":
        tree = parse(pattern)
        logging.debug(tree)
        callback_object = _PositionConstructionCallback()

        def callback(
            x: Optional[tuple[NamedIntConstant, Any]],
            ys: Iterable[PositionAutomaton],
        ) -> PositionAutomaton:
            automaton = callback_object(x, ys)
            logging.debug(automaton)
            logging.debug("\n")
            return automaton

        return fold(callback, tree)

    def eval_state(self, state: int, symbol: str) -> bool:
        assert len(symbol) == 1

        if state == -1:
            return True
        elif state not in self.states:
            raise ValueError(f"state {state} not found")
        elif isinstance(self.states[state], str):
            return bool(self.states[state] == symbol)
        elif isinstance(self.states[state], SubPattern):
            compiled = compiler.compile(self.states[state])
            return compiled.fullmatch(symbol) is not None
        assert False, type(self.states[state])

    def get_next_states(
        self, cur_states: Union[Iterable[int], int], symbol: str
    ) -> OrderedSet[int]:
        assert len(symbol) == 1

        if isinstance(cur_states, int):
            cur_states = [cur_states]

        adjacent_states = OrderedSet(
            chain.from_iterable(self.follow[p] for p in cur_states)
        )
        next_states = OrderedSet(
            state for state in adjacent_states if self.eval_state(state, symbol)
        )
        return next_states

    def __call__(self, w: str) -> bool:
        return self.match(w)

    def match(self, w: str) -> bool:
        last_states = reduce(self.get_next_states, w, OrderedSet([0]))
        return any(state in last_states for state in self.final)

    def match_prefix(self, w: str) -> Optional[int]:
        logging.debug("Prefix matching")
        end = None

        states = OrderedSet([0])
        for index, c in enumerate(w):
            logging.debug("%d %s", index, str(states))
            states = self.get_next_states(states, c)
            if -1 in states:
                end = index
            states = OrderedSet(takewhile(lambda x: x != -1, states))
        if any(state in states for state in self.final):
            end = len(w)
        return end

    def backtrack(self, w: str) -> bool:
        logging.debug("Backtrack matching")

        def _backtrack(w: str, state: int, index: int) -> bool:
            logging.debug("%d %d", index, state)
            if len(w) == index:
                return state in self.final

            next_states = self.get_next_states(state, w[index])
            return any(_backtrack(w, state, index + 1) for state in next_states)

        return _backtrack(w, 0, 0)

    def backtrack_prefix(self, w: str) -> Optional[int]:
        logging.debug("Backtrack prefix matching")

        def _backtrack(w: str, state: int, index: int) -> Optional[int]:
            logging.debug("%d %d", index, state)
            if state == -1:
                return index - 1

            if len(w) == index:
                return index if state in self.final else None

            next_states = self.get_next_states(state, w[index])
            for state in next_states:
                result = _backtrack(w, state, index + 1)
                if result is not None:
                    return result
            return None

        return _backtrack(w, 0, 0)

    def is_one_unambiguous(self) -> bool:
        flag = True
        for p in self.states:
            if any(
                len(self.get_next_states(p, c)) > 1 for c in string.printable
            ):
                flag = False
                break
        return flag


class _PositionConstructionCallback:
    """Callback function for constructing position automata.
    It is a stateful object."""

    def __init__(self) -> None:
        self.state = 0

    def call_empty(self) -> PositionAutomaton:
        follow: Follow = dd(OrderedSet)
        follow[0].append(-1)
        return PositionAutomaton({}, follow)

    def call_predicate(self, x: tuple[str, Any]) -> PositionAutomaton:
        self.state += 1
        _, operand = x

        follow: Follow = dd(OrderedSet)
        follow[0].append(self.state)
        follow[self.state].append(-1)
        return PositionAutomaton({self.state: operand}, follow)

    def call_at(self, x: tuple[str, Any]) -> PositionAutomaton:
        raise NotImplementedError("Anchor is not supported")

    def call_catenation(
        self, y1: PositionAutomaton, y2: PositionAutomaton
    ) -> PositionAutomaton:
        assert y1.states.keys().isdisjoint(y2.states.keys())

        for state in y1.final:
            y1.follow[state].substitute(-1, y2.initial)

        for state in y2.states:
            assert state not in y1.follow, str(y1.follow.keys())
            y1.follow[state] = y2.follow[state]

        y1.states.update(y2.states)
        return y1

    def call_union(
        self, y1: PositionAutomaton, y2: PositionAutomaton
    ) -> PositionAutomaton:
        assert y1.states.keys().isdisjoint(y2.states.keys())

        y1.initial.append_iterable(y2.initial)
        for state in y2.states:
            assert state not in y1.follow
            y1.follow[state] = y2.follow[state]

        y1.states.update(y2.states)
        return y1

    def call_star(self, y: PositionAutomaton, lazy: bool) -> PositionAutomaton:
        y = self.call_plus(y, lazy)
        y = self.call_question(y, lazy)
        return y

    def call_plus(self, y: PositionAutomaton, lazy: bool) -> PositionAutomaton:
        for state in y.final:
            y.follow[state].substitute(-1, y.initial)
            if lazy:
                y.follow[state].prepend(-1)
            else:
                y.follow[state].append(-1)
        return y

    def call_question(
        self, y: PositionAutomaton, lazy: bool
    ) -> PositionAutomaton:
        if lazy:
            if y.nullable:
                y.follow[0].remove(-1)
            y.follow[0].prepend(-1)
        else:
            y.follow[0].append(-1)
        return y

    def __call__(
        self,
        x: Optional[tuple[NamedIntConstant, Any]],
        ys: Iterable[PositionAutomaton],
    ) -> PositionAutomaton:

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
            else:
                raise NotImplementedError("Counter not supported")
        elif opcode in {ATOMIC_GROUP, SUBPATTERN}:
            return next(iter(ys))
        else:
            raise NotImplementedError(f"{opcode} not supported")
