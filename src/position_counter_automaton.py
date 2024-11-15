"""Position counter automaton."""

from collections import deque
from enum import Enum
from functools import reduce
from itertools import product
import re._compiler as compiler  # type: ignore[import-untyped]
from re._constants import _NamedIntConstant as NamedIntConstant  # type: ignore[import-untyped]
from re._constants import AT
from re._constants import ATOMIC_GROUP
from re._constants import BRANCH
from re._constants import MAXREPEAT
from re._constants import SUBPATTERN
from re._parser import SubPattern  # type: ignore[import-untyped]
from typing import Any, Iterable, Optional, Union

from .more_collections import OrderedSet
from .parser_tools import fold
from .parser_tools import has_extended_features
from .parser_tools import parse
from .parser_tools import PREDICATE_OPCODES
from .parser_tools import REPEAT_OPCODES


class CounterOperationComponent(Enum):
    NOOP = 0
    INCREMENT = 1
    RESET = 2


CounterPredicate = tuple[int, Optional[int]]
Counter = list[int]
CounterGuardSparse = Iterable[tuple[int, CounterPredicate]]
CounterOperationSparse = Iterable[tuple[int, CounterOperationComponent]]

CounterGuard = list[CounterPredicate]
CounterOperation = list[CounterOperationComponent]


def evaluate_guard(counter: Counter, guard: CounterGuard) -> bool:
    return all(
        (
            low <= counter[index]
            and (counter[index] <= high if high is not None else True)
        )
        for index, (low, high) in guard
    )


def apply_operation(counter: Counter, operation: CounterOperation) -> Counter:
    counter = counter.copy()
    for index, component in operation:
        if component is CounterOperationComponent.INCREMENT:
            counter[index] += 1
        elif component is CounterOperationComponent.RESET:
            counter[index] = 0
    return counter


class PositionCounterAutomaton:
    """Position counter automaton."""

    def __init__(
        self,
        states: Optional[dict[int, Any]] = None,
        initial: Optional[deque[int]] = None,
        follow: Optional[deque[tuple[int, int]]] = None,
        final: Optional[deque[int]] = None,
        nullable: bool = True,
        operations: Optional[
            deque[tuple[int, int, CounterOperationComponent]]
        ] = None,
        guards: Optional[deque[tuple[int, int, CounterPredicate]]] = None,
    ) -> None:
        super().__init__(states, initial, follow, final, nullable)
        self.operations = operations if operations is not None else deque()
        self.guards = guards if guards is not None else deque()

        self._operation_dict: Optional[dict[int, CounterOperation]] = None
        self._guard_dict: Optional[dict[int, CounterGuard]] = None
        self._len_counter: Optional[int] = None

    def __str__(self) -> str:
        return "\n".join(
            [
                f"states: {self.states}",
                f"initial: {set(self.initial)}",
                f"follow: {set(self.follow)}",
                f"final: {set(self.final)}",
                f"nullable: {self.nullable}",
                f"operations: {self.operations}",
            ]
        )

    @property
    def len_counter(self) -> int:
        if self._len_counter is None:
            max_index_operation = max(index for _, index, _ in self.operations)
            max_index_guard = max(index for _, index, _ in self.guards)
            self._len_counter = max(max_index_operation, max_index_guard) + 1
        return self._len_counter

    @property
    def guard_dict(self) -> dict[int, CounterGuard]:
        if self._guard_dict is None:
            self._guard_dict = {
                state: [(index, predicate)]
                for state, index, predicate in self.guards
            }
        return self._guard_dict

    @property
    def operation_dict(self) -> dict[int, CounterOperation]:
        if self._operation_dict is None:
            self._operation_dict = {
                state: [(index, operation)]
                for state, index, operation in self.operations
            }
        return self._operation_dict

    @classmethod
    def create(cls, pattern: str) -> "PositionCounterAutomaton":
        tree = parse(pattern)
        if has_extended_features(tree):
            raise ValueError("Pattern has extended features")
        callback = _PositionConstructionCallback()
        return fold(callback, tree)

    def eval_state(self, state: int, c: str) -> bool:
        if state not in self.states:
            raise ValueError(f"state {state} not found")
        if len(c) != 1:
            raise ValueError("c must be a single character")
        if isinstance(self.states[state], str):
            return bool(self.states[state] == c)
        elif isinstance(self.states[state], SubPattern):
            compiled = compiler.compile(self.states[state])
            return compiled.fullmatch(c) is not None
        assert False, type(self.states[state])

    def get_next_states(
        self,
        cur_states: Optional[Union[Iterable[int], int]],
        c: str,
        counter: Optional[Counter] = None,
    ) -> OrderedSet[int]:
        if counter is None:
            raise ValueError("counter must be provided")
        states = super().get_next_states(cur_states, c)
        return OrderedSet(
            state
            for state in states
            if evaluate_guard(counter, self.guard_dict[state])
        )

    def __call__(self, w: str) -> bool:
        raise NotImplementedError("Use backtrack instead")

    def backtrack(self, w: str) -> bool:
        counter = [0] * self.len_counter
        return any(
            self._backtrack(w, state, 0, counter) for state in self.initial
        )

    def _backtrack(
        self, w: str, state: int, index: int, counter: Counter
    ) -> bool:
        if len(w) == index:
            return state in self.final

        next_states = self.get_next_states(state, w[index])
        return any(
            self._backtrack(w, state, index + 1, counter)
            for state in next_states
        )

    def is_one_unambiguous(self) -> bool:
        raise NotImplementedError("Not implemented yet")


class _PositionConstructionCallback:
    """Callback function for constructing position automata.
    It is a stateful object."""

    def __init__(self) -> None:
        self.state = -1
        self.counter = -1

    def _call_catenation(
        self, y1: PositionCounterAutomaton, y2: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        y1.states.update(y2.states)

        if y1.nullable:
            y1.initial.extend(y2.initial)

        y1.follow.extend(product(y1.final, y2.initial))
        y1.follow.extend(y2.follow)  # FIXME: Slow

        if y2.nullable:
            y1.final.extend(y2.final)
        else:
            y1.final = y2.final

        if not y2.nullable:
            y1.nullable = False

        return y1

    def _call_union(
        self, y1: PositionCounterAutomaton, y2: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        y1.states.update(y2.states)
        y1.initial.extend(y2.initial)
        y1.final.extend(y2.final)
        y1.follow.extend(y2.follow)
        y1.nullable |= y2.nullable

        y1.operations.extend(y2.operations)
        y1.guards.extend(y2.guards)
        return y1

    def _call_predicate(self, x: tuple[str, Any]) -> PositionCounterAutomaton:
        self.state += 1
        _, operand = x
        return PositionCounterAutomaton(
            states={self.state: operand},
            initial=deque([self.state]),
            final=deque([self.state]),
            follow=deque(),
            nullable=False,
        )

    def _call_at(self, x: tuple[str, Any]) -> PositionCounterAutomaton:
        self.state += 1
        _, operand = x
        return PositionCounterAutomaton(
            states={self.state: operand},
            initial=deque([self.state]),
            final=deque([self.state]),
            follow=deque(),
            nullable=True,
        )

    def _call_star(
        self, y: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        y.follow.extend(product(y.final, y.initial))
        y.nullable = True
        return y

    def _call_plus(
        self, y: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        y.follow.extend(product(y.final, y.initial))
        return y

    def _call_question(
        self, y: PositionCounterAutomaton
    ) -> PositionCounterAutomaton:
        y.nullable = True
        return y

    def __call__(
        self,
        x: Optional[tuple[NamedIntConstant, Any]],
        ys: Iterable[PositionCounterAutomaton],
    ) -> PositionCounterAutomaton:

        if x is None:
            return reduce(self._call_catenation, ys, PositionCounterAutomaton())

        opcode, operand = x
        if opcode in PREDICATE_OPCODES:
            return self._call_predicate(x)
        elif opcode is AT:
            return self._call_at(x)
        elif opcode == BRANCH:
            return reduce(self._call_union, ys, PositionCounterAutomaton())
        elif opcode in REPEAT_OPCODES:
            y = next(iter(ys))

            m, n = operand
            if m == 0 and n is MAXREPEAT:
                return self._call_star(y)
            if m == 1 and n is MAXREPEAT:
                return self._call_plus(y)
            if m == 0 and n == 1:
                return self._call_question(y)
            else:
                raise NotImplementedError()
        elif opcode in {ATOMIC_GROUP, SUBPATTERN}:
            return next(iter(ys))
        else:
            assert False, f"Unknown opcode: {opcode}"
