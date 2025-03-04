"""Counter cartesian super config"""

from collections import defaultdict as dd
from collections import deque
from itertools import product
import json
import logging
from typing import Iterator, Mapping, Optional, TypedDict

from .counter_vector import Action
from .counter_vector import CounterOperationComponent
from .counter_vector import CounterPredicate
from .counter_vector import CounterPredicateType
from .counter_vector import Guard
from .more_collections import OrderedSet

ListPtr = int
CounterVariable = int
CounterValue = Optional[int]
Values = deque[int]  # TODO: deque[tuple[int, int]]
# (None, None) -> [None]
# (n, None) -> [n]
# (n, ptr) -> [n + v for v in access(ptr)]
# TODO: Optional[int] -> set[int]
CounterSet = tuple[Optional[int], Optional[ListPtr]]
CounterSetVector = dd[CounterVariable, CounterSet]
State = int
StateMap = dict[State, CounterSetVector]
Arc = tuple[Guard, Action, State]
Follow = dd[State, OrderedSet[Arc]]
ReferenceCounter = dd[ListPtr, int]
ReferenceTable = list[Values]


class CounterCartesianSuperConfigJson(TypedDict):
    """Counter cartesian super config json"""

    counter_set_vectors: StateMap
    reference_table: dict[int, list[int]]
    reference_counter: ReferenceCounter


class CounterCartesianSuperConfig(Mapping[State, CounterSetVector]):
    """Counter cartesian super config"""

    def __init__(
        self,
        initial_state: State,
        counters: dict[CounterVariable, int],
    ) -> None:
        self._reference_counter: ReferenceCounter = dd(int)
        self._reference_table: ReferenceTable = []
        self.counters = counters
        self.counter_set_vectors: StateMap = {
            initial_state: dd(lambda: (None, None))
        }

    def __getitem__(self, state: State) -> CounterSetVector:
        return self.counter_set_vectors[state]

    def __iter__(self) -> Iterator[State]:
        return iter(self.counter_set_vectors)

    def __len__(self) -> int:
        return len(self.counter_set_vectors)

    def access(self, pointer: ListPtr) -> Values:
        return self._reference_table[pointer]

    def malloc(self) -> ListPtr:
        self._reference_table.append(deque())
        ptr = len(self._reference_table) - 1
        self._reference_counter[ptr] += 1
        return ptr

    def reset(self, pointer: ListPtr) -> None:
        self._reference_counter[pointer] -= 1
        if self._reference_counter[pointer] == 0:
            self._reference_table[pointer].clear()

    def copy(self, pointer: ListPtr) -> ListPtr:
        self._reference_counter[pointer] += 1
        return pointer

    def union_counter_set_vector(
        self,
        counter_set_vector_1: CounterSetVector,
        counter_set_vector_2: CounterSetVector,
    ) -> CounterSetVector:
        new_counter_set_vector = self.copy_counter_set_vector(
            counter_set_vector_1
        )
        for counter_variable, counter_set_2 in counter_set_vector_2.items():
            counter_set_1 = new_counter_set_vector[counter_variable]
            new_counter_set_vector[counter_variable] = self.union_counter_set(
                counter_set_1, counter_set_2
            )
        return new_counter_set_vector

    def union_counter_set(
        self, counter_set_1: CounterSet, counter_set_2: CounterSet
    ) -> CounterSet:
        offset_1, ptr_1 = counter_set_1
        offset_2, ptr_2 = counter_set_2

        if offset_1 is None:
            return counter_set_2

        if offset_2 is None:
            return counter_set_1

        # We merge counter_set_2 into counter_set_1
        if ptr_1 is None:
            offset_1, ptr_1 = counter_set_1
            offset_2, ptr_2 = counter_set_2
            counter_set_2, counter_set_1 = counter_set_1, counter_set_2

        if counter_set_1 == counter_set_2:
            self.free_counter_set(counter_set_2)
            return counter_set_1

        self.free_counter_set(counter_set_2)

        raise NotImplementedError()

    def evaluate_predicate_on_counter_set(
        self,
        counter_set: CounterSet,
        predicate: CounterPredicate,
    ) -> bool:

        offset, ptr = counter_set
        if offset is None:
            return True

        if ptr is not None:
            values = self.access(ptr)
            value = offset - {
                CounterPredicateType.NOT_LESS_THAN: values[-1],
                CounterPredicateType.NOT_GREATER_THAN: values[0],
                CounterPredicateType.LESS_THAN: values[0],
            }[predicate.predicate_type]
        else:
            value = offset
        return predicate(value)

    def evaluate_guard(
        self, guard: Guard, counter_set_vector: CounterSetVector
    ) -> bool:
        for counter_variable, predicates in guard.items():
            counter_set = counter_set_vector[counter_variable]
            for predicate in predicates:
                if not self.evaluate_predicate_on_counter_set(
                    counter_set, predicate
                ):
                    return False
        return True

    def free_counter_set(self, counter_set: CounterSet) -> None:
        _, ptr = counter_set
        if ptr is None:
            return
        self.reset(ptr)

    def free_counter_set_vector(
        self, counter_set_vector: CounterSetVector
    ) -> None:
        for counter_set in counter_set_vector.values():
            self.free_counter_set(counter_set)

    def copy_counter_set(self, counter_set: CounterSet) -> CounterSet:
        offset, ptr = counter_set
        if ptr is None:
            return counter_set
        return offset, self.copy(ptr)

    def copy_counter_set_vector(
        self, counter_set_vector: CounterSetVector
    ) -> CounterSetVector:
        new_counter_set_vector = {}
        for counter_variable, counter_set in counter_set_vector.items():
            new_counter_set_vector[counter_variable] = self.copy_counter_set(
                counter_set
            )
        return dd(lambda: (None, None), new_counter_set_vector)

    def apply_arc_to_counter_set_vector(
        self,
        counter_set_vector: CounterSetVector,
        arc: Arc,
    ) -> Optional[tuple[State, CounterSetVector]]:
        guard, action, target_state = arc
        evaluation = self.evaluate_guard(guard, counter_set_vector)

        if not evaluation:
            return None

        next_counter_set_vector = self.apply_action_to_counter_set_vector(
            counter_set_vector, action
        )
        return target_state, next_counter_set_vector

    def apply_action_to_counter_set_vector(
        self, counter_set_vector: CounterSetVector, action: Action
    ) -> CounterSetVector:
        new_counter_set_vector: CounterSetVector = dd(lambda: (None, None))
        for counter_variable in self.counters:
            operation = action.get(
                counter_variable, CounterOperationComponent.NO_OPERATION
            )
            counter_set = counter_set_vector[counter_variable]
            new_counter_set_vector[counter_variable] = (
                self.apply_operation_to_counter_set(counter_set, operation)
            )
        return new_counter_set_vector

    def apply_operation_to_counter_set(
        self,
        counter_set: CounterSet,
        operation: CounterOperationComponent,
    ) -> CounterSet:
        offset, ptr = counter_set
        if operation is CounterOperationComponent.NO_OPERATION:
            return self.copy_counter_set(counter_set)
        if operation is CounterOperationComponent.ACTIVATE_OR_RESET:
            return 1, None
        if operation is CounterOperationComponent.INCREASE:
            if offset is None:
                return offset, ptr
            offset, ptr = self.copy_counter_set(counter_set)
            assert offset is not None
            return offset + 1, ptr
        if operation is CounterOperationComponent.INACTIVATE:
            offset, ptr = self.copy_counter_set(counter_set)
            return None, ptr
        raise NotImplementedError()

    def evaluate_counter_set(
        self, counter_set: CounterSet
    ) -> list[Optional[int]]:
        offset, ptr = counter_set
        if offset is None:
            return [None]
        if ptr is None:
            return [offset]
        return [offset - value for value in self.access(ptr)]

    def evaluate_counter_set_vector(
        self, counter_set_vector: CounterSetVector
    ) -> list[tuple[Optional[int], ...]]:
        evaluated_counter_sets = [
            self.evaluate_counter_set(counter_set_vector[counter_variable])
            for counter_variable in self.counters
        ]
        return list(product(*evaluated_counter_sets))

    def json_reference_table(self) -> dict[int, list[int]]:
        return {ptr: list(ref) for ptr, ref in enumerate(self._reference_table)}

    def to_json(self) -> CounterCartesianSuperConfigJson:
        return CounterCartesianSuperConfigJson(
            counter_set_vectors=self.counter_set_vectors,
            reference_table=self.json_reference_table(),
            reference_counter=self._reference_counter,
        )

    def unfold(self) -> dict[int, list[tuple[Optional[int], ...]]]:
        return {
            state: self.evaluate_counter_set_vector(counter_set_vector)
            for state, counter_set_vector in self.counter_set_vectors.items()
        }

    def __str__(self) -> str:
        return json.dumps(self.to_json())
