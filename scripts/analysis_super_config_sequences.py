"""Analysis super config."""

from collections import dd
from functools import reduce
import json
import operator
import sys
from typing import Any, Callable, Iterable

from utils import escape

State = int
CounterVector = tuple[int, ...]
SuperConfig = list[tuple[State, CounterVector]]


def to_state_to_counter_vector_set(
    super_config: SuperConfig,
) -> dict[State, set[CounterVector]]:
    state_to_counter_vector_set: dd[State, set[CounterVector]]
    state_to_counter_vector_set = dd(set)
    for state, counter_vector in super_config:
        state_to_counter_vector_set[state].add(counter_vector)
    return dict(state_to_counter_vector_set)


def to_cartesian_product(
    counter_vector_set: set[CounterVector],
) -> list[set[int]]:
    max_counter_variable = max(map(len, counter_vector_set))
    counter_to_value_set: list[set[int]] = [set() for _ in range(max_counter_variable)]
    for counter_vector in counter_vector_set:
        for i, value in enumerate(counter_vector):
            if value != 0:
                counter_to_value_set[i].add(value)
    return counter_to_value_set


def is_cartesian_counter_vector_set(
    counter_vector_set: set[CounterVector],
) -> bool:
    cartesian_product = to_cartesian_product(counter_vector_set)
    size = reduce(operator.mul, map(len, cartesian_product), 1)
    return size == len(counter_vector_set)


def is_strong_cartesian(super_config: SuperConfig) -> bool:
    # q |-> A_q
    state_to_counter_vector_set = to_state_to_counter_vector_set(super_config)

    # c |-> { a(c) | a in A_q for some q}
    cartesian_products = {}
    for state, counter_vector_set in state_to_counter_vector_set.items():
        if not is_cartesian_counter_vector_set(counter_vector_set):
            return False
        cartesian_products[state] = to_cartesian_product(counter_vector_set)

    if not cartesian_products:
        return True

    max_counter_variable = max(map(len, cartesian_products.values()))
    for counter_variable in range(max_counter_variable):
        value_set: set[int] = set()
        for state, cartesian_product in cartesian_products.items():
            if counter_variable >= len(cartesian_product):
                continue
            new_value_set = cartesian_product[counter_variable]
            if not new_value_set:
                continue
            if not value_set:
                value_set = new_value_set
                continue
            if value_set != new_value_set:
                return False

    return True


def is_weak_cartesian(super_config: SuperConfig) -> bool:
    counter_vector_sets = to_state_to_counter_vector_set(super_config).values()
    return all(map(is_cartesian_counter_vector_set, counter_vector_sets))


def is_boxed_counter_vector_set(counter_vector_set: set[CounterVector]) -> bool:
    cartesian_product = to_cartesian_product(counter_vector_set)

    size = reduce(operator.mul, map(len, cartesian_product), 1)
    if size != len(counter_vector_set):
        return False

    for value_set in cartesian_product:
        if max(value_set) - min(value_set) != len(value_set) - 1:
            return False
    return True


def is_counter_boxed(super_config: SuperConfig) -> bool:
    counter_vector_sets = to_state_to_counter_vector_set(super_config).values()
    return all(map(is_boxed_counter_vector_set, counter_vector_sets))


def is_single_counter_active(super_config: SuperConfig) -> bool:
    state_to_counter_vector_set = to_state_to_counter_vector_set(super_config)
    for counter_vector_set in state_to_counter_vector_set.values():
        for counter_vector in counter_vector_set:
            if sum(map(bool, counter_vector)) > 1:
                return False
    return True


def is_counter_deterministic(super_config: SuperConfig) -> bool:
    # There is at most one counter vector for each state
    state_to_counter_vector_set = to_state_to_counter_vector_set(super_config)
    for counter_vector_set in state_to_counter_vector_set.values():
        if len(counter_vector_set) > 1:
            return False
    return True


def is_deterministic(super_config: SuperConfig) -> bool:
    # Size of super config is 0 or 1
    return len(super_config) <= 1


def evaluate_sequence(
    callback: Callable[[SuperConfig], bool],
    sequence: list[SuperConfig],
) -> bool:
    return all(map(callback, sequence))


def evaluate_sequences(
    callback: Callable[[SuperConfig], bool],
    sequences: list[list[SuperConfig]],
) -> bool:
    return all(evaluate_sequence(callback, sequence) for sequence in sequences)


def to_super_config(configs: Iterable[Any]) -> SuperConfig:
    def load_counter_vector(counter_vector_like: Any) -> CounterVector:
        if len(counter_vector_like) == 0:
            return ()
        counter_variables = max(map(int, counter_vector_like.keys()))
        counter_vector = [0] * (counter_variables + 1)
        for counter, value in counter_vector_like.items():
            counter_vector[int(counter)] = value
        return tuple(counter_vector)

    super_config = [
        (state, load_counter_vector(counter_vector))
        for state, counter_vector in configs
    ]
    return super_config


def read_sequence(sequence_like: list[Any]) -> list[SuperConfig]:
    return [to_super_config(super_config) for super_config in sequence_like]


def read_sequences(sequences_like: list[list[Any]]) -> list[list[SuperConfig]]:
    return [read_sequence(sequence) for sequence in sequences_like]


def main() -> None:
    properties = {
        "total": lambda _: True,
        "deterministic": is_deterministic,
        "strong_cartesian": is_strong_cartesian,
        "weak_cartesian": is_weak_cartesian,
        "counter_boxed": is_counter_boxed,
        "counter_deterministic": is_counter_deterministic,
        "single_counter_active": is_single_counter_active,
    }

    counts = {key: 0 for key in properties}
    for analysis in map(json.loads, map(str.strip, sys.stdin)):
        pattern = analysis["pattern"]
        sequences = read_sequences(analysis["sequences"])
        for key, callback in properties.items():
            if evaluate_sequences(callback, sequences):
                counts[key] += 1
            else:
                print(f"{pattern} is not {key}", file=sys.stderr)

    for key, count in counts.items():
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()
