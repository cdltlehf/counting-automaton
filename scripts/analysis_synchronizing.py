"""Check finitely synchronizing"""

from collections import dd as dd
from enum import Enum
from itertools import tee
from json import dumps
import string
import sys
from typing import Any, Iterable, Optional, TypedDict

import timeout_decorator as td  # type: ignore

from src.counter_vector import CounterVector
from src.more_collections import OrderedSet
from src.parser_tools import fold
from src.parser_tools import MAX_REPEAT
from src.parser_tools import MAXREPEAT
from src.parser_tools import MIN_REPEAT
from src.parser_tools import NamedIntConstant
from src.parser_tools import parse
from src.parser_tools import SubPattern
from src.parser_tools import to_string
from src.parser_tools import to_string_f
from src.position_counter_automaton import PositionCounterAutomaton
from src.position_counter_automaton import super_config_to_json
from src.position_counter_automaton import SuperConfig
from src.utils import unescape


class NestedRepeatsError(Exception):
    pass


class SynchronizingError(Exception):
    pass


def printerr(*args: Any) -> None:
    print(*args, file=sys.stderr)


def get_maximum_synchronizing_gap(super_config: SuperConfig) -> int:
    counter_variable = 0
    maximum_synchronizing_gap = 0
    for state in super_config:
        if state == -1:
            continue
        counter_vectors = super_config[state]
        for counter_vector in counter_vectors:
            assert len(counter_vector) == 1
            counter_value = counter_vector[counter_variable]
            maximum_synchronizing_gap = max(
                maximum_synchronizing_gap, counter_value - 1
            )
    return maximum_synchronizing_gap


def normalize_super_config(super_config: SuperConfig) -> SuperConfig:
    normalized_super_config: SuperConfig = dd()

    minimum_counter_value = None
    counter_variable = 0
    counter_upper_bound = None

    for state in super_config:
        if state == -1:
            continue
        counter_vectors = super_config[state]
        for counter_vector in counter_vectors:
            counter_value = counter_vector[counter_variable]
            counter_upper_bound = counter_vector.upper_bound(counter_variable)
            if minimum_counter_value is None:
                minimum_counter_value = counter_value
            else:
                minimum_counter_value = min(minimum_counter_value, counter_value)

    for state in super_config:
        if state == -1:
            normalized_super_config[state] = super_config[state]
            continue

        assert counter_variable is not None
        assert minimum_counter_value is not None
        assert counter_upper_bound is not None

        counter_vectors = super_config[state]
        normalized_counter_vectors: OrderedSet[CounterVector] = OrderedSet()
        for counter_vector in counter_vectors:
            counter_value = counter_vector[counter_variable]
            normalized_counter_vector = CounterVector(
                {counter_variable: counter_upper_bound}
            )
            normalized_counter_vector[counter_variable] = (
                counter_value - minimum_counter_value + 1
            )
            normalized_counter_vectors.append(normalized_counter_vector)
        normalized_super_config[state] = normalized_counter_vectors
    return normalized_super_config


def hash_super_config(super_config: SuperConfig) -> int:
    return hash(dumps(super_config_to_json(super_config)))


def collect_deterministic_counter_configs(
    automaton: PositionCounterAutomaton, k: int
) -> tuple[list[SuperConfig], int]:
    printable = string.printable

    super_configs: list[SuperConfig] = []
    hashed_super_configs: set[int] = set()

    initial_super_config = automaton.get_initial_super_config()
    super_configs.append(initial_super_config)
    hashed_super_configs.add(hash_super_config(initial_super_config))

    current_super_configs: list[SuperConfig] = []
    current_super_configs.append(initial_super_config)
    maximum_synchronizing_gap = 0
    while True:
        next_super_configs: list[SuperConfig] = []
        for super_config in current_super_configs:
            for symbol in printable:
                next_super_config = automaton.get_next_super_config(
                    super_config, symbol
                )
                next_super_config = normalize_super_config(next_super_config)
                maximum_synchronizing_gap = max(
                    maximum_synchronizing_gap,
                    get_maximum_synchronizing_gap(next_super_config),
                )
                if maximum_synchronizing_gap > k:
                    raise SynchronizingError(
                        f"Maximum synchronizing gap is larger than {k}"
                    )
                hashed_super_config = hash_super_config(next_super_config)
                if hashed_super_config not in hashed_super_configs:
                    next_super_configs.append(next_super_config)
                    super_configs.append(next_super_config)
                    hashed_super_configs.add(hashed_super_config)

        if not next_super_configs:
            break
        current_super_configs = next_super_configs

    return super_configs, maximum_synchronizing_gap


def collect_repeats(
    tree: SubPattern,
) -> set[tuple[SubPattern, int, Optional[int]]]:
    def f(
        x: Optional[tuple[NamedIntConstant, Any]],
        ys: Iterable[tuple[set[tuple[str, int, Optional[int]]], str]],
    ) -> tuple[set[tuple[str, int, Optional[int]]], str]:
        ys1, ys2 = tee(ys)
        sets = [y[0] for y in ys1]
        strings = [y[1] for y in ys2]
        stringified = to_string_f(x, strings)

        if x is None:
            return set.union(set(), *sets), stringified
        opcode, operand = x
        if opcode in {MIN_REPEAT, MAX_REPEAT}:
            m, n = operand

            # Trivial counter
            if m in {0, 1} and n in {0, 1, MAXREPEAT}:
                return set.union(set()), stringified

            if any(len(s) != 0 for s in sets):
                raise NestedRepeatsError("Nested repeats")
            return (
                {(stringified, m, n if isinstance(n, int) else None)},
                stringified,
            )
        else:
            return set.union(set(), *sets), stringified

    return {(parse(pattern), m, n) for pattern, m, n in fold(f, tree)[0]}


class ErrorCode(Enum):
    UNKNOWN = 0
    EXTENDED_FEATURES = 1
    NESTED_REPEATS = 2
    SYNCHRONIZING_ERROR = 3
    TIMEOUT_ERROR = 4


class SubPatternResult(TypedDict):
    pattern: str
    lower_bound: int
    upper_bound: Optional[int]
    maximum_synchronizing_gap: int
    number_of_super_configs: int
    error: Optional[tuple[int, str]]


class Result(TypedDict):
    pattern: str
    results: list[SubPatternResult]
    error: Optional[tuple[int, str]]


def main() -> None:
    k = 30
    for pattern in map(unescape, sys.stdin):
        result = Result(pattern=pattern, results=[], error=None)
        results = result["results"]
        try:
            subpatterns = collect_repeats(parse(pattern))
        except NestedRepeatsError as e:
            result["error"] = (ErrorCode.NESTED_REPEATS.value, str(e))
        except NotImplementedError as e:
            result["error"] = (ErrorCode.EXTENDED_FEATURES.value, str(e))
        except Exception as e:  # pylint: disable=broad-except
            result["error"] = (ErrorCode.UNKNOWN.value, str(e))
            printerr(e)

        for subpattern, m, n in subpatterns:
            subresult = SubPatternResult(
                pattern="",
                lower_bound=m,
                upper_bound=n,
                maximum_synchronizing_gap=-1,
                number_of_super_configs=-1,
                error=None,
            )
            try:
                subpattern_string = to_string(subpattern)
                automaton = PositionCounterAutomaton.create(subpattern_string)
                super_configs, maximum_synchronizing_gap = td.timeout(10)(
                    collect_deterministic_counter_configs
                )(automaton, k)
                subresult["pattern"] = subpattern_string
                subresult["maximum_synchronizing_gap"] = maximum_synchronizing_gap
                subresult["number_of_super_configs"] = len(super_configs)
            except NestedRepeatsError as e:
                result["error"] = (ErrorCode.NESTED_REPEATS.value, str(e))
            except SynchronizingError as e:
                result["error"] = (ErrorCode.SYNCHRONIZING_ERROR.value, str(e))
            except td.TimeoutError as e:
                result["error"] = (ErrorCode.TIMEOUT_ERROR.value, str(e))
            except Exception as e:  # pylint: disable=broad-except
                result["error"] = (ErrorCode.UNKNOWN.value, str(e))
                printerr(e)
            results.append(subresult)

        print(dumps(result))


if __name__ == "__main__":
    main()
