"""Analysis pattern."""

import csv
from itertools import chain
import math
from math import ceil
import sys
from typing import Any, Callable, Iterable, Optional, Union

import src.parser_tools as pt
from src.parser_tools.constants import ANY
from src.parser_tools.constants import BRANCH
from src.parser_tools.constants import IN
from src.parser_tools.constants import LITERAL
from src.parser_tools.constants import MAX_REPEAT
from src.parser_tools.constants import MAXREPEAT
from src.parser_tools.constants import MIN_REPEAT
from src.parser_tools.constants import NamedIntConstant
from src.parser_tools.constants import NOT_LITERAL
from src.parser_tools.constants import SUBPATTERN
from src.utils import escape
from src.utils import unescape

Ordinal = Union[int, float]

def printerr(*args: Any) -> None:
    print(*args, file=sys.stderr)


def has_nullable_non_trivial_counter(tree: pt.SubPattern) -> bool:
    def f(
        x: Optional[tuple[NamedIntConstant, Any]],
        ys: Iterable[tuple[bool, bool]]
    ) -> tuple[bool, bool]:
        """
        Return: [is_nullable, has_nullable_non_trivial_counter]
        """
        is_nullable: bool
        ys = list(ys)
        ret: bool = any(y[1] for y in ys)

        if x is None:
            is_nullable = all(y[0] for y in ys)
        else:
            opcode, operand = x
            if opcode in {LITERAL, ANY, NOT_LITERAL, IN}:
                is_nullable = False
            elif opcode is BRANCH:
                is_nullable = any(y[0] for y in ys)
            elif opcode in {MAX_REPEAT, MIN_REPEAT}:
                assert len(ys) == 1
                y = ys[0]
                if y[0] and not is_trivial(operand):
                    ret = True
                is_nullable = y[0] or operand[0] == 0

            elif opcode is SUBPATTERN:
                is_nullable = all(y[0] for y in ys)
            else:
                assert False, f"Unknown opcode: {opcode}"

        return is_nullable, ret

    return pt.fold(f, tree)[1]


def get_counter_ranges(tree: pt.SubPattern) -> list[tuple[int, Ordinal]]:
    def f(
        x: Optional[tuple[NamedIntConstant, Any]],
        ys: Iterable[Iterable[tuple[int, Ordinal]]],
    ) -> Iterable[tuple[int, Ordinal]]:

        ret = chain.from_iterable(ys)
        if x is not None:
            opcode, operand = x
            if opcode in {MAX_REPEAT, MIN_REPEAT}:
                m, n = operand
                if n is MAXREPEAT:
                    n = math.inf
                return chain([(m, n)], ret)
        return ret

    return list(pt.fold(f, tree))


def is_trivial(counter: tuple[int, Ordinal]) -> bool:
    if counter[0] == 0 and counter[1] == math.inf:
        return True
    if counter[0] == 1 and counter[1] == math.inf:
        return True
    if counter[0] == 0 and counter[1] == 1:
        return True
    return False


def is_half_bounded(counter: tuple[int, Ordinal]) -> bool:
    if counter[0] == 0 or counter[1] == math.inf:
        return True
    return False


def get_max_lower_bound(counters: list[tuple[int, Ordinal]]) -> int:
    assert counters
    return max(lower_bound for lower_bound, _ in counters)


def get_max_upper_bound(
    counters: list[tuple[int, Ordinal]]
) -> Ordinal:
    assert counters
    max_upper_bound: Ordinal = -1
    for _, upper_bound in counters:
        if upper_bound < math.inf:
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
        else:
            return math.inf
    return max_upper_bound


def get_max_counter_bound(
    counters: list[tuple[int, Ordinal]]
) -> Ordinal:
    assert counters

    def get_counter_bound(counter: tuple[int, Ordinal]) -> Ordinal:
        return counter[1] if counter[1] < math.inf else counter[0]

    max_counter_bound: Ordinal = -1
    for counter in counters:
        counter_bound = get_counter_bound(counter)
        if counter_bound < math.inf:
            if counter_bound > max_counter_bound:
                max_counter_bound = counter_bound
        else:
            return math.inf
    return max_counter_bound

def get_max_counter_set_size(
    counters: list[tuple[int, Ordinal]]
) -> int:
    assert counters

    def get_counter_set_size(counter: tuple[int, Ordinal]) -> int:
        if counter[1] == math.inf or counter[0] == 0:
            return 1
        return ceil(counter[1] / (counter[1] - counter[0] + 1))

    return max(get_counter_set_size(counter) for counter in counters)


def to_histogram(
    values: Iterable[Ordinal],
    bins_num: int,
    bin_size: int,
) -> tuple[list[int], int]:
    bins = [0] * bins_num
    overflow = 0
    for value in values:
        bin_index = value // bin_size
        if bin_index < bins_num:
            assert isinstance(bin_index, int)
            bins[bin_index] += 1
        else:
            overflow += 1
    return bins, overflow


def printerr_histogram(xs: list[Ordinal], bins_num: int, bin_size: int) -> None:
    bins, overflow = to_histogram(xs, bins_num, bin_size)
    printerr(f"bin_size: {bin_size}, overflow: {bins_num * bin_size}")
    printerr(str(bins), overflow)

def main() -> None:
    non_trivial = 0
    nullable_counter = 0
    half_bounded = 0

    metric_to_functions: dict[
        str,
        Callable[[list[tuple[int, Ordinal]]], Ordinal]
    ] = {
        "lower_bound": get_max_lower_bound,
        "upper_bound": get_max_upper_bound,
        "counter_bound": get_max_counter_bound,
        "counter_set_size": get_max_counter_set_size
    }

    fieldnames = (
        ["pattern", "nullable", "half_bounded"] + list(metric_to_functions)
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    metric_to_values: dict[str, list[Ordinal]] = {
        metric: [] for metric in metric_to_functions
    }

    for pattern in map(unescape, sys.stdin):
        parsed = pt.parse(pattern)
        counters = get_counter_ranges(parsed)

        if all(is_trivial(counter) for counter in counters):
            continue

        jsonobj: dict[str, Any] = { "pattern": escape(pattern) }
        non_trivial += 1
        if has_nullable_non_trivial_counter(parsed):
            jsonobj["nullable"] = True
            nullable_counter += 1
        else:
            jsonobj["nullable"] = False

        if all(is_half_bounded(counter) for counter in counters):
            jsonobj["half_bounded"] = True
            half_bounded += 1
        else:
            jsonobj["half_bounded"] = False

        for metric, func in metric_to_functions.items():
            value = func(counters)
            jsonobj[metric] = value
            metric_to_values[metric].append(value)

        writer.writerow(jsonobj)

    printerr(f"Non-trivial counter: {non_trivial}")
    printerr(f"Nullable non-trivial counter: {nullable_counter}")
    printerr(f"Half-bounded counter: {half_bounded}")
    for metric, values in metric_to_values.items():
        print(metric, end=": ", file=sys.stderr)
        printerr_histogram(values, 8, 8)
        printerr()

if __name__ == "__main__":
    main()
