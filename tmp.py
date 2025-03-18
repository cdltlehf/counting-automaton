"""Test"""

# import logging

import io
import logging
from typing import Callable, cast, Iterable

from counting_automaton.logging import MatchingInfo
from counting_automaton.logging import VERBOSE
import counting_automaton.position_counting_automaton as pca
import counting_automaton.super_config as sc
import parser_tools as pt
import utils as ut


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # return record.levelno == logging.INFO
        return record.levelno == VERBOSE


def collect_matching_info(
    automaton: pca.PositionCountingAutomaton,
    w: str,
    get_computation: Callable[
        [pca.PositionCountingAutomaton, str], Iterable[sc.SuperConfigBase]
    ],
) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)

    logger = logging.getLogger()
    logger.setLevel(VERBOSE)
    logger.addFilter(VerboseFilter())
    logger.handlers.clear()
    logger.addHandler(handler)
    matching_info = {
        label.value: 0 for label in MatchingInfo.__members__.values()
    }
    for _ in get_computation(automaton, w):
        for msg in stream.readlines():
            if msg in matching_info:
                label = cast(MatchingInfo, msg)
                matching_info[label] += 1
            else:
                raise ValueError(f"Invalid matching info: {msg}")
    print(matching_info)


def main() -> None:
    patterns = map(
        ut.unescape,
        open("./data/patterns/examples.txt", "r", encoding="utf-8").readlines(),
    )
    for pattern in patterns:
        print(f"Pattern: {pattern}")
        parsed = pt.parse(pattern)
        # print(f"Parsed: {parsed}")
        normalized = pt.normalize(parsed)
        pattern = pt.to_string(normalized)
        # print(f"Normalized: {pattern}")
        automaton = pca.PositionCountingAutomaton.create(pattern)

        # print("Automaton:")
        # print(automaton)

        for n in range(1, 10):
            w = "a" * n
            print(f"String: {w}")
            get_computations = {
                "Super-config": sc.SuperConfig.get_computation,
                "Counter-config": sc.CounterConfig.get_computation,
                "Sparse-counter-config": sc.SparseCounterConfig.get_computation,
            }

            for label, get_computation in get_computations.items():
                print(f"Computation: {label}")
                collect_matching_info(automaton, w, get_computation)


if __name__ == "__main__":
    main()
