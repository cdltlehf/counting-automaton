"""Config Shape"""

import json
import sys
from typing import Optional

import timeout_decorator  # type: ignore

from src.position_counter_automaton import PositionCounterAutomaton
from src.utils import read_test_cases

TargetJson = list[list[tuple[int, dict[int, int]]]]


def analyze(
    automaton: PositionCounterAutomaton, text: str, second: int
) -> Optional[TargetJson]:
    @timeout_decorator.timeout(second)  # type: ignore
    def _analyze(automaton: PositionCounterAutomaton, text: str) -> TargetJson:
        return [
            [
                (state, {k: v for k, v in counter_vector.items()})
                for state, counter_vector in super_config
            ]
            for super_config in automaton.iterate_super_configs(text)
        ]

    try:
        return _analyze(automaton, text)  # type: ignore
    except timeout_decorator.TimeoutError:
        return None


def main() -> None:
    for pattern, texts in read_test_cases(sys.stdin):
        automaton = PositionCounterAutomaton.create(pattern)
        if not automaton.counters:
            continue

        sequences = [analyze(automaton, text, 5) for text in texts]
        if any(sequence is None for sequence in sequences):
            continue

        output = {}
        output["pattern"] = pattern
        output["sequences"] = sequences  # type: ignore
        json.dump(output, sys.stdout)
        print()


if __name__ == "__main__":
    main()
