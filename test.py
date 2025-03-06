"""Test"""

# import logging

import counting_automaton.position_counting_automaton as pca
import counting_automaton.super_config as sc
import parser_tools as pt
import utils as ut


def main() -> None:
    # logging.basicConfig(level=logging.DEBUG)
    patterns = map(
        ut.unescape,
        open("./data/patterns/examples.txt", "r", encoding="utf-8").readlines(),
    )
    for pattern in patterns:
        print(f"Pattern: {pattern}")
        parsed = pt.parse(pattern)  # type: ignore
        normalized = pt.normalize(parsed)
        pattern = pt.to_string(normalized)
        print(f"Normalized: {pattern}")
        automaton = pca.PositionCountingAutomaton.create(pattern)
        print(f"Automaton:\n{automaton}")

        w = "aaaaaaaaaaaa"
        print(f"String: {w}")

        computation = sc.SuperConfig.get_computation(automaton, w)
        last_super_config = None
        for super_config in computation:
            print(f"Super config: {super_config}")
            last_super_config = super_config
        assert last_super_config is not None
        match = last_super_config.is_final()
        assert match == automaton(w)


if __name__ == "__main__":
    main()
