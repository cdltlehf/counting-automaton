"""Test"""

import counting_automaton.position_counting_automaton as pca
import parser_tools as pt
import utils as ut


def main() -> None:
    patterns = map(
        ut.unescape,
        open("./data/patterns/examples.txt", "r", encoding="utf-8").readlines(),
    )
    for pattern in patterns:
        print(f"Pattern: {pattern}")
        parsed = pt.parse(pattern)
        normalized = pt.normalize(parsed)
        pattern = pt.to_string(normalized)
        print(f"Normalized: {pattern}")
        automaton = pca.PositionCountingAutomaton.create(pattern)
        print(f"Automaton:\n{automaton}")

        w = "aaaaaaaaaaaa"
        print(f"String: {w}")

        super_configs = automaton.iterate_super_configs(w)
        for super_config in super_configs:
            print(f"Super config: {pca.super_config_to_json(super_config)}")
        match = automaton.match(w)
        print(f"Match: {match}")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    main()
