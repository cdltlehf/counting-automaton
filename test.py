"""Test"""

# import logging

import logging

import counting_automaton.position_counting_automaton as pca
import counting_automaton.super_config as sc
import parser_tools as pt
import utils as ut


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    # logging.basicConfig(level=logging.WARNING)
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

        for n in range(1, 10):
            w = "a" * n
            print(f"String: {w}")

            super_config_computations = {
                "Super-config": sc.SuperConfig.get_computation(automaton, w),
                "Counter-config": sc.CounterConfig.get_computation(
                    automaton, w
                ),
                "Sparse-counter-config": sc.SparseCounterConfig.get_computation(
                    automaton, w
                ),
            }

            print("Automaton:")
            print(automaton)
            for super_config, counter_config, sparse_counter_config in zip(
                *super_config_computations.values()
            ):
                print(f"Super-config: {super_config}")
                print(f"Counter-config: {counter_config}")
                print(f"Sparse-Counter-config: {sparse_counter_config}")

            # for (
            #     config_type,
            #     super_config_computation,
            # ) in super_config_computations.items():
            #     last_super_config = None
            #     logging.debug("Automaton:\n%s", automaton)

            #     for super_config in super_config_computation:
            #         print(f"{config_type}: {super_config}")
            #         last_super_config = super_config

            #     assert last_super_config is not None
            #     match = last_super_config.is_final()
            #     if automaton.is_flat():
            #         backtracking_match = automaton(w)
            #         assert match == backtracking_match, (
            #             f"Match: {match} != {backtracking_match}"
            #         )
            #     print(f"Match: {automaton(w)}")


if __name__ == "__main__":
    main()
