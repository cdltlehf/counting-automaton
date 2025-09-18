import argparse
import logging

import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from cai4py.parser_tools import parse
from cai4py.parser_tools import to_string
from cai4py.parser_tools.utils import flatten_inner_quantifiers
from cai4py.parser_tools.utils import flatten_quantifiers


def main() -> None:
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", type=str)
    parser.add_argument("text", type=str)
    args = parser.parse_args()

    method_to_get_computation = {
        "super_config": sc.SuperConfig.get_computation,
        "bounded_super_config": sc.BoundedSuperConfig.get_computation,
        "counter_config": sc.CounterConfig.get_computation,
        "bounded_counter_config": sc.BoundedCounterConfig.get_computation,
        "sparse_counter_config": sc.SparseCounterConfig.get_computation,
        "determinized_counter_config": (
            sc.DeterminizedCounterConfig.get_computation
        ),
        "determinized_bounded_counter_config": (
            sc.DeterminizedBoundedCounterConfig.get_computation
        ),
        "determinized_sparse_counter_config": (
            sc.DeterminizedSparseCounterConfig.get_computation
        ),
        # NOTE: Counter expansion
        "counter_expansion": sc.SuperConfig.get_computation,
    }
    for label, get_computation in method_to_get_computation.items():
        print(label)
        if label == "counter_expansion":
            normalized_pattern = to_string(
                flatten_quantifiers(parse(args.pattern))
            )
        else:
            normalized_pattern = to_string(
                flatten_inner_quantifiers(parse(args.pattern))
            )
        print(normalized_pattern)
        automaton = pca.PositionCountingAutomaton.create(normalized_pattern)
        print(str(automaton))
        for config in get_computation(automaton, args.text):
            print(str(config), config.is_final())


if __name__ == "__main__":
    main()
