"""Time matching with the position counting automaton and attack strings"""

import sys
import time
import argparse
import logging
from typing import Type
from cai4py.counting_automaton.logging import VERBOSE
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def main(args: argparse.Namespace) -> None:
    method: str = args.method
    sc_class: Type[sc.SuperConfigBase] = {
        "super_config": sc.SuperConfig,
        "bounded_super_config": sc.BoundedSuperConfig,
        "counter_config": sc.CounterConfig,
        "bounded_counter_config": sc.BoundedCounterConfig,
        "sparse_counter_config": sc.SparseCounterConfig,
        "determinized_counter_config": sc.DeterminizedCounterConfig,
        "determinized_bounded_counter_config": sc.DeterminizedBoundedCounterConfig,
        "determinized_sparse_counter_config": sc.DeterminizedSparseCounterConfig,
    }[method]
    automaton = pca.PositionCountingAutomaton.create(
        args.regex, expansion_type=args.expansion_type
    )
    t0 = time.perf_counter()
    # Step through matching
    computation = None
    with open("temp-op-log.txt", "w", encoding="utf-8") as log_file:
        for c in sc_class.get_computation(automaton, args.input_string):
            print(c.is_final())
            print(c)
            computation = c
        assert computation is not None

    t1 = time.perf_counter()
    is_match = computation.is_final()
    duration = t1 - t0
    num_bytes = len(args.input_string.encode("latin1"))
    print(f"match: {is_match}\nthroughput: {duration * 1000 / num_bytes}\n")


if __name__ == "__main__":
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default="sparse_counter_config",
        choices=[
            "super_config",
            "bounded_super_config",
            "counter_config",
            "bounded_counter_config",
            "sparse_counter_config",
            "determinized_counter_config",
            "determinized_bounded_counter_config",
            "determinized_sparse_counter_config",
        ],
    )
    parser.add_argument("--input-string", required=True, type=str)
    parser.add_argument("--regex", required=True, type=str)
    parser.add_argument(
        "--expansion-type", required=True, type=str, choices=["inner", "outer"]
    )
    main(parser.parse_args())
