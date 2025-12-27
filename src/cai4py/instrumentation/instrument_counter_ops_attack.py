"""
Instrument counter operations using EvilStrGen-generated attack strings.
Reads one attack string per regex ID and records operation counts,
merge-set sizes, and clone-set sizes during matching.
"""

import argparse
from collections import namedtuple
import logging
import re
import sys
from typing import Type

from cai4py.custom_counters.counter_type import CounterType
import numpy as np
import pandas as pd
from tqdm import tqdm

from cai4py.counting_automaton._logging import VERBOSE
from cai4py.counting_automaton.fullmatch import fullmatch
from cai4py.counting_automaton.instrumentation_vars import clone_set_sizes
from cai4py.counting_automaton.instrumentation_vars import merge_set_sizes
from cai4py.counting_automaton.instrumentation_vars import op_name_to_count
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from cai4py.instrumentation.utils import get_matching_timeout
from cai4py.instrumentation.utils import run_with_timeout
from cai4py.instrumentation.utils import add_common_arguments

from .constants import OP_NAMES

logger = logging.getLogger(__name__)

# Define namedtuples for merge and clone set sizes
MergeSetSize = namedtuple("MergeSetSize", "size1 density1 size2 density2")
CloneSetSize = namedtuple("CloneSetSize", "size density")


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def instrument_matching(
    sc_class,
    automaton: pca.PositionCountingAutomaton,
    attack_str: str,
    op_counts: list[dict[str, int]],
    overall_merge_set_sizes: list[tuple[int, float, int, float]],
    overall_clone_set_sizes: list[tuple[int, float]],
):
    fullmatch(sc_class, automaton, attack_str, CounterType.COUNTING_SET, "none")
    # Save the operation count in the list for this run
    assert len(op_name_to_count) == len(
        OP_NAMES
    ), "Operation count dictionary has unexpected number of entries"
    assert all(
        isinstance(v, int) for v in op_name_to_count.values()
    ), "All operation counts should be integers"
    op_counts.append(op_name_to_count.copy())
    # Save the merge and clone set sizes
    for (
        size1,
        density1,
        size2,
        density2,
    ) in merge_set_sizes:
        assert size1 >= 0
        assert size2 >= 0
        assert 0.0 <= density1 <= 1.0
        assert 0.0 <= density2 <= 1.0
        overall_merge_set_sizes.append(
            MergeSetSize(size1, density1, size2, density2)
        )
    for size, density in clone_set_sizes:
        assert size >= 0
        assert 0.0 <= density <= 1.0
        overall_clone_set_sizes.append(CloneSetSize(size, density))
    return (
        op_counts,
        overall_merge_set_sizes,
        overall_clone_set_sizes,
    )


def reset_instrumentation_variables():
    for op in OP_NAMES:
        op_name_to_count[op] = 0
    merge_set_sizes.clear()
    clone_set_sizes.clear()


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
    op_counts: list[dict[str, int]] = []
    overall_merge_set_sizes = []
    overall_clone_set_sizes = []

    # Read regexes
    with open(args.regex_file, "r", encoding=args.input_encoding) as regex_file:
        num_regexes = len(regex_file.readlines())
    with open(args.regex_file, "r", encoding=args.input_encoding) as regex_file:
        for i, regex in enumerate(
            tqdm(
                regex_file,
                total=num_regexes,
                miniters=1,
                mininterval=0,
            ),
            start=1,
        ):
            regex = regex[:-1]  # Remove newline
            print(f"Processing regex {i}: {regex}")
            try:
                automaton = run_with_timeout(
                    func=pca.PositionCountingAutomaton.create,
                    args=(regex, args.expansion_type),
                    timeout=10,
                )
                assert isinstance(automaton, pca.PositionCountingAutomaton)
                if automaton is None:
                    raise RuntimeError("Automaton creation failed")
            except TimeoutError as e:
                print(e, file=sys.stderr)
                continue
            except NotImplementedError as e:
                print(e, file=sys.stderr)
                continue
            except re.PatternError as e:
                print(e, file=sys.stderr)
                continue
            except ValueError as e:
                print(e, file=sys.stderr)
                continue

            attack_path = f"{args.attack_string_dir}/{i}.txt"
            try:
                with open(
                    attack_path, "r", encoding=args.input_encoding
                ) as attack_str_file:
                    reset_instrumentation_variables()
                    attack_str = attack_str_file.read()
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                continue

            num_bytes = len(attack_str.encode(args.input_encoding))
            assert num_bytes >= 0

            # Run matching with a timeout
            matching_timeout = get_matching_timeout(num_bytes)
            try:
                (
                    op_counts,
                    overall_merge_set_sizes,
                    overall_clone_set_sizes,
                ) = run_with_timeout(
                    instrument_matching,
                    args=(
                        sc_class,
                        automaton,
                        attack_str,
                        op_counts,
                        overall_merge_set_sizes,
                        overall_clone_set_sizes,
                    ),
                    timeout=matching_timeout,
                )  # type: ignore
            except TimeoutError as e:
                print(e, file=sys.stderr)
                continue

    # Persist outputs
    pd.DataFrame(data=op_counts, columns=OP_NAMES).to_csv(
        args.op_counts_output, index=False
    )
    np.save(args.merge_sizes_output, overall_merge_set_sizes)
    np.save(args.clone_sizes_output, overall_clone_set_sizes)


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
        help="Configuration method to use",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--attack-string-dir",
        required=True,
        type=str,
        help="Directory containing EvilStrGen attack strings: {regex_id}.txt",
    )
    parser.add_argument(
        "--op-counts-output",
        required=True,
        type=str,
        help="Output CSV file to save operation counts.",
    )
    parser.add_argument(
        "--merge-sizes-output",
        required=True,
        type=str,
        help="Output .npy file to save MERGE sizes.",
    )
    parser.add_argument(
        "--clone-sizes-output",
        required=True,
        type=str,
        help="Output .npy file to save CLONE sizes.",
    )
    main(parser.parse_args())
