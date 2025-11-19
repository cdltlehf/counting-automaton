"""
Count the operations performed during matching and track the sizes
and densities counting-sets during merges and clones.
"""

import argparse
from collections import namedtuple
import logging
import re
import sys
from typing import Type

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
    random_str: str,
    op_counts: list[dict[str, int]],
    overall_merge_set_sizes: list[tuple[int, float, int, float]],
    overall_clone_set_sizes: list[tuple[int, float]],
):
    fullmatch(sc_class, automaton, random_str, "none")
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
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        num_regexes = len(regex_file.readlines())
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
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

            for j in range(1, args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{args.random_string_dir}/{i}/{j}.txt",
                        "r",
                        encoding="utf-8",
                    ) as random_str_file:
                        # Initialize instrumentation variables
                        reset_instrumentation_variables()
                        random_str = random_str_file.read()
                except FileNotFoundError as e:
                    print(e, file=sys.stderr)
                    continue

                num_bytes = len(random_str.encode("utf-8"))
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
                            random_str,
                            op_counts,
                            overall_merge_set_sizes,
                            overall_clone_set_sizes,
                        ),
                        timeout=matching_timeout,
                    )  # type: ignore
                except TimeoutError as e:
                    print(e, file=sys.stderr)
                    break
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
    )
    parser.add_argument("--random-string-dir", required=True, type=str)
    parser.add_argument("--regex-file", required=True, type=str)
    parser.add_argument("--num-strings-per-regex", required=True, type=int)
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
    parser.add_argument(
        "--expansion-type",
        required=True,
        type=str,
        choices=["inner", "outer", "all"],
    )
    main(parser.parse_args())
