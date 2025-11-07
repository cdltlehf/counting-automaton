"""Count the operations performed during matching and track the sizes and densities of counting-sets during merges and clones."""

import argparse
import logging
import re
import signal
import time
from typing import Type
from concurrent.futures import ThreadPoolExecutor
from cai4py.instrumentation.constants import THROUGHPUT_THRES

import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
import numpy as np
import pandas as pd
from cai4py.counting_automaton.instrumentation import (
    clone_set_sizes,
    merge_set_sizes,
    op_name_to_count,
)
from cai4py.counting_automaton._logging import VERBOSE
from tqdm import tqdm

from .constants import OP_NAMES


def timeout(seconds):
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()

        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = f(*args, **kwargs)
            finally:
                # reinstall the old signal handler
                signal.signal(signal.SIGALRM, old)
                # cancel the alarm
                # this line should be inside the "finally" block (per Sam Kortchmar)
                signal.alarm(0)
            return result

        new_f.__name__ = f.__name__
        return new_f

    return decorate


logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


@timeout(seconds=10)
def timed_automaton_construction(regex, args):
    return pca.PositionCountingAutomaton.create(
        regex, expansion_type=args.expansion_type
    )


def time_matching(
    sc_class,
    automaton: pca.PositionCountingAutomaton,
    random_str: str,
    op_counts: list[dict[str, int]],
    overall_merge_set_sizes: list[tuple[int, float, int, float]],
    overall_clone_set_sizes: list[tuple[int, float]],
    regex,
):
    t0 = time.perf_counter()
    # Step through matching
    c = None
    for c in sc_class.get_computation(automaton, random_str):
        pass  # do nothing
    assert c is not None

    t1 = time.perf_counter()
    duration = t1 - t0

    # Save the operation count in the list for this run
    op_counts.append(op_name_to_count.copy())
    # Save the merge and clone set sizes
    for (
        size1,
        density1,
        size2,
        density2,
    ) in merge_set_sizes:
        overall_merge_set_sizes.append((size1, density1, size2, density2))
    for size, density in clone_set_sizes:
        overall_clone_set_sizes.append((size, density))

    if c.is_final():
        assert re.fullmatch(regex, random_str) is not None
    else:
        assert re.fullmatch(regex, random_str) is None

    return duration


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

            try:
                automaton = timed_automaton_construction(regex, args)
            except NotImplementedError:
                continue
            except re.PatternError:
                continue
            except ValueError:
                continue
            except TimeoutError:
                continue

            for j in range(1, args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{args.random_string_dir}/{i}-{j}.txt",
                        "r",
                        encoding="utf-8",
                    ) as random_str_file:
                        # Initialize instrumentation variables
                        reset_instrumentation_variables()
                        random_str = random_str_file.read()

                        num_bytes = len(random_str.encode("utf-8"))
                        assert num_bytes >= 0

                        # Run matching with a timeout
                        try:
                            with ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(
                                    time_matching,
                                    sc_class,
                                    automaton,
                                    random_str,
                                    op_counts,
                                    overall_merge_set_sizes,
                                    overall_clone_set_sizes,
                                    regex,
                                )
                                timeout = num_bytes / THROUGHPUT_THRES + 1
                                _ = future.result(timeout=timeout)
                        except TimeoutError:
                            print("TIMEOUT")
                            break
                except FileNotFoundError:
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
