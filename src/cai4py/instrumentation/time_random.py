"""Time matching with the position counting automaton and random strings"""

from cai4py.instrumentation.constants import THROUGHPUT_THRES

import argparse
import logging
import re
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Type

import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from cai4py.counting_automaton._logging import VERBOSE
from tqdm import tqdm

from cai4py.instrumentation.utils import (
    time_matching,
    timed_automaton_construction,
)

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def main(args: argparse.Namespace) -> None:
    method: str = args.method
    sc_class: sc.SuperConfigBase = {
        "super_config": sc.SuperConfig,
        "bounded_super_config": sc.BoundedSuperConfig,
        "counter_config": sc.CounterConfig,
        "bounded_counter_config": sc.BoundedCounterConfig,
        "sparse_counter_config": sc.SparseCounterConfig,
        "determinized_counter_config": sc.DeterminizedCounterConfig,
        "determinized_bounded_counter_config": sc.DeterminizedBoundedCounterConfig,
        "determinized_sparse_counter_config": sc.DeterminizedSparseCounterConfig,
    }[method]
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        num_regexes = len(regex_file.readlines())
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        timing_log_file = open(args.timing_log_file, "w", encoding="utf-8")
        timing_log_file.write("Regex ID\tMean throughput (KB/sec)\n")
        for i, regex in enumerate(
            tqdm(
                regex_file,
                total=num_regexes,
                miniters=1,
                mininterval=0,
            ),
            start=1,
        ):
            regex = regex[:-1]  # strip newline
            total_throughput = 0
            can_write = True
            try:
                automaton = timed_automaton_construction(
                    regex, args.expansion_type
                )
            except NotImplementedError as e:
                print(e)
                continue
            except re.PatternError as e:
                print(e)
                continue
            except ValueError as e:
                print(e)
                continue
            except TimeoutError as e:
                print(e)
                continue
            for j in range(1, args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{args.random_string_dir}/{i}-{j}.txt",
                        "r",
                        encoding=args.input_encoding,
                    ) as random_str_file:
                        random_str = random_str_file.read()

                        num_bytes = len(random_str.encode("utf-8"))
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                time_matching,
                                sc_class,
                                automaton,
                                random_str,
                                args.cache_type,
                            )
                            try:
                                secs_per_kb = 1 / THROUGHPUT_THRES
                                secs_per_b = secs_per_kb / 1000
                                matching_timeout = secs_per_b * num_bytes + 1
                                duration = future.result(
                                    timeout=matching_timeout
                                )
                                assert duration < matching_timeout
                            except TimeoutError as e:
                                print(e)
                                timing_log_file.write(
                                    f"{i}\t{THROUGHPUT_THRES/1e6}\n"
                                )
                                can_write = False
                                break

                        total_throughput += (
                            num_bytes / 1000 / duration
                        )  # KB/sec
                except FileNotFoundError as e:
                    print(e)
                    can_write = False
                    break
            if can_write:
                timing_log_file.write(
                    f"{i}\t{total_throughput / args.num_strings_per_regex}\n"
                )
        timing_log_file.close()


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
        default="sparse_counter_config",
    )
    parser.add_argument("--random-string-dir", required=True, type=str)
    parser.add_argument("--regex-file", required=True, type=str)
    parser.add_argument("--timing-log-file", required=True, type=str)
    parser.add_argument("--num-strings-per-regex", required=True, type=int)
    parser.add_argument(
        "--expansion-type",
        required=True,
        type=str,
        choices=["inner", "outer", "full"],
    )
    parser.add_argument(
        "--input-encoding", required=True, choices=["utf-8", "latin1"]
    )
    parser.add_argument(
        "--cache-type", required=True, choices=["lru", "flush_on_full", "none"]
    )
    main(parser.parse_args())
