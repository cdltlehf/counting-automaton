"""Time matching with the position counting automaton and random strings"""

import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import sys

from memory_profiler import memory_usage
from tqdm import tqdm

from cai4py.counting_automaton._logging import VERBOSE
import cai4py.counting_automaton.super_config as sc
from cai4py.instrumentation.constants import THROUGHPUT_THRES
from cai4py.instrumentation.utils import run_with_timeout
from cai4py.counting_automaton.fullmatch import fullmatch
import cai4py.counting_automaton.position_counting_automaton as pca

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def main(args: argparse.Namespace) -> None:
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        num_regexes = len(regex_file.readlines())
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        mem_usage_log_file = open(args.log_file, "w", encoding="utf-8")
        mem_usage_log_file.write("Regex ID\tPeak memory usage (MiB)\n")
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
            print(regex, file=sys.stderr)
            can_write = True
            try:
                automaton = run_with_timeout(
                    func=pca.PositionCountingAutomaton.create,
                    args=(regex, args.expansion_type),
                    timeout=10,
                )
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
            mem_usage = []
            for j in range(1, args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{args.random_string_dir}/{i}/{j}.txt",
                        "r",
                        encoding=args.input_encoding,
                    ) as random_str_file:
                        random_str = random_str_file.read()
                        num_bytes = len(random_str.encode("utf-8"))

                        def measure_memory_used_during_matching(
                            automaton, random_str, args
                        ):
                            peak_mem_usage = run_with_timeout(
                                func=lambda func, args: memory_usage(
                                    (func, args),  # type: ignore
                                    max_usage=True,
                                    retval=False,
                                ),
                                args=(
                                    fullmatch,
                                    (
                                        sc.SparseCounterConfig,
                                        automaton,
                                        random_str,
                                        args.cache_type,
                                    ),
                                ),
                                timeout=1 / THROUGHPUT_THRES * num_bytes + 3,
                            )
                            assert isinstance(peak_mem_usage, float)
                            return (
                                peak_mem_usage * 1e6
                                - len(random_str.encode("utf-8"))
                            ) / 1e6

                        try:
                            peak_mem_usage = (
                                measure_memory_used_during_matching(
                                    automaton, random_str, args
                                )
                            )
                            mem_usage.append(peak_mem_usage)
                        except TimeoutError as e:
                            print(e, file=sys.stderr)
                            break

                except FileNotFoundError:
                    break
            if len(mem_usage) > 0:
                mem_usage_log_file.write(f"{i}\t{np.mean(mem_usage)}\n")

        mem_usage_log_file.close()


if __name__ == "__main__":
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-string-dir", required=True, type=str)
    parser.add_argument("--regex-file", required=True, type=str)
    parser.add_argument("--log-file", required=True, type=str)
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
        "--cache-type",
        required=False,
        type=str,
        choices=["none", "lru", "flush_on_full"],
        default="none",
    )
    main(parser.parse_args())
