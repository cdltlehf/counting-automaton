"""Time matching with the position counting automaton and random strings"""

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
from cai4py.instrumentation.utils import timed_automaton_construction
from cai4py.counting_automaton.fullmatch import fullmatch

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
                regex_file.readlines(),
                total=num_regexes,
                miniters=1,
                mininterval=0,
            ),
            start=1,
        ):
            can_write = True
            try:
                automaton = timed_automaton_construction(
                    regex, args.expansion_type
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
            mean_mem_usage = 0
            for j in range(1, args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{args.random_string_dir}/{i}-{j}.txt",
                        "r",
                        encoding=args.input_encoding,
                    ) as random_str_file:
                        random_str = random_str_file.read()

                        def measure_memory_used_during_matching(
                            automaton, random_str, args
                        ):

                            max_mem_usage = memory_usage(
                                (
                                    fullmatch,
                                    (
                                        sc.SparseCounterConfig,
                                        automaton,
                                        random_str,
                                        args.cache_type,
                                    ),
                                ),  # type: ignore
                                max_usage=True,
                            )
                            return (
                                max_mem_usage * 1e6
                                - len(random_str.encode("utf-8"))
                            ) / 1e6

                        num_bytes = len(random_str.encode("utf-8"))
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                measure_memory_used_during_matching,
                                (automaton, random_str, args),  # type: ignore
                            )
                            try:
                                matching_timeout = (
                                    num_bytes / THROUGHPUT_THRES + 1
                                )
                                peak_mem_usage = future.result(
                                    timeout=matching_timeout
                                )
                                mean_mem_usage += (
                                    peak_mem_usage / args.num_strings_per_regex
                                )
                            except TimeoutError as e:
                                print(e, file=sys.stderr)
                                can_write = False
                                executor.shutdown(wait=False)
                                break

                except FileNotFoundError:
                    can_write = False
                    break
            if can_write:
                mem_usage_log_file.write(f"{i}\t{mean_mem_usage}\n")

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
