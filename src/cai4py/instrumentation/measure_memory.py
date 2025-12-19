"""Time matching with the position counting automaton and random strings"""

import argparse
import logging
import re
import sys

from memory_profiler import memory_usage
from tqdm import tqdm

from cai4py.counting_automaton.computation_logging import VERBOSE
from cai4py.counting_automaton.fullmatch import fullmatch
from cai4py.counting_automaton.position_counting_automaton import (
    PositionCountingAutomaton,
)
from cai4py.counting_automaton.super_config.counter_config import (
    SparseCounterConfig,
)
from cai4py.counting_automaton.super_config.super_config import SuperConfig
from cai4py.custom_counters.counter_type import CounterType
from cai4py.instrumentation.utils import (
    run_with_timeout,
    add_common_arguments,
    add_super_config_argument,
    add_counter_type_argument,
    add_cache_type_argument,
    add_random_string_arguments,
    add_sample_interval_argument,
)

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def main(args: argparse.Namespace) -> None:
    sc_class = {
        "SparseCounterConfig": SparseCounterConfig,
        "SuperConfig": SuperConfig,
    }[args.super_config_class]
    counter_type = {
        "bitvector": CounterType.BIT_VECTOR,
        "counting-set": CounterType.SPARSE_COUNTING_SET,
        "none": CounterType.NONE,
    }[args.counter_type]
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        num_regexes = len(regex_file.readlines())
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        mem_usage_log_file = open(args.log_file, "w", encoding="utf-8")
        mem_usage_log_file.write(
            "Regex ID\tString ID\tPeak Memory Usage (MiB)\n"
        )
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
            print(f"Processing regex {i}: {regex}", file=sys.stderr)
            try:
                from cai4py.instrumentation.constants import (
                    AUTOMATON_CREATION_TIMEOUT,
                )

                automaton = run_with_timeout(
                    func=PositionCountingAutomaton.create,
                    args=(regex, args.expansion_type),
                    timeout=AUTOMATON_CREATION_TIMEOUT,
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
                            # peak_mem_usage = run_with_timeout(
                            #     func=lambda func, args: memory_usage(
                            #         (func, args),  # type: ignore
                            #         max_usage=True,
                            #         retval=False,
                            #     ),
                            #     args=(
                            #         fullmatch,
                            #         (
                            #             sc_class,
                            #             automaton,
                            #             random_str,
                            #             counter_type,
                            #             args.cache_type,
                            #         ),
                            #     ),
                            #     timeout=1 / THROUGHPUT_THRES * num_bytes + 3,
                            # )
                            peak_mem_usage = memory_usage(
                                (
                                    fullmatch,  # type: ignore
                                    (
                                        sc_class,
                                        automaton,
                                        random_str,
                                        counter_type,
                                        args.cache_type,
                                        args.sample_interval,
                                    ),
                                ),
                                max_usage=True,
                                retval=False,
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
                            mem_usage_log_file.write(
                                f"{i}\t{j}\t{peak_mem_usage}\n"
                            )
                        except TimeoutError as e:
                            print(e, file=sys.stderr)
                            break

                except FileNotFoundError:
                    break
        mem_usage_log_file.close()


if __name__ == "__main__":
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    add_super_config_argument(parser)
    add_random_string_arguments(parser)
    add_common_arguments(parser)
    parser.add_argument(
        "--log-file",
        required=True,
        type=str,
        help="Output file for memory usage results",
    )
    add_cache_type_argument(parser, required=False)
    add_counter_type_argument(parser)
    add_sample_interval_argument(parser)
    main(parser.parse_args())
