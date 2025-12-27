"""Time matching with the position counting automaton and attack strings"""

import argparse
import logging
import os
import re
from typing import Type
import sys

from cai4py.counting_automaton.computation_logging import VERBOSE
from cai4py.custom_counters.counter_type import CounterType
from cai4py.instrumentation.constants import THROUGHPUT_THRES
from cai4py.instrumentation.utils import (
    NoMatchError,
    add_counter_type_argument,
    get_matching_timeout,
    run_with_timeout,
)
from tqdm import tqdm

from cai4py.counting_automaton._logging import VERBOSE
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc

from cai4py.instrumentation.utils import (
    run_with_timeout,
    time_matching,
    add_common_arguments,
    add_cache_type_argument,
    add_super_config_argument,
    add_sample_interval_argument,
)

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def main(args: argparse.Namespace) -> None:
    sc_class: Type[sc.SuperConfigBase] = {
        "SuperConfig": sc.SuperConfig,
        "SparseCounterConfig": sc.SparseCounterConfig,
    }[args.super_config_class]
    counter_type = {
        "counting-set": CounterType.COUNTING_SET,
        "bitvector": CounterType.BIT_VECTOR,
        "none": CounterType.NONE,
    }[args.counter_type]

    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        num_regexes = len(regex_file.readlines())

    # Open cache history file if sampling is enabled
    cache_history_file = None
    if args.sample_interval > 0 and args.cache_history_log_file:
        cache_history_file = open(
            args.cache_history_log_file, "w", encoding="utf-8"
        )
        cache_history_file.write(
            "Regex ID\tPosition\tHits\tMisses\tMaxsize\tCurrsize\n"
        )

    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        directory = args.timing_log_file.rsplit("/", 1)[0]
        os.makedirs(directory, exist_ok=True)
        timing_log_file = open(args.timing_log_file, "w", encoding="utf-8")
        timing_log_file.write("Regex ID\tThroughput (KB/sec)\n")
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
            print(f"Processing regex {i}: {regex}")
            with open("regex.txt", "w", encoding="utf-8") as debug_regex_file:
                debug_regex_file.write(regex)
            try:
                from cai4py.instrumentation.constants import (
                    AUTOMATON_CREATION_TIMEOUT,
                )

                automaton = run_with_timeout(
                    func=pca.PositionCountingAutomaton.create,
                    args=(regex, args.expansion_type),
                    timeout=AUTOMATON_CREATION_TIMEOUT,
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
            try:
                with open(
                    f"{args.attack_string_dir}/{i}.txt",
                    "r",
                    encoding=args.input_encoding,
                ) as attack_str_file:
                    try:
                        attack_str = attack_str_file.read()
                    except UnicodeDecodeError as e:
                        print(e, file=sys.stderr)
                        continue
                    try:
                        with open(
                            "string.txt", "w", encoding="utf-8"
                        ) as debug_str_file:
                            debug_str_file.write(attack_str)
                        num_bytes = len(attack_str.encode(args.input_encoding))
                        matching_timeout = get_matching_timeout(num_bytes)
                        sample_interval = getattr(args, "sample_interval", 0)
                        result = run_with_timeout(
                            func=time_matching,
                            args=(
                                sc_class,
                                automaton,
                                attack_str,
                                args.cache_type,
                                counter_type,
                                sample_interval,
                                False,  # raise_error_if_no_match
                            ),
                            timeout=matching_timeout,
                        )
                        if result is None:
                            continue
                        duration, cache_history = result
                    except TimeoutError as e:
                        print(e, file=sys.stderr)
                        timing_log_file.write(f"{i}\t{THROUGHPUT_THRES/1e6}\n")
                        continue
                    except RuntimeError as e:
                        print(e, file=sys.stderr)
                        timing_log_file.write(f"{i}\t{THROUGHPUT_THRES/1e6}\n")
                        continue
                    except NoMatchError as e:
                        print(e, file=sys.stderr)
                        continue
                    if result is None:
                        continue
                    assert isinstance(result, tuple) and len(result) == 2
                    duration, cache_history = result
                    assert isinstance(duration, (int, float))
                    duration = float(duration)
                    if duration <= 0:
                        throughput = THROUGHPUT_THRES / 1e6
                    else:
                        throughput = num_bytes / 1000 / duration

                    timing_log_file.write(f"{i}\t{throughput}\n")

                    # Write cache history if enabled
                    if cache_history_file and cache_history:
                        for position, stats in cache_history:
                            try:
                                cache_history_file.write(
                                    f"{i}\t{position}\t{stats.hits}\t{stats.misses}\t{stats.maxsize}\t{stats.currsize}\n"
                                )
                            except AttributeError:
                                # Handle cases where stats might not have expected attributes
                                pass
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                continue
        timing_log_file.close()
        if cache_history_file:
            cache_history_file.close()


if __name__ == "__main__":
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    add_counter_type_argument(parser)
    add_super_config_argument(parser)
    parser.add_argument(
        "--attack-string-dir",
        required=True,
        type=str,
        help="Directory containing attack strings",
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--timing-log-file",
        required=True,
        type=str,
        help="Output file for timing results",
    )
    add_cache_type_argument(parser, required=True)
    add_sample_interval_argument(parser)
    parser.add_argument(
        "--cache-history-log-file",
        type=str,
        default=None,
        help="Output file for cache history timeline (requires --sample-interval > 0)",
    )
    args = parser.parse_args()
    if args.sample_interval > 0 and not args.cache_history_log_file:
        parser.error(
            "--cache-history-log-file must be specified if --sample-interval > 0"
        )
    main(args)
