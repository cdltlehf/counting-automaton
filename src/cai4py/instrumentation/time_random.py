"""Time matching with the position counting automaton and random strings"""

import sys
import os
from cai4py.instrumentation.constants import THROUGHPUT_THRES

import argparse
import logging
import re
import sys

from cai4py.counting_automaton.computation_logging import VERBOSE
from cai4py.custom_counters.counter_type import CounterType
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from tqdm import tqdm

from cai4py.instrumentation.utils import (
    NoMatchError,
    get_matching_timeout,
    run_with_timeout,
    time_matching,
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
    sc_class: sc.SuperConfigBase = {
        "SuperConfig": sc.SuperConfig,
        "SparseCounterConfig": sc.SparseCounterConfig,
    }[args.super_config_class]
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        num_regexes = len(regex_file.readlines())

    # Open cache history file if sampling is enabled
    cache_history_file = None
    if args.sample_interval > 0 and args.cache_history_log_file:
        cache_history_file = open(
            args.cache_history_log_file, "w", encoding="utf-8"
        )
        cache_history_file.write(
            "Regex ID\tString ID\tPosition\tHits\tMisses\tMaxsize\tCurrsize\n"
        )

    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        directory = args.timing_log_file.rsplit("/", 1)[0]
        os.makedirs(directory, exist_ok=True)
        timing_log_file = open(args.timing_log_file, "w", encoding="utf-8")
        timing_log_file.write("Regex ID\tString ID\tThroughput (KB/sec)\n")
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
                automaton = run_with_timeout(
                    func=pca.PositionCountingAutomaton.create,
                    args=(regex, args.expansion_type),
                    timeout=10,
                )
                assert isinstance(automaton, pca.PositionCountingAutomaton)
                if automaton is None:
                    raise RuntimeError("Automaton is None")
            except NotImplementedError as e:
                print(e, file=sys.stderr)
                continue
            except re.PatternError as e:
                print(e, file=sys.stderr)
                continue
            except ValueError as e:
                print(e, file=sys.stderr)
                continue
            except TimeoutError as e:
                print(e, file=sys.stderr)
                continue
            if automaton is None:
                raise RuntimeError("Automaton creation failed")
            for j in range(1, args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{args.random_string_dir}/{i}/{j}.txt",
                        "r",
                        encoding=args.input_encoding,
                    ) as random_str_file:
                        random_str = random_str_file.read()
                        if re.fullmatch(regex, random_str) is None:
                            continue  # Skip non-matching strings
                        with open(
                            "string.txt", "w", encoding="utf-8"
                        ) as debug_str_file:
                            debug_str_file.write(random_str)
                        num_bytes = len(random_str.encode("utf-8"))
                        matching_timeout = get_matching_timeout(num_bytes)
                        sample_interval = getattr(args, "sample_interval", 0)
                        counter_type = {
                            "counting-set": CounterType.COUNTING_SET,
                            "bitvector": CounterType.BIT_VECTOR,
                            "none": CounterType.NONE,
                        }[args.counter_type]
                        try:
                            result = run_with_timeout(
                                func=time_matching,
                                args=(
                                    sc_class,
                                    automaton,
                                    random_str,
                                    args.cache_type,
                                    counter_type,
                                    sample_interval,
                                ),
                                timeout=matching_timeout,
                            )
                        except TimeoutError as e:
                            print(e, file=sys.stderr)
                            timing_log_file.write(
                                f"{i}\t{j}\t{THROUGHPUT_THRES/1e6}\n"
                            )
                            break
                        except RuntimeError as e:
                            print(e, file=sys.stderr)
                            timing_log_file.write(
                                f"{i}\t{j}\t{THROUGHPUT_THRES/1e6}\n"
                            )
                            break
                        except NoMatchError as e:
                            if re.fullmatch(regex, random_str) is not None:
                                raise NoMatchError(
                                    "No match was found, but one should have been found."
                                ) from e  # Re-raise if string should match
                            assert re.fullmatch(regex, random_str) is None
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

                        timing_log_file.write(f"{i}\t{j}\t{throughput}\n")

                        # Write cache history if enabled
                        if cache_history_file and cache_history:
                            for position, stats in cache_history:
                                try:
                                    cache_history_file.write(
                                        f"{i}\t{j}\t{position}\t{stats.hits}\t{stats.misses}\t{stats.maxsize}\t{stats.currsize}\n"
                                    )
                                except AttributeError:
                                    # Handle cases where stats might not have expected attributes
                                    pass

                except FileNotFoundError as e:
                    print(e, file=sys.stderr)
                    break
        timing_log_file.close()
        if cache_history_file:
            cache_history_file.close()


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
        help="Output file for cache utilization history (only used if --sample-interval > 0)",
    )
    add_counter_type_argument(parser)
    main(parser.parse_args())
