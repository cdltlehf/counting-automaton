"""Time matching with the position counting automaton and random strings"""

import argparse
import logging
import re
import sys

from tqdm import tqdm

from cai4py.counting_automaton._logging import VERBOSE
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from cai4py.instrumentation.constants import THROUGHPUT_THRES
from cai4py.instrumentation.utils import get_matching_timeout
from cai4py.instrumentation.utils import run_with_timeout
from cai4py.instrumentation.utils import time_matching

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
            print(regex)
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
            for j in range(1, args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{args.random_string_dir}/{i}/{j}.txt",
                        "r",
                        encoding=args.input_encoding,
                    ) as random_str_file:
                        random_str = random_str_file.read()
                        num_bytes = len(random_str.encode("utf-8"))
                        matching_timeout = get_matching_timeout(num_bytes)
                        sample_interval = getattr(args, "sample_interval", 0)
                        try:
                            result = run_with_timeout(
                                func=time_matching,
                                args=(
                                    sc_class,
                                    automaton,
                                    random_str,
                                    args.cache_type,
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
                        if result is None:
                            continue
                        assert isinstance(result, tuple) and len(result) == 2
                        duration, cache_history = result
                        assert isinstance(duration, (int, float))
                        duration_f = float(duration)
                        if duration_f <= 0:
                            throughput = THROUGHPUT_THRES / 1e6
                        else:
                            throughput = num_bytes / 1000 / duration_f

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
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=0,
        help="Sample cache stats every N characters (0 = no sampling)",
    )
    parser.add_argument(
        "--cache-history-log-file",
        type=str,
        default=None,
        help="Output file for cache utilization history (only used if --sample-interval > 0)",
    )
    main(parser.parse_args())
