"""Time matching with the position counting automaton and random strings"""

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
from cai4py.counting_automaton.logging import VERBOSE
from tqdm import tqdm


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
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        num_regexes = len(regex_file.readlines())
    with open(args.regex_file, "r", encoding="utf-8") as regex_file:
        timing_log_file = open(args.timing_log_file, "w", encoding="utf-8")
        timing_log_file.write("Regex ID\tMean throughput (KB/sec)\n")
        for i, regex in enumerate(
            tqdm(
                regex_file.readlines(),
                total=num_regexes,
                miniters=1,
                mininterval=0,
            ),
            start=1,
        ):
            if not Path(
                f'{args.random_string_dir.replace("random", "attack")}/{i}.txt'
            ):
                continue
            total_throughput = 0
            can_write = True
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
                        random_str = random_str_file.read()

                        def time_matching(automaton, random_str):
                            t0 = time.perf_counter()
                            # Step through matching
                            for computation in sc_class.get_computation(
                                automaton, random_str
                            ):
                                pass  # do nothing
                            assert computation is not None
                            assert computation.is_final()
                            t1 = time.perf_counter()
                            duration = t1 - t0
                            return duration

                        num_bytes = len(random_str.encode("utf-8"))
                        if num_bytes == 0:
                            continue
                        THROUGHPUT_THRES = 0.5 * 1e6  # .5 KB / s
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                time_matching, automaton, random_str
                            )
                            try:
                                matching_timeout = (
                                    num_bytes / THROUGHPUT_THRES + 1
                                )
                                duration = future.result(
                                    timeout=matching_timeout
                                )
                                assert duration < matching_timeout
                            except TimeoutError:
                                print("TIMEOUT")
                                timing_log_file.write(
                                    f"{i}\t{THROUGHPUT_THRES/1e6}\n"
                                )
                                can_write = False
                                break

                        total_throughput += num_bytes / 1000 / duration
                except FileNotFoundError:
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
        choices=["inner", "outer", "all"],
    )
    main(parser.parse_args())
