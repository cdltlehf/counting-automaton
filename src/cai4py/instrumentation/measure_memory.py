"""Time matching with the position counting automaton and random strings"""

import argparse
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Type

import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from cai4py.counting_automaton.logging import VERBOSE
from tqdm import tqdm
from memory_profiler import memory_usage

logger = logging.getLogger(__name__)

import signal


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


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


@timeout(seconds=10)
def timed_automaton_construction(regex, expansion_type):
    return pca.PositionCountingAutomaton.create(regex, expansion_type)


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
            print(regex)
            if not Path(
                f'{args.random_string_dir.replace("random", "attack")}/{i}.txt'
            ):
                continue
            can_write = True
            try:
                automaton = timed_automaton_construction(
                    regex, args.expansion_type
                )
            except TimeoutError:
                print("TimeoutError")
                continue
            except NotImplementedError:
                continue
            except re.PatternError:
                continue
            except ValueError:
                continue
            mean_mem_usage = 0
            for j in range(1, args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{args.random_string_dir}/{i}-{j}.txt",
                        "r",
                        encoding="utf-8",
                    ) as random_str_file:
                        random_str = random_str_file.read()

                        def measure_memory_used_during_matching(
                            automaton, random_str
                        ):
                            # Step through matching
                            def match():
                                matcher = sc.SuperConfig(automaton)
                                matcher.match(random_str)

                            max_mem_usage = memory_usage(
                                (match,), max_usage=True
                            )
                            return (
                                max_mem_usage * 1e6
                                - len(random_str.encode("utf-8"))
                            ) / 1e6

                        num_bytes = len(random_str.encode("utf-8"))
                        if num_bytes == 0:
                            continue
                        THROUGHPUT_THRES = 0.5 * 1e6  # .5 KB / s
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                measure_memory_used_during_matching,
                                automaton,
                                random_str,
                            )
                            try:
                                timeout = num_bytes / THROUGHPUT_THRES + 1
                                peak_mem_usage = future.result(timeout=timeout)
                                mean_mem_usage += (
                                    peak_mem_usage / args.num_strings_per_regex
                                )
                            except TimeoutError:
                                print("TIMEOUT")
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
    main(parser.parse_args())
