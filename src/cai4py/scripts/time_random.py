"""Time matching with the position counting automaton and random strings"""

import argparse
import logging
from typing import Type
from cai4py.counting_automaton.logging import VERBOSE
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
import time
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def main(parsed_args: argparse.Namespace) -> None:
    method: str = parsed_args.method
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
    with open(parsed_args.regex_file, "r", encoding="utf-8") as regex_file:
        num_regexes = len(regex_file.readlines())
    with open(parsed_args.regex_file, "r", encoding="utf-8") as regex_file:
        i = 1
        timing_log_file = open(
            parsed_args.timing_log_file, "w", encoding="utf-8"
        )
        timing_log_file.write("Regex ID\tMean normalised matching duration\n")
        for regex in tqdm(regex_file, total=num_regexes):
            sum = 0
            input_found = True
            for j in range(1, parsed_args.num_strings_per_regex + 1):
                try:
                    with open(
                        f"{parsed_args.random_string_dir}/{i}-{j}.txt",
                        "r",
                        encoding="latin1",
                    ) as random_str_file:
                        random_str = random_str_file.read()
                        try:
                            automaton = pca.PositionCountingAutomaton.create(
                                regex
                            )
                        except NotImplementedError:
                            continue
                        except re.PatternError:
                            continue
                        except ValueError:
                            continue
                        t0 = time.perf_counter()
                        # Step through matching
                        for computation in sc_class.get_computation(
                            automaton, random_str
                        ):
                            pass  # do nothing
                        t1 = time.perf_counter()
                        duration = t1 - t0
                        num_bytes = len(random_str.encode("latin1"))
                        if num_bytes == 0:
                            continue
                        sum += duration * 1000 / num_bytes
                except FileNotFoundError:
                    input_found = False
                    break
            if input_found:
                timing_log_file.write(
                    f"{i}\t{sum / parsed_args.num_strings_per_regex}\n"
                )
            i += 1
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
        required=True,
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
    parser.add_argument("--random_string_dir", required=True, type=str)
    parser.add_argument("--regex_file", required=True, type=str)
    parser.add_argument("--timing_log_file", required=True, type=str)
    parser.add_argument("--num_strings_per_regex", required=True, type=int)
    args = parser.parse_args()
    main(args)
