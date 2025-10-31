"""Time matching with the position counting automaton and attack strings"""

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
            try:
                automaton = pca.PositionCountingAutomaton.create(
                    regex, expansion_type=args.expansion_type
                )
                with open(
                    f"{args.attack_string_dir}/{i}.txt",
                    "r",
                    encoding=args.input_encoding,
                ) as attack_str_file:
                    try:
                        attack_str = attack_str_file.read()
                    except UnicodeDecodeError as e:
                        print(e)
                        continue
                    t0 = time.perf_counter()
                    # Step through matching
                    for _ in sc_class.get_computation(automaton, attack_str):
                        pass  # Do nothing
                    t1 = time.perf_counter()
                    duration = t1 - t0
                    num_bytes = len(attack_str.encode(args.input_encoding))
                    timing_log_file.write(
                        f"{i}\t{num_bytes / 1000 / duration}\n"
                    )
            except FileNotFoundError as e:
                print(e)
                continue
            except NotImplementedError as e:
                print(e)
                continue
            except re.PatternError as e:
                print(e)
                continue
            except ValueError as e:
                print(e)
                continue
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
    parser.add_argument("--attack-string-dir", required=True, type=str)
    parser.add_argument("--regex-file", required=True, type=str)
    parser.add_argument("--timing-log-file", required=True, type=str)
    parser.add_argument(
        "--expansion-type", required=True, type=str, choices=["inner", "outer"]
    )
    parser.add_argument(
        "--input-encoding", required=True, choices=["utf-8", "latin1"]
    )
    main(parser.parse_args())
