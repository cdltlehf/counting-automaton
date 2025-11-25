"""Time matching with the position counting automaton and attack strings"""

import argparse
import logging
import re
from typing import Type
import sys

from cai4py.counting_automaton.computation_logging import VERBOSE
from cai4py.custom_counters.counter_type import CounterType
from cai4py.instrumentation.constants import THROUGHPUT_THRES
from cai4py.instrumentation.utils import (
    NoMatchError,
    add_counter_type_argument,
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
    }[args.counter_type]
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
            regex = regex[:-1]  # strip newline
            print(f"Processing regex {i}: {regex}")
            try:
                automaton = run_with_timeout(
                    func=pca.PositionCountingAutomaton.create,
                    args=(regex, args.expansion_type),
                    timeout=10,
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
                        duration, _ = time_matching(
                            sc_class,
                            automaton,
                            attack_str,
                            args.cache_type,
                            counter_type,
                            raise_error_if_no_match=False,  # don't raise an error if there is no match
                        )
                    except TimeoutError as e:
                        print(e, file=sys.stderr)
                        timing_log_file.write(f"{i}\t{THROUGHPUT_THRES/1e6}\n")
                        continue
                    except NoMatchError as e:
                        print(e, file=sys.stderr)
                        raise Exception("This should not happen")
                    num_bytes = len(attack_str.encode(args.input_encoding))
                    timing_log_file.write(
                        f"{i}\t{num_bytes / 1000 / duration}\n"
                    )
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                continue
        timing_log_file.close()


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
    main(parser.parse_args())
