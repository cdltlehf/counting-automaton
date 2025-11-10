"""Time matching with the position counting automaton and attack strings"""

import argparse
import logging
import time
from typing import Literal, Type

from cai4py.cache_utils import make_cached_versions
from cai4py.collections import OrderedSet
from cai4py.counting_automaton._logging import VERBOSE
from cai4py.counting_automaton.position_counting_automaton import Config
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from cai4py.counting_automaton.super_config import SuperConfigBase

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def fullmatch(
    sc_class: SuperConfigBase,
    automaton: pca.PositionCountingAutomaton,
    w: str,
    cache_type: Literal["lru", "flush_on_full", "none"] = "lru",
) -> tuple[bool, object]:

    super_config = sc_class.get_initial(automaton)

    def get_next_super_config(
        super_config: SuperConfigBase,
        symbol: str,
    ) -> SuperConfigBase:
        return super_config.update(symbol)

    cached_get_next_super_config = make_cached_versions(
        get_next_super_config, maxsize=1024
    )
    for symbol in w:
        logger.debug(f"Processing symbol: {symbol}")
        logger.debug(f"Current configs: {super_config}")
        super_config = cached_get_next_super_config[cache_type](
            super_config, symbol
        )
        logger.debug(f"Next configs: {super_config}")
    try:
        cache_stats = cached_get_next_super_config[cache_type].cache_info()
    except AttributeError:
        cache_stats = None
    if super_config.is_final():
        return True, cache_stats
    return False, cache_stats


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
    automaton = pca.PositionCountingAutomaton.create(
        args.regex, expansion_type=args.expansion_type
    )
    print(automaton)
    t0 = time.perf_counter()

    is_match, cache_stats = fullmatch(
        sc_class.get_computation, automaton, args.input_string, args.cache_type
    )
    t1 = time.perf_counter()
    duration = t1 - t0
    print("cache_stats:", cache_stats)
    num_bytes = len(args.input_string.encode("latin1"))
    print(f"match: {is_match}\nthroughput: {duration * 1000 / num_bytes}\n")


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
        default="sparse_counter_config",
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
    parser.add_argument("--input-string", required=True, type=str)
    parser.add_argument("--regex", required=True, type=str)
    parser.add_argument(
        "--expansion-type", required=True, type=str, choices=["inner", "outer"]
    )
    parser.add_argument(
        "--cache-type",
        type=str,
        required=True,
        choices=["lru", "flush_on_full", "none"],
    )
    main(parser.parse_args())
