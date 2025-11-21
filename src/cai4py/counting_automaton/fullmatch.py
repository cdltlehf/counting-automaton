"""Time matching with the position counting automaton and attack strings"""

import time
import argparse
import logging
import pickle
import sys
import time
from typing import Literal, Type

from cai4py.cache_utils import make_cached_versions
from cai4py.counting_automaton._logging import VERBOSE
import cai4py.counting_automaton.position_counting_automaton as pca
from cai4py.counting_automaton.super_config.super_config import SuperConfig
from cai4py.custom_counters.counter_type import CounterType
from cai4py.more_collections import OrderedSet
from cai4py.counting_automaton.super_config import SuperConfigBase
import cai4py.counting_automaton.super_config as sc

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def fullmatch(
    sc_class: Type[SuperConfigBase],  # Adjusted to accept a class type
    automaton: pca.PositionCountingAutomaton,
    w: str,
    counter_type: CounterType,
    cache_type: Literal["lru", "flush_on_full", "none"] = "lru",
    sample_interval: int = 0,  # if > 0, sample cache stats every N characters
) -> tuple[bool, list]:
    super_config = sc_class.get_initial(
        automaton, counter_type
    )  # Use class method directly

    # Fast path: when caching is disabled, avoid pickling/unpickling entirely.
    # This removes large per-character overhead and mirrors the BVA matcher flow.
    if cache_type == "none":
        cache_history: list = []
        for _symbol in w:
            super_config = super_config.update(_symbol)
        return super_config.is_final(), cache_history

    # Cached path: use lightweight wrappers with explicit cache policies.
    def get_next_super_config(
        pickled_super_config: bytes,
        symbol: str,
    ) -> bytes:
        # Retrieve the super_config from cache
        super_config_local = pickle.loads(pickled_super_config)
        logger.debug("Current super_config: %s", str(super_config_local))
        return pickle.dumps(super_config_local.update(symbol))

    cached_get_next_super_config = make_cached_versions(
        get_next_super_config, maxsize=1024
    )
    pickled_super_config = pickle.dumps(super_config)
    cache_history: list = []
    for i, symbol in enumerate(w):
        # Load current super_config before processing symbol for accurate logging/state.
        current_super_config = pickle.loads(pickled_super_config)
        logger.debug("Processing symbol: %s", symbol)
        logger.debug("Current configs: %s", current_super_config)
        pickled_super_config = cached_get_next_super_config[cache_type](
            pickled_super_config, symbol
        )
        # Optionally could log next state; omitted to reduce overhead.
        # Sample cache stats at specified intervals
        if sample_interval > 0 and (i + 1) % sample_interval == 0:
            try:
                cache_info = cached_get_next_super_config[
                    cache_type
                ].cache_info()
                cache_history.append((i + 1, cache_info))
            except AttributeError as e:
                print(e, file=sys.stderr)
    # Load final super_config after processing all symbols
    super_config = pickle.loads(pickled_super_config)
    # Capture final cache stats once matching is complete.
    try:
        final_cache_info = cached_get_next_super_config[cache_type].cache_info()
    except AttributeError:
        final_cache_info = None
    if final_cache_info is not None:
        print(f"final_cache_info: {final_cache_info}")
    return super_config.is_final(), cache_history


def main(args: argparse.Namespace) -> None:
    sc_class: Type[sc.SuperConfigBase] = {
        "SuperConfig": sc.SuperConfig,
        "SparseCounterConfig": sc.SparseCounterConfig,
    }[args.super_config_class]
    counter_type = {
        "bit-vector": CounterType.BIT_VECTOR,
        "sparse-counting-set": CounterType.SPARSE_COUNTING_SET,
    }[args.counter_type]
    automaton = pca.PositionCountingAutomaton.create(
        args.regex, expansion_type=args.expansion_type
    )
    print(automaton)
    t0 = time.perf_counter()

    is_match, cache_history = fullmatch(
        sc_class, automaton, args.input_string, counter_type, args.cache_type
    )
    t1 = time.perf_counter()
    duration = t1 - t0
    if cache_history:
        print("cache_history samples:", len(cache_history))
    num_bytes = len(args.input_string.encode("latin1"))
    print(f"match: {is_match}\nthroughput: {duration * 1000 / num_bytes}\n")


if __name__ == "__main__":
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--super-config-class",
        type=str,
        required=False,
        default="SparseCounterConfig",
        choices=[
            "SuperConfig",
            "SparseCounterConfig",
        ],
    )
    parser.add_argument("--input-string", required=True, type=str)
    parser.add_argument("--regex", required=True, type=str)
    parser.add_argument(
        "--expansion-type",
        required=True,
        type=str,
        choices=["inner", "outer", "full"],
    )
    parser.add_argument(
        "--cache-type",
        type=str,
        required=True,
        choices=["lru", "flush_on_full", "none"],
    )
    parser.add_argument(
        "--counter-type",
        type=str,
        required=True,
        default="sparse-counting-set",
        choices=["bit-vector", "sparse-counting-set"],
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=0,
        help="Sample cache stats every N characters if > 0",
    )
    main(parser.parse_args())
