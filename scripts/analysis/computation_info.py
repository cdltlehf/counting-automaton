"""Analyze the computation steps of the position counter automaton."""

import argparse
from collections import defaultdict as dd
# from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import sys
import time
from typing import Any, Callable, Iterable, Optional

import timeout_decorator  # type: ignore

from cai4py.counting_automaton.logging import ComputationStep
from cai4py.counting_automaton.logging import ComputationStepMark
from cai4py.counting_automaton.logging import VERBOSE
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from cai4py.counting_automaton.super_config.determinized_counter_config_base import DeterminizedCounterConfigBase
from cai4py.parser_tools import parse
from cai4py.parser_tools import to_string
from cai4py.parser_tools.utils import flatten_inner_quantifiers
from cai4py.parser_tools.utils import flatten_quantifiers

logger = logging.getLogger(__name__)


class VerboseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == VERBOSE


def collect_computation_info(
    automaton: pca.PositionCountingAutomaton,
    w: str,
    get_computation: Callable[
        [pca.PositionCountingAutomaton, str], Iterable[sc.SuperConfigBase]
    ],
) -> tuple[dict[str, int], bool]:
    logger_dict = logging.Logger.manager.loggerDict
    counting_automaton_loggers = {
        name: counting_automaton_logger
        for name, counting_automaton_logger in logger_dict.items()
        if name.startswith("cai4py")
        and isinstance(counting_automaton_logger, logging.Logger)
    }
    counting_automaton_logger_configs = {
        name: (logger.level, logger.handlers.copy())
        for name, logger in counting_automaton_loggers.items()
    }
    stream = io.StringIO()
    handler: logging.Handler = logging.StreamHandler(stream)
    logger_filter = VerboseFilter()
    handler.addFilter(logger_filter)

    for counting_automaton_logger in counting_automaton_loggers.values():
        counting_automaton_logger.setLevel(VERBOSE)
        counting_automaton_logger.handlers.clear()
        counting_automaton_logger.addHandler(handler)

    computation_info: dd[str, int] = dd(int)
    last_super_config: Optional[sc.SuperConfigBase] = None
    try:
        mark_flags: dd[str, bool] = dd(bool)
        max_num_keys = 0
        max_synchronization_degree = 0
        start_time_perf = time.perf_counter_ns()
        start_time_process = time.process_time_ns()
        for i, super_config in enumerate(get_computation(automaton, w)):
            logger.debug("Super config %d: %s", i, super_config)
            # if isinstance(super_config, DeterminizedCounterConfigBase):
            #     num_keys = super_config.get_num_keys()
            #     synchronization_degrees = (
            #         super_config.get_synchronization_degrees()
            #     )
            #     max_num_keys = max(  # pylint: disable=nested-min-max
            #         max_num_keys, max(num_keys.values(), default=0)
            #     )
            #     max_synchronization_degree = (
            #         max(  # pylint: disable=nested-min-max
            #             max_synchronization_degree,
            #             max(synchronization_degrees.values(), default=0),
            #         )
            #     )
            pos = stream.tell()
            value = stream.getvalue()[:pos]
            for computation_step in value.splitlines():
                if computation_step in ComputationStep.__members__:
                    computation_info[computation_step] += 1

                    for mark, flag in mark_flags.items():
                        if not flag:
                            continue
                        marked_computation_step = f"{mark}_{computation_step}"
                        computation_info[marked_computation_step] += 1

                elif computation_step in ComputationStepMark.__members__:
                    if computation_step.startswith("START_"):
                        computation_step = computation_step[6:]
                        if mark_flags[computation_step]:
                            raise ValueError(
                                f"Duplicate start mark: {computation_step}"
                            )
                        mark_flags[computation_step] = True
                    elif computation_step.startswith("END_"):
                        computation_step = computation_step[4:]
                        if not mark_flags[computation_step]:
                            raise ValueError(
                                f"Duplicate end mark: {computation_step}"
                            )
                        mark_flags[computation_step] = False
                else:
                    print(list(ComputationStepMark.__members__))
                    raise ValueError(
                        f"Unknown computation step: {computation_step}"
                    )
            stream.seek(0)
            stream.truncate(pos)
            last_super_config = super_config
            computation_info["max_num_keys"] = max_num_keys
            computation_info["max_synchronization_degree"] = (
                max_synchronization_degree
            )
        end_time_perf = time.perf_counter_ns()
        end_time_process = time.process_time_ns()
        computation_info["time_perf"] = end_time_perf - start_time_perf
        computation_info["time_process"] = end_time_process - start_time_process
        logger.debug("Computation info %d: %s", i, dict(computation_info))
        assert last_super_config is not None
    finally:
        handler.close()
        for counting_automaton_logger, (level, handlers) in zip(
            counting_automaton_loggers.values(),
            counting_automaton_logger_configs.values(),
        ):
            counting_automaton_logger.setLevel(level)
            counting_automaton_logger.handlers.clear()
            for handler in handlers:
                counting_automaton_logger.addHandler(handler)
    return dict(computation_info), last_super_config.is_final()


def collect_optional_computation_info(
    automaton: pca.PositionCountingAutomaton,
    w: str,
    get_computation: Callable[
        [pca.PositionCountingAutomaton, str], Iterable[sc.SuperConfigBase]
    ],
    timeout: int,
) -> Optional[dict[str, Any]]:
    try:
        timeout_collect_computation_info = timeout_decorator.timeout(timeout)(
            collect_computation_info
        )
        computation_info, is_final = timeout_collect_computation_info(
            automaton, w, get_computation
        )
        return {"computation_info": computation_info, "is_final": is_final}
    except timeout_decorator.TimeoutError:
        logger.warning("Computation timeout when processing text %s", w[:100])
        return None


def main() -> None:
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    method_to_get_computation = {
        "super_config": sc.SuperConfig.get_computation,
        "bounded_super_config": sc.BoundedSuperConfig.get_computation,
        "counter_config": sc.CounterConfig.get_computation,
        "bounded_counter_config": sc.BoundedCounterConfig.get_computation,
        "sparse_counter_config": sc.SparseCounterConfig.get_computation,
        "determinized_counter_config": (
            sc.DeterminizedCounterConfig.get_computation
        ),
        "determinized_bounded_counter_config": (
            sc.DeterminizedBoundedCounterConfig.get_computation
        ),
        "determinized_sparse_counter_config": (
            sc.DeterminizedSparseCounterConfig.get_computation
        ),
        # NOTE: Counter expansion
        "counter_expansion": sc.SuperConfig.get_computation,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(method_to_get_computation.keys()),
    )

    args = parser.parse_args()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if __debug__:
        timeout = 0
    else:
        timeout = 1

    get_computation = method_to_get_computation[args.method]
    create_position_automaton_with_timeout = timeout_decorator.timeout(timeout)(
        pca.PositionCountingAutomaton.create
    )

    def job(entry: dict[str, Any]) -> dict[str, Any]:
        pattern = entry["pattern"]
        texts = entry["texts"]
        logger.debug("Pattern: %s", pattern)
        results: Optional[list[dict[str, Any]]] = None
        results = []
        try:
            if args.method == "counter_expansion":
                normalized_pattern = to_string(
                    flatten_quantifiers(parse(pattern))
                )
            else:
                normalized_pattern = to_string(
                    flatten_inner_quantifiers(parse(pattern))
                )
            automaton = create_position_automaton_with_timeout(
                normalized_pattern
            )
            for text in texts:
                result = collect_optional_computation_info(
                    automaton, text, get_computation, timeout
                )
                results.append({"text": text, "result": result})
        except timeout_decorator.TimeoutError:
            logger.warning(
                "Construction timeout in pattern %s", normalized_pattern
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error in pattern %s: %s", pattern, e)
        finally:
            output_dict = {"pattern": pattern, "results": results}
        return output_dict

    # max_workers = 4
    entries = map(json.loads, sys.stdin)
    # with ThreadPoolExecutor(max_workers) as executor:
    for _, output_dict in enumerate(map(job, entries)):
        print(json.dumps(output_dict))


if __name__ == "__main__":
    main()
    main()
