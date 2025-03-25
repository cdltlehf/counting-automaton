"""Analyze the computation steps of the position counter automaton."""

import argparse
import io
import json
import logging
import sys
from typing import Callable, Iterable, Optional

import timeout_decorator  # type: ignore

from counting_automaton.logging import VERBOSE
import counting_automaton.position_counting_automaton as pca
import counting_automaton.super_config as sc
from utils import read_test_cases
from utils.analysis import ComputationInfo
from utils.analysis import OutputDict
from utils.analysis import TestCaseDict
from utils.analysis import TestCaseResult

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
) -> tuple[ComputationInfo, bool]:
    logger_dict = logging.Logger.manager.loggerDict
    counting_automaton_loggers = {
        name: counting_automaton_logger
        for name, counting_automaton_logger in logger_dict.items()
        if name.startswith("counting_automaton")
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

    computation_info = ComputationInfo(
        EVAL_SYMBOL=0,
        EVAL_PREDICATE=0,
        APPLY_OPERATION=0,
        ACCESS_NODE_MERGE=0,
        ACCESS_NODE_CLONE=0,
    )
    last_super_config: Optional[sc.SuperConfigBase] = None
    try:
        for i, super_config in enumerate(get_computation(automaton, w)):
            logger.debug("Super config %d: %s", i, super_config)
            pos = stream.tell()
            value = stream.getvalue()[:pos]
            for computation_step in value.splitlines():
                if computation_step in computation_info:
                    computation_info[computation_step] += 1  # type: ignore
                else:
                    raise ValueError(
                        f"Unknown computation step: {computation_step}"
                    )
            stream.seek(0)
            stream.truncate(pos)
            last_super_config = super_config
            logger.debug("Computation info %d: %s", i, computation_info)
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
    return computation_info, last_super_config.is_final()


def collect_optional_computation_info(
    automaton: pca.PositionCountingAutomaton,
    w: str,
    get_computation: Callable[
        [pca.PositionCountingAutomaton, str], Iterable[sc.SuperConfigBase]
    ],
    timeout: int,
) -> Optional[TestCaseResult]:
    try:
        computation_info, is_final = timeout_decorator.timeout(timeout)(
            collect_computation_info
        )(automaton, w, get_computation)
        return TestCaseResult(
            computation_info=computation_info, is_final=is_final
        )
    except timeout_decorator.TimeoutError:
        logger.warning("Computation timeout when processing text %s", w)
        return None


def main(method: str) -> None:
    get_computation = {
        "super_config": sc.SuperConfig.get_computation,
        "bounded_super_config": sc.BoundedSuperConfig.get_computation,
        "counter_config": sc.CounterConfig.get_computation,
        "bounded_counter_config": sc.BoundedCounterConfig.get_computation,
        "sparse_counter_config": sc.SparseCounterConfig.get_computation,
    }[method]
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if __debug__:
        timeout = 0
    else:
        timeout = 60

    create_position_automaton_with_timeout = timeout_decorator.timeout(timeout)(
        pca.PositionCountingAutomaton.create
    )
    for pattern, texts in read_test_cases(sys.stdin):
        logger.info("Pattern: %s", pattern)
        try:
            results: Optional[list[TestCaseDict]] = None
            automaton = create_position_automaton_with_timeout(pattern)
            results = []
            for text in texts:
                result = collect_optional_computation_info(
                    automaton, text, get_computation, timeout
                )
                results.append(TestCaseDict(text=text, result=result))
        except timeout_decorator.TimeoutError:
            logger.warning("Construction timeout in pattern %s", pattern)
            pass

        finally:
            output_dict = OutputDict(pattern=pattern, results=results)
            print(json.dumps(output_dict))


if __name__ == "__main__":
    if __debug__:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()
    main(args.method)
