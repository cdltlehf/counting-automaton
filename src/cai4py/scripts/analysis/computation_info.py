"""Analyze the computation steps of the position counter automaton."""

import argparse
from collections import defaultdict as dd
import io
import json
import logging
import sys
from typing import Any, Callable, Iterable, Optional, Type

import timeout_decorator  # type: ignore

from cai4py.counting_automaton.logging import ComputationStep
from cai4py.counting_automaton.logging import ComputationStepMark
from cai4py.counting_automaton.logging import VERBOSE
import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc
from cai4py.utils import read_test_cases

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
        logger.info("Collecting computation info from execution with '%s'.", w)
        for i, super_config in enumerate(get_computation(automaton, w)):
            logger.info("Super config %d: %s", i, super_config)
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
                            raise ValueError(f"Duplicate end mark: {computation_step}")
                        mark_flags[computation_step] = False
                else:
                    print(list(ComputationStepMark.__members__))
                    raise ValueError(f"Unknown computation step: {computation_step}")
            stream.seek(0)
            stream.truncate(pos)
            last_super_config = super_config
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
        computation_info, is_final = timeout_decorator.timeout(timeout)(
            collect_computation_info
        )(automaton, w, get_computation)
        return {"computation_info": computation_info, "is_final": is_final}
    except timeout_decorator.TimeoutError:
        logger.warning("Computation timeout when processing text %s", w[:100])
        return None


def run_and_log_trace(
    sc_class: Type[sc.SuperConfigBase],
    test_cases: Iterable[tuple[str, list[str]]],
):
    """
    Create and run a `PositionCountingAutomaton` for each test case, and log a trace of its execution (with log level INFO).


    Args:
        sc_class (sc.SuperConfigBase):
            The class that defines the type of super configuration to use for the automaton.
        test_cases (Iterable[tuple[str, list[str]]]):
            An iterable of test cases, where each test case is a tuple containing a pattern (str)
            and a list of texts (list[str]) to process. Call `cai4py.utils.read_test_cases` to create the `Iterable` `test_cases`.
    Logs:
        - Logs the pattern being processed.
        - Logs warnings if the automaton construction times out.
        - Logs errors if any exceptions occur during processing.
    Raises:
        timeout_decorator.TimeoutError: If the automaton construction exceeds the specified timeout.
        Exception: For any other errors encountered during processing.
    Outputs:
        Logs the results of the computations in JSON format, including the pattern and the results
        for each text in the test case.
    Notes:
        - The timeout for automaton creation is set to 0 in debug mode and 60 seconds otherwise.
        - Uses a timeout decorator to enforce the timeout for automaton creation.
    """
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if __debug__:
        timeout = 0
    else:
        timeout = 60

    create_position_automaton_with_timeout = timeout_decorator.timeout(timeout)(
        pca.PositionCountingAutomaton.create
    )
    for pattern, texts in test_cases:
        logger.info("Pattern: %s", pattern)
        try:
            results: Optional[list[dict[str, Any]]] = None
            automaton = create_position_automaton_with_timeout(pattern)
            results = []
            for text in texts:
                result = collect_optional_computation_info(
                    automaton, text, sc_class.get_computation, timeout
                )
                results.append({"text": text, "result": result})
        except timeout_decorator.TimeoutError:
            logger.warning("Construction timeout in pattern %s", pattern)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error in pattern %s: %s", pattern, e)
        finally:
            output_dict = {"pattern": pattern, "results": results}
            print(json.dumps(output_dict, indent=2))


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
    run_and_log_trace(sc_class, test_cases=read_test_cases(sys.stdin))


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
    main(parser.parse_args())
