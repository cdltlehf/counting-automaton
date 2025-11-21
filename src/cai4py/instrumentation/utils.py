import sys
import traceback
import time
import multiprocessing
from multiprocessing.queues import Empty # type: ignore
from typing import Literal  # type: ignore
from cai4py.counting_automaton.super_config.super_config_base import (
    SuperConfigBase,
)
from cai4py.instrumentation.constants import THROUGHPUT_THRES
from cai4py.counting_automaton.position_counting_automaton import (
    PositionCountingAutomaton,
)
from cai4py.counting_automaton.fullmatch import fullmatch
from cai4py.custom_counters.counter_type import CounterType


class NoMatchError(Exception):
    """Raised when no match is found during matching."""

    pass


def run_with_timeout(func, args=(), timeout=None):
    """Run a function with the given arguments in a separate process with a timeout.
    If the function does not complete within the timeout, terminate the process.
    Raises TimeoutError if the function times out.
    Raises RuntimeError if the function terminates with an error.

    Args:
        func: The function to run.
        args: The arguments to pass to the function.
        timeout: The timeout in seconds.

    Returns:
        The return value of the function.
    """

    print(
        f"\rRunning function '{func.__name__}' with timeout {timeout:.9f} seconds.",
        end="",
        flush=True,
    )

    def target_func(args: tuple, out: multiprocessing.Queue):
        try:
            result = func(*args)
            out.put(result)
        except TimeoutError as e:
            print(e, file=sys.stderr)
            out.put(e)
        except Exception as e:
            print(e, file=sys.stderr)
            traceback.print_exc()
            out.put(e)

    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=target_func, args=(args, q))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError("Function timed out and was terminated")
    if proc.exitcode != 0:
        print(args)
        print(proc.exitcode)
        raise RuntimeError("Function terminated with an error")
    try:
        result = q.get(timeout=1)
        if isinstance(result, Exception):
            raise result from result  # Re-raise the exception
        return result
    except Empty:
        return None


def get_matching_timeout(num_bytes: int) -> float:
    return num_bytes / THROUGHPUT_THRES + 1


def time_matching(
    sc_class: type[SuperConfigBase],
    automaton: PositionCountingAutomaton,
    random_str: str,
    cache_type: Literal["lru", "flush_on_full", "none"],
    counter_type: CounterType,
    sample_interval: int = 0,
) -> tuple[float, list]:
    t0 = time.perf_counter()
    match_found, cache_history = fullmatch(
        sc_class,
        automaton,
        random_str,
        counter_type,
        cache_type,
        sample_interval=0 if cache_type == "none" else sample_interval,
    )
    if not match_found:
        # Raise runtime error because all inputs should match (there are some edge cases where Xeger generates non-matching strings)
        raise NoMatchError("String did not match the automaton")
    t1 = time.perf_counter()
    duration = t1 - t0
    return duration, cache_history
