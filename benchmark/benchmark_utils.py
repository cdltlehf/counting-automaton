from multiprocessing import Process, Queue
from typing import Optional

import cai4py.counting_automaton.position_counting_automaton as pca
from cai4py.custom_counters.counter_type import CounterType
import cai4py.parser_tools as pt
import cai4py.counting_automaton.super_config.super_config as sc

import time

"""
    Perform the actual runtime and density experiments.
"""

flatten = "Counter Expansion"

# For runtime benchmarks
methods = [
    CounterType.NAIVE_COUNTER,
    CounterType.BIT_VECTOR,
    CounterType.COUNTING_SET,
    CounterType.SPARSE_COUNTING_SET
]

# For maximum density benchmarks
density_methods = [
    CounterType.COUNTING_SET,
    CounterType.SPARSE_COUNTING_SET
]

method_colour = {
    flatten: "green",
    CounterType.NAIVE_COUNTER: "red",
    CounterType.BIT_VECTOR: "blue",
    CounterType.COUNTING_SET: "orange",
    CounterType.SPARSE_COUNTING_SET: "purple"
}

# Perform match and measure time
def match_process(method: str, word: str, regex: str, queue: Queue):

    try:
        normalizer = pt.flatten_quantifiers if method is flatten else pt.flatten_inner_quantifiers
        pattern = pt.parse(regex)

        normalized_pattern = normalizer(pattern, depth=50)
        regex = pt.to_string(normalized_pattern)

        automaton = pca.PositionCountingAutomaton.create(regex)
        matcher = sc.SuperConfig(automaton, CounterType.NAIVE_COUNTER) if method is flatten else sc.SuperConfig(automaton, method)
    except:
        print("Could not construct automaton.")
        queue.put(0)
        return queue

    start = time.process_time()
    matcher.match(word)
    end = time.process_time()
    queue.put(end - start)
    return queue

def runtime(method: str, word: str, regex: str, timeout: float = 15.0) -> Optional[float]:
  
    runs = 3
    total_time = 0.0
    for _ in range(runs):   # Do three runs
        queue = Queue()
        p = Process(target=match_process, args=(method, word, regex, queue))
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            print("Timed out!")
            return None  # indicate timeout

        if not queue.empty():
            total_time += queue.get()
            queue.close()
        else:
            print("No result returned!")
            return None

    return total_time / runs

# Experiments that only require data extraction not runtime
def data_experiment(regex: str, word: str, method: str):

    normalizer = pt.flatten_quantifiers if method is flatten else pt.flatten_inner_quantifiers
    pattern = pt.parse(regex)

    normalized_pattern = normalizer(pattern, depth=50)
    regex = pt.to_string(normalized_pattern)

    automaton = pca.PositionCountingAutomaton.create(regex)
    matcher = sc.SuperConfig(automaton, CounterType.NAIVE_COUNTER) if method is flatten else sc.SuperConfig(automaton, method)

    _, data_collection = matcher.match(word)

    return data_collection

def extract_max_density(data_collection) -> int:
    return data_collection["Maximum Density"]
