import signal
import time

import cai4py.counting_automaton.position_counting_automaton as pca


def timeout(seconds):
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()

        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = f(*args, **kwargs)
            finally:
                # reinstall the old signal handler
                signal.signal(signal.SIGALRM, old)
                # cancel the alarm
                # this line should be inside the "finally" block (per Sam Kortchmar)
                signal.alarm(0)
            return result

        new_f.__name__ = f.__name__
        return new_f

    return decorate


@timeout(seconds=10)
def timed_automaton_construction(regex, expansion_type):
    return pca.PositionCountingAutomaton.create(regex, expansion_type)


def time_matching(automaton, random_str, sc_class):
    t0 = time.perf_counter()
    # Step through matching
    for computation in sc_class.get_computation(automaton, random_str):
        pass  # do nothing
    assert computation is not None
    if not computation.is_final():
        pass
        # print("NOT FINAL")
        # print(
        #     re.fullmatch(regex, random_str) is not None
        # )
        # print()
        # print(f"'{regex}'", f"'{random_str}'", sep="\n")
    assert computation.is_final()
    t1 = time.perf_counter()
    duration = t1 - t0
    return duration
