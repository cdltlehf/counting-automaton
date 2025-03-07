import math

from counting_automaton.counting_set import CountingSet
from counting_automaton.counting_set import SparseCountingSet

if __name__ == "__main__":
    l = [3, 5, 6, 10, 13]
    for low in range(1, 10):
        for high in range(low, 10):
            k = high - low + 1
            print(f"low: {low}, high: {high}")
            counting_set = CountingSet.from_list(l, low, high)
            sparse_counting_set = SparseCountingSet.from_list(l, low, high)

            print(counting_set)
            print(sparse_counting_set)

            assert counting_set.check() == sparse_counting_set.check()
            length_bound = 2 * math.ceil(high / (k + 1))
            assert len(list(sparse_counting_set)) <= length_bound
