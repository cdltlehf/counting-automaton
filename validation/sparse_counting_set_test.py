from collections import deque
import unittest
from cai4py.custom_counters.sparse_counting_set import SparseCountingSet

class TestSparseCountingSet(unittest.TestCase):

    def test_init(self):
        counter = SparseCountingSet(3, 5)

        self.assertEqual(counter.lower_bound, 3)
        self.assertEqual(counter.upper_bound, 5)
        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 0)

    def test_inc_bounded(self):
        counter = SparseCountingSet(3, 5)
        
        counter.inc()   # o=1, l=[1]

        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 1)
        
        counter.inc()   # o=2, l=[1]
        counter.inc()   # o=3, l=[1]
        counter.inc()   # o=4, l=[1]

        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 4)

    def test_inc_unbounded(self):
        counter = SparseCountingSet(3, -1)
        
        counter.inc()   # o=1, l=[1]

        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 1)

        counter.inc()   # o=2, l=[1]

        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 2)

        # Value should keep increasing, because of unbounded case
        counter.inc()   # o=3, l=[1]

        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 3)

    def test_union_bounded(self):
        counter1 = SparseCountingSet(3, 10)
        counter2 = SparseCountingSet(3, 10)

        for _ in range(8):
         counter1.inc()  # o=8, l=[1]
        
        counter2.inc()  # o=1, l=[1]
        counter2.inc()  # o=2, l=[1]

        union_counter = SparseCountingSet.union(counter1, counter2) # [3, 9]

        self.assertEqual(list(union_counter.queue), [3])
        self.assertEqual(union_counter.offset, 0)

    def test_union_special_case(self):
        counter1 = SparseCountingSet(12, 16)
        counter2 = SparseCountingSet(12, 16)

        counter1.queue = deque([1, 4, 13, 15])
        counter2.queue = deque([6, 9, 10])

        union_counter = SparseCountingSet.union(counter1, counter2) # [1, 4, 6, 9, 13]

        self.assertEqual(list(union_counter.queue), [1, 4, 6, 9, 13])
        self.assertEqual(union_counter.offset, 0)

    def test_union_unbounded(self):
        counter1 = SparseCountingSet(5, -1)
        counter2 = SparseCountingSet(5, -1)

        for _ in range(8):
         counter1.inc()  # o=8, l=[1]
        
        counter2.inc()  # o=1, l=[1]
        counter2.inc()  # o=2, l=[1]

        union_counter = SparseCountingSet.union(counter1, counter2) # [9]

        self.assertEqual(list(union_counter.queue), [9])
        self.assertEqual(union_counter.offset, 0)

if __name__ == "__main__":
    unittest.main()