import unittest
from cai4py.custom_counters.counting_set import CountingSet

class TestCountingSet(unittest.TestCase):

    def test_init(self):
        counter = CountingSet(3, 5)

        self.assertEqual(counter.lower_bound, 3)
        self.assertEqual(counter.upper_bound, 5)
        self.assertFalse(counter.overflow)
        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 0)

    def test_inc_bounded(self):
        counter = CountingSet(3, 5)
        
        counter.inc()   # [2]

        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 1)
        
        counter.inc()   # [3]
        counter.inc()   # [4]
        counter.inc()   # o=4, l=[1]

        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 4)

    def test_inc_unbounded(self):
        counter = CountingSet(3, -1)
        
        counter.inc()   # o=1, l=[1]

        self.assertEqual(list(counter.queue), [1])
        self.assertEqual(counter.offset, 1)
        self.assertFalse(counter.overflow)

        counter.inc()   # o=2, l=[]*

        self.assertEqual(list(counter.queue), [])
        self.assertEqual(counter.offset, 2)
        self.assertTrue(counter.overflow)

        # Incrementing further should not change anything
        counter.inc()   # []*

        self.assertEqual(list(counter.queue), [])
        self.assertEqual(counter.offset, 2)
        self.assertTrue(counter.overflow)

    def test_union_bounded(self):
        counter1 = CountingSet(3, 5)
        counter2 = CountingSet(3, 5)

        counter1.inc()  # o=1, l=[1]
        counter1.inc()  # o=2, l=[1]
        
        counter2.inc()  # o=1, l=[1]
        counter2.inc()  # o=2, l=[1]
        counter2.inc()  # o=3, l=[1]

        union_counter = CountingSet.union(counter1, counter2)

        self.assertEqual(list(union_counter.queue), [3, 4])
        self.assertEqual(union_counter.offset, 0)
        self.assertFalse(union_counter.overflow)

    def test_union_unbounded(self):
        counter1 = CountingSet(3, -1)
        counter2 = CountingSet(3, -1) # o=0, l=[1]

        counter1.inc()  # o=1, l=[1]
       
        union_counter = CountingSet.union(counter1, counter2)

        self.assertEqual(list(union_counter.queue), [1, 2])
        self.assertEqual(union_counter.offset, 0)
        self.assertFalse(union_counter.overflow)

    def test_union_empty_set(self):
        counter1 = CountingSet(3, -1)
        counter2 = CountingSet(3, -1) # o=0, l=[1]

        counter1.inc()  # o=1, l=[1]
        counter1.inc()  # o=2, l=[1]
        counter1.inc()  # o=2, l=[]*
       
        union_counter = CountingSet.union(counter1, counter2)

        self.assertEqual(list(union_counter.queue), [1])
        self.assertEqual(union_counter.offset, 0)
        self.assertTrue(union_counter.overflow)

    def test_query_bounded(self):
        counter = CountingSet(3, 5)

        counter.inc()   # o=1, l=[1]
    
        self.assertFalse(counter.ge_lower_bound())
        self.assertTrue(counter.le_upper_bound())

        counter.inc()  # o=2, l=[1]

        self.assertTrue(counter.ge_lower_bound())
        self.assertTrue(counter.le_upper_bound())

        counter.inc()  # o=3, l=[1]
        counter.inc()  # o=4, l=[1]

        self.assertTrue(counter.ge_lower_bound())
        self.assertTrue(counter.le_upper_bound())

        counter.inc()  # o=5, l=[]

        self.assertFalse(counter.ge_lower_bound())
        self.assertFalse(counter.le_upper_bound())

    def test_query_unbounded(self):
        counter = CountingSet(3, -1)

        counter.inc()  # o=1, l=[1]

        self.assertFalse(counter.ge_lower_bound())

        counter.inc()  # o=2, l=[]*

        self.assertTrue(counter.ge_lower_bound())

        counter.inc()  # o=2, l=[]*

        self.assertTrue(counter.ge_lower_bound())   


if __name__ == "__main__":
    unittest.main()