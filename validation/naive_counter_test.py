import unittest
from cai4py.custom_counters.naive_counter import NaiveCounter

class TestNaiveCounter(unittest.TestCase):
    
    def test_init(self):
        counter = NaiveCounter(3, 5)

        self.assertEqual(counter.lower_bound, 3)
        self.assertEqual(counter.upper_bound, 5)
        self.assertFalse(counter.overflow)
        self.assertEqual(counter.list, [1])

    def test_inc_bounded(self):
        counter = NaiveCounter(3, 5)
        
        counter.inc()
        self.assertEqual(counter.list, [2])
        self.assertFalse(counter.overflow)

    def test_inc_unbounded(self):
        counter = NaiveCounter(3, -1)
        
        counter.inc()   # [2]

        self.assertEqual(counter.list, [2])
        self.assertFalse(counter.overflow)

        counter.inc()   # []*

        self.assertEqual(counter.list, []) # Should only keep elements < lower_bound
        self.assertTrue(counter.overflow)

        # Incrementing further should not change anything
        counter.inc()   # []*

        self.assertEqual(counter.list, [])
        self.assertTrue(counter.overflow)

    def test_union_bounded(self):
        counter1 = NaiveCounter(3, 5)
        counter2 = NaiveCounter(3, 5)

        counter1.inc()  # [2]
        counter1.inc()  # [3]
        
        counter2.inc()  # [2]
        counter2.inc()  # [3]
        counter2.inc()  # [4]

        union_counter = NaiveCounter.union(counter1, counter2)

        self.assertEqual(union_counter.list, [3, 4])
        self.assertFalse(union_counter.overflow)

    def test_union_unbounded(self):
        counter1 = NaiveCounter(3, -1)
        counter2 = NaiveCounter(3, -1)

        counter1.inc()  # [2]
        counter1.inc()  # []*
        
        counter2.inc()  # [2]
       
        union_counter = NaiveCounter.union(counter1, counter2)

        self.assertEqual(union_counter.list, [2])
        self.assertTrue(union_counter.overflow)

        union_counter.inc()  # []*
        union_counter.inc()  # []*

        self.assertTrue(union_counter.overflow)

    def test_query_bounded(self):
        counter = NaiveCounter(3, 5)

        counter.inc()  # [2]

        self.assertFalse(counter.ge_lower_bound())
        self.assertTrue(counter.le_upper_bound())

        counter.inc()  # [3]

        self.assertTrue(counter.ge_lower_bound())
        self.assertTrue(counter.le_upper_bound())

        counter.inc()  # [4]
        counter.inc()  # [5]

        self.assertTrue(counter.ge_lower_bound())
        self.assertTrue(counter.le_upper_bound())

        counter.inc()  # []

        self.assertFalse(counter.ge_lower_bound())
        self.assertFalse(counter.le_upper_bound())

    def test_query_unbounded(self):
        counter = NaiveCounter(3, -1)

        counter.inc()  # [2]

        self.assertFalse(counter.ge_lower_bound())

        counter.inc()  # []*

        self.assertTrue(counter.ge_lower_bound())

        counter.inc()  # []*

        self.assertTrue(counter.ge_lower_bound())

if __name__ == "__main__":
    unittest.main()