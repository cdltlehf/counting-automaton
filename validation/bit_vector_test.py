import unittest
import numpy as np
from numpy.testing import assert_array_equal
from cai4py.custom_counters.bit_vector import BitVector

class TestBitVector(unittest.TestCase):
    
    def test_init_bounded_64(self):
        counter = BitVector(3, 5)

        self.assertEqual(counter.lower_bound, 3)
        self.assertEqual(counter.upper_bound, 5)
        self.assertEqual(counter.bits, 0x8000000000000000)   # Only bit 63 set  
        self.assertEqual(counter.mask, 0x3800000000000000)  # 3 to 5 inclusive

    def test_init_unbounded_64(self):
        counter = BitVector(3, -1)

        self.assertEqual(counter.lower_bound, 3)
        self.assertEqual(counter.upper_bound, -1)
        self.assertEqual(counter.bits,  0x8000000000000000)   # Only bit 63 set  
        self.assertEqual(counter.mask, 0x4000000000000000)  # Only bit 62 set  

    def test_init_bounded(self):
        counter = BitVector(3, 65)

        self.assertEqual(counter.lower_bound, 3)
        self.assertEqual(counter.upper_bound, 65)
        self.assertEqual(counter.vector.size, 65)
        self.assertEqual(counter.vector[0], 1)    
        self.assertEqual(counter.size, 1)
        self.assertEqual(counter.start, 0)  
        self.assertEqual(counter.density, 1)
        self.assertEqual(counter.lowest_in_range, -1)

    def test_init_unbounded(self):
        counter = BitVector(65, -1)

        self.assertEqual(counter.lower_bound, 65)
        self.assertEqual(counter.upper_bound, -1)
        self.assertEqual(counter.vector.size, 64)
        self.assertEqual(counter.vector[0], 1)  
        self.assertEqual(counter.size, 1)
        self.assertEqual(counter.start, 0)  
        self.assertEqual(counter.density, 1)
        self.assertFalse(counter.overflow)

    def test_inc_bounded_64(self):
        counter = BitVector(3, 5)
        
        counter.inc()
        self.assertEqual(counter.bits, 0x4000000000000000)   # Bit 62 set

        counter.inc()
        self.assertEqual(counter.bits, 0x2000000000000000)   # Bit 61 set

        counter.inc()
        self.assertEqual(counter.bits, 0x1000000000000000)   # Bit 60 set

        counter.inc()
        self.assertEqual(counter.bits, 0x0800000000000000)   # Bit 59 set

        for _ in range(59):
            counter.inc()
        self.assertEqual(counter.bits, 0x0000000000000001)   # Bit 0 set

        counter.inc()
        self.assertEqual(counter.bits, 0x0000000000000000)   # No bits set

    def test_inc_unbounded_64(self):
        counter = BitVector(3, -1)
        
        counter.inc()   # Bit 62 - 2

        self.assertEqual(counter.bits, 0x4000000000000000)
        self.assertFalse(counter.overflow)

        counter.inc()   # Bit 61 - overflowed

        self.assertEqual(counter.bits, 0x2000000000000000)
        self.assertTrue(counter.overflow)

        for _ in range(61):
            counter.inc()   # Should reach bit 0

        self.assertEqual(counter.bits, 0x0000000000000001)
        self.assertTrue(counter.overflow)

        counter.inc()

        self.assertEqual(counter.bits, 0x0000000000000000)
        self.assertTrue(counter.overflow)

    # Check wrap properly
    def test_inc_bounded(self):
        counter = BitVector(60, 65)
        
        counter.inc()
        self.assertEqual(counter.start, 64)
        counter.inc()
        self.assertEqual(counter.start, 63)
        counter.inc()
        self.assertEqual(counter.start, 62)

    # Check wrap properly
    def test_inc_unbounded(self):
        counter = BitVector(65, -1)
        
        counter.inc()
        self.assertEqual(counter.start, 63)
        counter.inc()
        self.assertEqual(counter.start, 62)
        counter.inc()
        self.assertEqual(counter.start, 61)

    def test_union_bounded_64(self):
        counter1 = BitVector(3, 5)
        counter2 = BitVector(3, 5)

        counter1.inc() # Bit 62 set

        counter2.inc()
        counter2.inc() # Bit 61 set

        union_counter = BitVector.union(counter1, counter2)

        self.assertEqual(union_counter.bits, 0x6000000000000000)

    def test_union_unbounded_64(self):
        counter1 = BitVector(3, -1)
        counter2 = BitVector(3, -1)

        counter1.inc() # Bit 62 set

        counter2.inc()
        counter2.inc() # Bit 61 set

        union_counter = BitVector.union(counter1, counter2)

        self.assertEqual(union_counter.bits, 0x6000000000000000)

    def test_union_bounded(self):
        counter1 = BitVector(60, 65)
        counter2 = BitVector(60, 65)

        counter1.inc() # 2

        for _ in range(5):
            counter2.inc()  # 5

        union_counter = BitVector.union(counter1, counter2)

        expected_array = np.array([0, 1, 0, 0, 0, 1])  # replace with the correct expected values
        assert_array_equal(union_counter.vector[0:6], expected_array)

    def test_query_bounded_64(self):
        counter1 = BitVector(3, 5)
        counter1.inc()

        self.assertFalse(counter1.le_upper_bound()) # similar to the technicality case of hardware version
        self.assertFalse(counter1.ge_lower_bound())

        counter1.inc()
        counter1.inc()
        counter1.inc() # 4

        self.assertTrue(counter1.le_upper_bound())
        self.assertTrue(counter1.ge_lower_bound())

        counter1.inc()
        counter1.inc() # 6

        self.assertFalse(counter1.le_upper_bound())
        self.assertFalse(counter1.ge_lower_bound()) # bits are present but we don't count them


if __name__ == "__main__":
    unittest.main()