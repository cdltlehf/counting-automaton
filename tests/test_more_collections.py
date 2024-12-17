"""Unit tests for more_collections.py"""
import unittest

from src.more_collections import OrderedSet


class TestOrderedSet(unittest.TestCase):
    def test_substitute(self) -> None:
        ordered_set = OrderedSet([1, 2, 3, 4])
        ordered_set.substitute(2, [5, 6, 7])
        self.assertEqual(list(ordered_set), [1, 5, 6, 7, 3, 4])
