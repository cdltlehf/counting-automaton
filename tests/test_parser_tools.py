"""Unit tests for parser_tools.py"""

import re
import unittest
import warnings

import cai4py.parser_tools as pt
from cai4py.utils import unescape


class TestParserTools(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.maxDiff = None  # pylint: disable=invalid-name
        self.dataset = open("data/filtered/all_regexes.txt", "r", encoding="utf-8")
        self.patterns = (unescape(line) for line in self.dataset.readlines())

    def test_to_string(self) -> None:
        total = 0
        succeeded = 0
        for line in self.patterns:
            total += 1
            pattern = line.rstrip("\n")
            try:
                parsed = pt.parse(pattern)
            except (OverflowError, re.error):
                continue

            try:
                printed = pt.to_string(parsed)
                reparsed = pt.parse(printed)
                self.assertEqual(str(parsed), str(reparsed))
            except NotImplementedError:
                continue
            succeeded += 1
        print(f"Total: {total}, Succeeded: {succeeded}")

    def tearDown(self) -> None:
        self.dataset.close()
