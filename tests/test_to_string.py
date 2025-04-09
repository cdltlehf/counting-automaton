"""Unit tests for parser_tools.py"""

import re
import unittest
import warnings

import cai4py.parser_tools as pt


class TestParserTools(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.maxDiff = None
        self.dataset = open(
            "data/patterns/all_regexes.txt", "r", encoding="unicode_escape"
        )

    def test_to_string(self) -> None:
        total = 0
        succeeded = 0
        for line in self.dataset.readlines():
            total += 1
            pattern = line.rstrip("\n")
            try:
                parsed = pt.parse(pattern)
            except (
                OverflowError,
                re.error,
            ):
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
