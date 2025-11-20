"""Unit tests for parser_tools.py"""

import logging
import re
import unittest
import warnings
from pathlib import Path

import cai4py.parser_tools as pt
from cai4py.utils import unescape


class TestParserTools(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO)
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.maxDiff = None  # pylint: disable=invalid-name
        test_dir = Path(__file__).parent
        dataset_path = test_dir / "regexes.txt"
        with open(str(dataset_path), "r", encoding="utf-8") as f:
            self.test_cases = [line[:-1] for line in f.readlines()]
        self.timeout = 1

    def test_to_string(self) -> None:
        total = 0
        succeeded = 0
        for line in self.test_cases:
            total += 1
            pattern = line.rstrip("\n")
            try:
                parsed = pt.parse(pattern)
            except (OverflowError, re.error):
                continue

            try:
                printed = pt.to_string(parsed)
                reparsed = pt.parse(printed)
                self.assertEqual(str(parsed.dump()), str(reparsed.dump()))
                self.assertEqual(str(parsed), str(reparsed))
            except NotImplementedError:
                continue
            succeeded += 1
        print(f"Total: {total}, Succeeded: {succeeded}")
