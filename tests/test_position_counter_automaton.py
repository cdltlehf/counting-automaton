"""Unit tests for position_counter_automaton.py"""

import logging
import random
import re
import string
import unittest
import warnings

from timeout_decorator import timeout  # type: ignore
from timeout_decorator.timeout_decorator import TimeoutError  # type: ignore

import cai4py.counting_automaton.position_counting_automaton as pca
from cai4py.utils import load_test_cases


class TestPositionCountingAutomaton(unittest.TestCase):
    """Unit tests for position_counter_automaton.py"""

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.maxDiff = None  # pylint: disable=invalid-name
        dataset_path = "data/patterns/all_regexes.txt"
        self.test_cases = load_test_cases(dataset_path)
        self.timeout = 1

    def test_match(self) -> None:
        def modify_text(text: str) -> str:
            index = random.randint(0, len(text))
            character = random.choice(string.printable)
            modified_text = text[:index] + character + text[index:]
            return modified_text

        for pattern, texts in self.test_cases:
            modified_texts = [modify_text(text) for text in texts]
            try:
                compiled = re.compile(pattern)
            except re.error as re_error:
                logging.warning(
                    # print the error message,
                    # then print the pattern (with the location of the error coloured red)
                    "%s\n%s\033[91m%s\033[0m%s\nSkipping test case due to error above...",
                    re_error,
                    pattern[: re_error.pos],
                    pattern[re_error.pos],
                    pattern[re_error.pos + 1 :],
                )
                # Nothing more we can test if we can't compile the pattern
                continue
            automaton = pca.PositionCountingAutomaton.create(pattern)
            for text in texts + modified_texts:
                logging.debug(pattern)
                logging.debug(text)
                try:
                    re_result = timeout(self.timeout)(compiled.fullmatch)(text)
                except TimeoutError:
                    continue

                try:
                    pca_result = timeout(self.timeout)(automaton.match)(text)
                except TimeoutError:
                    logging.warning("Timeout Occurred")
                    continue

                self.assertEqual(pca_result, bool(re_result))
