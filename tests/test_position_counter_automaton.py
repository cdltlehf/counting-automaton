"""Unit tests for position_counter_automaton.py"""

import pickle
import logging
import random
import re
import string
import unittest
import warnings

from timeout_decorator import timeout  # type: ignore

import cai4py.counting_automaton.position_counting_automaton as pca
from cai4py.utils import load_test_cases


class TestPositionCountingAutomaton(unittest.TestCase):
    """Unit tests for position_counter_automaton.py"""

    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.maxDiff = None  # pylint: disable=invalid-name
        dataset_path = "../data/filtered/all_regexes.txt"
        self.test_cases = load_test_cases(dataset_path)
        self.timeout = 1

    def test_pickle(self) -> None:
        from pickle import dumps as pickle_dumps

        for pattern, _ in self.test_cases:
            try:
                automaton = pca.PositionCountingAutomaton.create(pattern)
                pickle.dumps(list(automaton.states.keys()))
                vals = list(automaton.states.values())
                print(vals)
                pickle.dumps(vals)
                serialized = pickle_dumps(automaton)
                deserialized = pickle.loads(serialized)
                self.assertEqual(automaton, deserialized)
            except ValueError as ve:
                print(ve)
                continue

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
