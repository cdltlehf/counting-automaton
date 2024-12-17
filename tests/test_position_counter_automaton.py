"""Unit tests for position_counter_automaton.py"""
import logging
import random
import re
import string
import unittest
import warnings

from timeout_decorator import timeout  # type: ignore
from timeout_decorator.timeout_decorator import TimeoutError  # type: ignore

import src.position_counter_automaton as pca
from src.utils import load_test_cases


class TestPositionCounterAutomaton(unittest.TestCase):
    """Unit tests for position_counter_automaton.py"""
    def setUp(self) -> None:
        # logging.basicConfig(level=logging.DEBUG)
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.maxDiff = None  # pylint: disable=invalid-name
        # dataset_path = "data/test_cases/polyglot.txt"
        dataset_path = "data/test_cases/snort30.txt"
        self.test_cases = load_test_cases(dataset_path)
        self.timeout = 10

    def test_match(self) -> None:
        def modify_text(text: str) -> str:
            index = random.randint(0, len(text))
            character = random.choice(string.printable)
            modified_text = text[:index] + character + text[index:]
            return modified_text

        for pattern, texts in self.test_cases:
            modified_texts = [modify_text(text) for text in texts]
            automaton = pca.PositionCounterAutomaton.create(pattern)
            compiled = re.compile(pattern)
            for text in texts + modified_texts:
                try:
                    re_result = timeout(self.timeout)(compiled.fullmatch)(text)
                except TimeoutError:
                    continue
                pca_result = timeout(self.timeout)(automaton.match)(text)
                self.assertEqual(pca_result, bool(re_result))
