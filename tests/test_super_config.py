"""Unit tests that run examples (to get coverage reports for each method)"""

import logging
import unittest
import warnings

from cai4py.utils import load_test_cases
from cai4py.scripts.analysis.computation_info import run_and_log_trace
import cai4py.counting_automaton.super_config as sc


class TestSuperConfig(unittest.TestCase):
    """Unit tests for position_counter_automaton.py"""

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self.maxDiff = None  # pylint: disable=invalid-name
        dataset_path = "data/test-cases/examples.tsv"
        self.test_cases = load_test_cases(dataset_path)
        self.timeout = 1

    def test_super_config(self):
        run_and_log_trace(sc.SuperConfig, self.test_cases)

    def test_bounded_super_config(self):
        run_and_log_trace(sc.BoundedSuperConfig, self.test_cases)

    def test_counter_config(self):
        run_and_log_trace(sc.CounterConfig, self.test_cases)

    def test_bounded_counter_config(self):
        run_and_log_trace(sc.BoundedCounterConfig, self.test_cases)

    def test_sparse_counter_config(self):
        run_and_log_trace(sc.SparseCounterConfig, self.test_cases)

    def test_determinized_counter_config(self):
        run_and_log_trace(sc.DeterminizedCounterConfig, self.test_cases)

    def test_determinized_bounded_counter_config(self):
        run_and_log_trace(sc.DeterminizedBoundedCounterConfig, self.test_cases)
