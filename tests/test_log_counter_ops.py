import os
import unittest

import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.counting_automaton.super_config as sc


class TestLogCounterOps(unittest.TestCase):
    TEST_REGEXES = [
        # Patterns with counters
        r"a{3}",
        r"(ab){2,5}",
        r"[0-9]{4,6}",
        r"(cat|dog){1,3}",
        r"(a|b|c){2,}",
    ]
    TEST_INPUTS = ["a" * 3, "ab" * 4, "12345", "cat" * 2, "a" * 2 + "b" * 10]
    EXPECTED_INCREMENT_COUNTS = [2, 3, 4, 1, 11]

    def test_increment_logging(self):
        def count_increment_logs(log_file):
            count = 0
            for line in log_file:
                if line.startswith("INCREASE"):
                    count += 1
            return count

        for regex, test_input, expected_count in zip(
            self.TEST_REGEXES, self.TEST_INPUTS, self.EXPECTED_INCREMENT_COUNTS
        ):
            automaton = pca.PositionCountingAutomaton.create(
                regex, expansion_type="inner"
            )
            with open("temp-op-log.txt", "w", encoding="utf-8") as log_file:
                for _ in sc.CounterConfig.get_computation(
                    automaton, test_input, log_file=log_file
                ):
                    pass

            with open("temp-op-log.txt", "r", encoding="utf-8") as log_file:
                count = count_increment_logs(log_file)
                self.assertEqual(count, expected_count)
        os.remove("temp-op-log.txt")
