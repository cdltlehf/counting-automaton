"""Unit tests for parser_tools.py"""

import unittest

import cai4py.parser_tools as pt
import cai4py.parser_tools.utils as utils


class TestNestedCounterExpansion(unittest.TestCase):
    regexes = [r"(?:(?:a){2,3}b){4,5}", r"(?:(?:a|b){2,4}){2,4}"]
    inner_expansions = [
        r"(?:(?:a)(?:a)(?:a)?b){4,5}",
        r"(?:(?:(?:a|b)(?:a|b)(?:a|b)?(?:a|b)?)){2,4}",
    ]
    outer_expansions = [
        r"(?:(?:a{2,3}b)(?:a{2,3}b)(?:a{2,3}b)(?:a{2,3}b)(?:a{2,3}b)?)",
        r"(?:(?:a|b){2,4})(?:(?:a|b){2,4})(?:(?:a|b){2,4})?(?:(?:a|b){2,4})?"
    ]
    full_expansions = [
        4*r"(?:(?:(?:a)(?:a)(?:a)?b))" + r"(?:(?:(?:a)(?:a)(?:a)?b))?",
       2*r"(?:(?:(?:a|b)(?:a|b)(?:a|b)?(?:a|b)?))"  + r"(?:(?:(?:a|b)(?:a|b)(?:a|b)?(?:a|b)?))?" * 2
    ]

    def test_full_expansion(self):
        for i, regex in enumerate(self.regexes):
            tree = pt.parse(regex)
            tree.dump()
            tree = utils.expand_counters(tree, method="full")
            expected_tree = pt.parse(self.full_expansions[i])
            if str(tree) != str(expected_tree):
                tree.dump()
                expected_tree.dump()
                raise RuntimeError("Full expansion did not match expected output")

    def test_inner_expansion(self):
        for i, regex in enumerate(self.regexes):
            tree = pt.parse(regex)
            tree.dump()
            tree = utils.expand_counters(tree, method="inner")
            expected_tree = pt.parse(self.inner_expansions[i])
            if str(tree) != str(expected_tree):
                print(tree)
                tree.dump()
                expected_tree.dump()
                assert False

    def test_outer_expansion(self):
        for i, regex in enumerate(self.regexes):
            tree = pt.parse(regex)
            tree.dump()
            tree = utils.expand_counters(tree, method="outer")
            expected_tree = pt.parse(self.outer_expansions[i])
            if str(tree) != str(expected_tree):
                tree.dump()
                expected_tree.dump()
                assert False

    def test_inner_expansion_captures(self):
        regex = r"((a){2,3}){2,3}"
        tree = pt.parse(regex)
        tree.dump()
        tree = utils.expand_counters(tree, method="inner")
        tree.dump()

    def test_outer_expansion_captures(self):
        regex = r"((a){2,3}){2,3}"
        tree = pt.parse(regex)
        tree.dump()
        tree = utils.expand_counters(tree, method="outer")
        tree.dump()