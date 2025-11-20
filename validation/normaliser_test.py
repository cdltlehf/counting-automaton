import unittest
import cai4py.parser_tools as pt

class TestParserTools(unittest.TestCase):

    def test_flatten_inner_quantifiers_basic(self):
        regex = pt.parse("a{2,5}")
        flattened = pt.flatten_inner_quantifiers(regex)
        self.assertEqual(pt.to_string(flattened), "(a){2,5}")

        regex = pt.parse("(a{5}){3}")
        flattened = pt.flatten_inner_quantifiers(regex)
        self.assertEqual(pt.to_string(flattened), "((aaaaa)){3,3}")

    def test_flatten_quantifiers(self):
        regex = pt.parse("a{2,5}")
        flattened = pt.flatten_quantifiers(regex)
        self.assertEqual(pt.to_string(flattened), "aa(a(aa?)?)?")

        regex = pt.parse("a{2,4}")
        flattened = pt.flatten_quantifiers(regex)
        self.assertEqual(pt.to_string(flattened), "aa(aa?)?")

        regex = pt.parse("(a{5}){3}")
        flattened = pt.flatten_quantifiers(regex)
        self.assertEqual(pt.to_string(flattened), "(((aaaaa))((aaaaa))((aaaaa)))")

    def test_bounded_expansion(self):
        regex = pt.parse("a{2,12}")
        flattened = pt.flatten_quantifiers(regex, depth=5)

        expected = "aa" + "(a(a(a(aa?)?)?)?)?" + "(a(a(a(aa?)?)?)?)?"
        self.assertEqual(pt.to_string(flattened), expected)

        regex = pt.parse("b{5,15}")
        flattened = pt.flatten_quantifiers(regex, depth=9)

        expected = "bbbbb" + "(b(b(b(b(b(b(b(bb?)?)?)?)?)?)?)?)?b?"
        self.assertEqual(pt.to_string(flattened), expected)

    def test_complex_expansion(self):
        regex = pt.parse("(ab{2}c){3,4}")
        flattened = pt.flatten_inner_quantifiers(regex, depth=4)

        expected = "(a(bb)c){3,4}"
        self.assertEqual(pt.to_string(flattened), expected)

        regex = pt.parse("((x{1,2}z){2}){2,6}")
        flattened = pt.flatten_inner_quantifiers(regex)

        expected = "(((xx?z)(xx?z))){2,6}"
        self.assertEqual(pt.to_string(flattened), expected)

if __name__ == "__main__":
    unittest.main()