import unittest
from cai4py.parser_tools.parser import SubPattern
from cai4py.parser_tools.constants import LITERAL
from cai4py.counting_automaton.position_counting_automaton import (
    PositionCountingAutomaton,
    State,
)


class TestEvalState(unittest.TestCase):
    def test_eval_state_with_literal(self):
        # Create a SubPattern with a LITERAL opcode
        subpattern = SubPattern(None, data=[(LITERAL, ord("a"))])

        # Create a PositionCountingAutomaton instance
        automaton = PositionCountingAutomaton(
            states={State(1): subpattern},
            follow={},
        )

        # Test eval_state
        result = automaton.eval_state(State(1), "a")
        self.assertTrue(result)

        result = automaton.eval_state(State(1), "b")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
