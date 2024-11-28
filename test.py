"""Test"""

import logging
import os
import re
import sys
import warnings

from src.position_automaton import PositionAutomaton
from src.position_counter_automaton import PositionCounterAutomaton

if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
    warnings.simplefilter(action="ignore", category=FutureWarning)

    pattern = sys.argv[1]
    text = sys.argv[2]

    logging.info("Pattern: %s", pattern)
    logging.info("Text: %s", text)

    compiled = re.compile(pattern)
    re_result = compiled.fullmatch(text) is not None

    try:
        automaton = PositionAutomaton.create(pattern)
        thompson_result = automaton(text)
        backtrack_result = automaton.backtrack(text)

        if re_result != thompson_result or re_result != backtrack_result:
            logging.error("  Pattern: %s", pattern)
            logging.error("     Text: %s", text)
            logging.error("       re: %s", re_result)
            logging.error(" Thompson: %s", thompson_result)
            logging.error("Backtrack: %s", backtrack_result)
    except NotImplementedError as e:
        logging.error("Automaton Error: %s", e)

    try:
        counter_automaton = PositionCounterAutomaton.create(pattern)
        counter_backtrack_result = counter_automaton.backtrack(text)

        if re_result != counter_backtrack_result:
            logging.error("          Pattern: %s", pattern)
            logging.error("             Text: %s", text)
            logging.error("               re: %s", re_result)
            logging.error("Counter Backtrack: %s", counter_backtrack_result)
    except NotADirectoryError as e:
        logging.error("Counter Automaton Error: %s", e)
