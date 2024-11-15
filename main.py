"""Main"""

import logging
import os
import re
import sys
import warnings

import xeger  # type: ignore

from src.position_automaton import PositionAutomaton


class XegerError(Exception):
    """XegerError"""

    def __init__(self, message: str):
        super().__init__(message)


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
    warnings.simplefilter(action="ignore", category=FutureWarning)

    x = xeger.Xeger(limit=10)
    for pattern in sys.stdin:
        pattern = pattern.strip()
        try:
            if "\0" in pattern:
                raise ValueError("Pattern contains null byte")
            try:
                text = x.xeger(pattern)
            except Exception as e:
                raise XegerError(
                    f"Xeger failed to generate a string for pattern {pattern}"
                ) from e

            logging.info("Pattern: %s", pattern)
            logging.info("Text: %s", text)
            automaton = PositionAutomaton.create(pattern)
            compiled = re.compile(pattern)

            thompson_result = automaton(text)
            backtrack_result = automaton.backtrack(text)
            re_result = compiled.fullmatch(text) is not None

            if re_result != thompson_result:
                logging.error(" Pattern: %s", pattern)
                logging.error("    Text: %s", text)
                logging.error("      re: %s", re_result)
                logging.error("Thompson: %s", thompson_result)

            if re_result != backtrack_result:
                logging.error("  Pattern: %s", pattern)
                logging.error("     Text: %s", text)
                logging.error("       re: %s", re_result)
                logging.error("Backtrack: %s", backtrack_result)

            thompson_prefix = automaton.match_prefix(text)
            backtrack_prefix = automaton.backtrack_prefix(text)
            match = compiled.match(text)
            re_prefix =  match.end(0) if match is not None else None

            if re_prefix != thompson_prefix:
                logging.error(" Pattern: %s", pattern)
                logging.error("    Text: %s", text)
                logging.error("      re: %s", re_prefix)
                logging.error("Thompson: %s", thompson_prefix)

            if re_prefix != backtrack_prefix:
                logging.error("  Pattern: %s", pattern)
                logging.error("     Text: %s", text)
                logging.error("       re: %s", re_prefix)
                logging.error("Backtrack: %s", backtrack_prefix)

        except (
            XegerError,
            ValueError,
            UnicodeEncodeError,
            NotImplementedError,
        ) as e:
            logging.info(str(e))
        except Exception as e:  # pylint: disable=broad-except
            logging.error(str(e))
            raise e
