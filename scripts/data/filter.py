"""Filter regexes with counting heights"""

import logging
import sys

import cai4py.parser_tools as pt
import cai4py.parser_tools.utils as pt_utils


def main() -> None:
    for i, pattern in enumerate(map(lambda e: e.rstrip("\r\n"), sys.stdin), 1):
        try:
            parsed = pt.parse(pattern)
            counting_height = pt_utils.counting_height(parsed)

            if counting_height == 0:
                logging.info(
                    "Line %d: Counting height is %d", i, counting_height
                )
                continue

            print(pattern, flush=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Line %d: %s", i, e)
            continue


if __name__ == "__main__":
    main()
