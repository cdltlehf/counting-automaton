"""Normalize unicode regexes"""

import logging
import sys

import cai4py.parser_tools as pt


def main() -> None:
    for i, pattern in enumerate(map(str.rstrip, sys.stdin), 1):
        try:
            parsed = pt.parse(pattern)
            normalized = pt.unicode_normalize(parsed)
            stringified = pt.to_string(normalized)
            print(stringified, flush=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Line %d: %s", i, e)
            continue


if __name__ == "__main__":
    main()
