"""Normalize a regular expression pattern."""

import re
import sys

import src.parser_tools as pt
from src.utils import escape
from src.utils import unescape


def main() -> None:
    for pattern in map(unescape, sys.stdin):
        try:
            parsed = pt.parse(pattern)
        except (
            re.error,
            OverflowError,
        ):
            continue
        try:
            normalized = pt.normalize(parsed)
        except NotImplementedError as e:
            print(e, file=sys.stderr)
            continue
        print(escape(pt.to_string(normalized)))


if __name__ == "__main__":
    main()
