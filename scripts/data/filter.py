"""Filter regexes with counting operations"""

import re
import sys

import cai4py.parser_tools as pt
import cai4py.parser_tools.utils as pt_utils
from cai4py.utils import escape
from cai4py.utils import unescape


def main() -> None:
    for line in sys.stdin:
        try:
            line = unescape(line.strip())
            parsed = pt.parse(line)
            normalized = pt.normalize(parsed)

            counting_height = pt_utils.counting_height(normalized)
            if counting_height != 1:
                continue

            stringified = pt.to_string(normalized)
            print(escape(stringified))
        except (re.error, OverflowError, ValueError):
            continue


if __name__ == "__main__":
    main()
