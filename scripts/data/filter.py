"""Filter regexes with counting operations"""

import re
import sys

import parser_tools as pt
import parser_tools.utils
from utils import escape
from utils import unescape


def main() -> None:
    for line in sys.stdin:
        try:
            parsed = pt.parse(unescape(line))
            normalized = pt.normalize(parsed)
            if parser_tools.utils.counting_height(normalized) != 1:
                continue
            print(escape(pt.to_string(normalized)))
        except NotImplementedError as e:
            print(e, file=sys.stderr)
            continue
        except (re.error, OverflowError, ValueError):
            continue


if __name__ == "__main__":
    main()
