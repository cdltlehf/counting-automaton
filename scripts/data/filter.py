"""Filter regexes with counting operations"""

import re
import sys

import cai4py.parser_tools as pt
import cai4py.parser_tools.utils as pt_utils
from cai4py.utils import escape, unescape


def main() -> None:
    for line in sys.stdin:
        try:
            parsed = pt.parse(unescape(line))
            normalized = pt.normalize(parsed)
            if pt_utils.counting_height(normalized) != 1:
                continue
            print(escape(pt.to_string(normalized)))
        except NotImplementedError as e:
            print(e, file=sys.stderr)
            continue
        except (re.error, OverflowError, ValueError):
            continue


if __name__ == "__main__":
    main()
