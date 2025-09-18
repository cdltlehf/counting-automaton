"""Format jsonl to txt"""

import argparse
import json
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--empty-line-on-error",
        action="store_true",
        help="print an empty line for invalid lines",
    )
    args = parser.parse_args()

    for i, entry in enumerate(sys.stdin):
        try:
            line = json.loads(entry)
            if "\n" in line:
                raise ValueError("Unescaped line contains newline character")
            print(line, flush=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Line %d, %s", i, e)
            if args.empty_line_on_error:
                print(flush=True)
            continue


if __name__ == "__main__":
    main()
