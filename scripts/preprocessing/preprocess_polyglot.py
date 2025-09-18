"""scripts/preprocessing/preprocess_polyglot.py"""

import json
import logging
import sys


def main() -> None:
    for i, entry in enumerate(map(json.loads, sys.stdin), 1):
        try:
            pattern = entry["pattern"]
            if "\n" in pattern:
                raise ValueError("Unescaped line contains newline character")
            print(pattern, flush=True)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Line %d: %s", i, e)
            continue


if __name__ == "__main__":
    main()
