"""Format txt to jsonl"""

import json
import sys


def main() -> None:
    for line in map(str.rstrip, sys.stdin):
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
