import sys

from cai4py.utils import unescape


def main() -> None:
    for line in sys.stdin:
        try:
            pattern = unescape(line.strip())
            if "\n" in pattern:
                raise ValueError("Pattern contains newline character")
            print(pattern)
        except Exception:
            continue


if __name__ == "__main__":
    main()
