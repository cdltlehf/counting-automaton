import argparse

from cai4py.parser_tools import parse
from cai4py.parser_tools import to_string
from cai4py.parser_tools.utils import flatten_inner_quantifiers
from cai4py.parser_tools.utils import flatten_quantifiers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", type=str)
    args = parser.parse_args()

    parsed = parse(args.pattern)
    print(args.pattern)
    print("Parsed", to_string(parsed), sep="\n")
    print("Double Parsed", to_string(parse(to_string(parsed))), sep="\n")
    print("Flatten", to_string(flatten_quantifiers(parsed)), sep="\n")
    print(
        "Inner flatten", to_string(flatten_inner_quantifiers(parsed)), sep="\n"
    )


if __name__ == "__main__":
    main()
