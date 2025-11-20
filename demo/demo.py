import cai4py.counting_automaton.position_counting_automaton as pca
from cai4py.custom_counters.counter_type import CounterType
import cai4py.parser_tools as pt
import cai4py.counting_automaton.super_config.super_config as sc

import argparse

# Used to demo the program for different counter methods and regexes.
def main() -> None:

    parser = argparse.ArgumentParser(description="Demo to test specific regexes.")
    parser.add_argument("--method", type=int, default=1, help="Select counter method: 1 - Bit Vector; 2 - Naive Counter; 3 - Counting-set; 4 - Sparse Counting-set; 5 - Counter Expansion")

    args = parser.parse_args()

    # Method selection
    match args.method:
        case 1:
            counter = CounterType.BIT_VECTOR
            normaliser = pt.flatten_inner_quantifiers
        case 2:
            counter = CounterType.NAIVE_COUNTER
            normaliser = pt.flatten_inner_quantifiers
        case 3:
            counter = CounterType.COUNTING_SET
            normaliser = pt.flatten_inner_quantifiers
        case 4:
            counter = CounterType.SPARSE_COUNTING_SET
            normaliser = pt.flatten_inner_quantifiers
        case 5:
            counter = CounterType.BIT_VECTOR
            normaliser = pt.flatten_quantifiers
        case _:
            print(f"Unknown method selected: {args.method}")
            exit()

    print("Enter regular expression:")
    regex = input()

    print("Enter word to match:")
    word = input()
    print("")

    # Parse regex and setup automaton
    parsed_regex = pt.parse(regex)
    normalised_regex = normaliser(parsed_regex)
    print(f"Normalised regex:\n{normalised_regex}")
    pattern = pt.to_string(normalised_regex)
    print(f"Pattern: {pattern}")

    automaton = pca.PositionCountingAutomaton.create(pattern)
    print(f"Automaton: {automaton} \n")

    matcher = sc.SuperConfig(automaton, counter)
    has_matched, _ = matcher.match(word)
    print(f"Match: {has_matched}")

if __name__ == "__main__":
    main()