import sys
import os

# Add the parent directory of 'src' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import cai4py.counting_automaton.position_counting_automaton as pca
from cai4py.custom_counters.counter_type import CounterType
import cai4py.parser_tools as pt
import cai4py.counting_automaton.super_config.super_config as sc

import argparse

# Used to test specific regexes and input strings.

def main() -> None:

    parser = argparse.ArgumentParser(description="Demo to test specific regexes.")
    parser.add_argument("--method", type=int, default=1, help="Select counter method: 1 - bit vector; 2 - lazy counting set; 3 - counting set; 4 - sparse counting set; 5 - counter expansion")

    args = parser.parse_args()

    # Method selection
    match args.method:
        case 1:
            counter = CounterType.BIT_VECTOR
            normaliser = pt.flatten_inner_quantifiers
        case 2:
            counter = CounterType.LAZY_COUNTING_SET
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

    print("Enter regex pattern:") #".*a{0,10}a{2}"
    regex = ".*a.{10}c"
    regex = ".*a{2,10}a{2}"

    print("Enter word to match:")
    word = "bbaaaaaaaa"

    # Parse regex and setup automaton
    parsed_regex = pt.parse(regex)
    normalised_regex = normaliser(parsed_regex)
    pattern = pt.to_string(normalised_regex)
    print(f"Pattern: {pattern}")

    automaton = pca.PositionCountingAutomaton.create(pattern)
    print(f"Automaton: {automaton} \n")

    matcher = sc.SuperConfig(automaton, counter)
    has_matched, _ = matcher.match(word)
    print(f"Match: {has_matched}")

if __name__ == "__main__":
    main()