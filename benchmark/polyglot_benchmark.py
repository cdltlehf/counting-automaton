import benchmark_utils as bench
import cai4py.utils.ambiguity_analysis as ambiguity

import cai4py.counting_automaton.position_counting_automaton as pca
import cai4py.parser_tools as pt

import re

from pathlib import Path
import os
import argparse
import json

current_dir = Path(__file__).resolve().parent
ambiguous_dir = current_dir / "ambiguous"
unambiguous_dir = current_dir / "unambiguous"

# Parsing of arguments
parser = argparse.ArgumentParser(description="Polyglot Corpus runtime benchmark.")
parser.add_argument("--category", type=int, default=1, help="Category to perform Polyglot Corpus benchmark: 1 - 0 < k <= 50; 2 - 50 < k <= 100; 3 - 100 < k <= 200.")
args = parser.parse_args()

"""
    Divide the Polyglot Corpus into different categories based on input length
    and upper bound of quantifier.
"""
def process(directory: str, output: str, min: int = 50, max: int = 150, upper: int = 64):

    entries = list()

    for filename in os.listdir(directory):
        samples = []

        with open(os.path.join(directory, filename), "r") as file:
            data = json.load(file)
            for element in data:
                regex = element["regex"]
                # Check positive string
                if "positive" in element and len(element["positive"]) < max and len(element["positive"]) > min:
                        # Check bounds
                        if correct_bounds(regex, upper):
                            # Check if regex can be parsed
                            if is_parsable(regex):
                                element.pop("negative", None)
                                samples.append(element)
            entries.extend(samples)

    print(f"Total valid entires: {len(entries)}")

    with open(f"{output}.json", "w") as file:
        json.dump(entries, file, indent=4)

"""
    Separate ambiguous/unambiguous patterns.
"""
def process_ambiguity(filename: str, output: str):

    ambiguous_entries = list()
    unambiguous_entries = list()

    with open(filename, "r") as file:
        data = json.load(file)

        for i, element in enumerate(data):
            regex = element["regex"]
            print(i)

            ambiguous = ambiguity.is_ambiguous(regex)
            if ambiguous is not None and ambiguous:
                ambiguous_entries.append(element)
            elif ambiguous is not None and not ambiguous:
                unambiguous_entries.append(element)

    print(f"Total valid entires: {len(ambiguous_entries) + len(unambiguous_entries)}")
    print(f"\tAmbiguous entires: {len(ambiguous_entries)}")
    print(f"\tUnambiguous entires: {len(unambiguous_entries)}")

    with open(f"ambiguous/{output}.json", "w") as file:
        json.dump(ambiguous_entries, file, indent=4)

    with open(f"unambiguous/{output}.json", "w") as file:
        json.dump(unambiguous_entries, file, indent=4)

def correct_bounds(regex: str, upper: int = 10000) -> bool:
    quantifier_capture = re.compile(r"\{[0-9]*,[0-9]*\}")

    quantifiers = quantifier_capture.findall(regex)
    for quantifer in quantifiers:
        numbers = quantifer[1:-1]
        bounds = numbers.split(",")

        if bounds[1] == "" or int(bounds[1]) > upper:
            return False
        
    return True

def is_parsable(regex) -> bool:
    try:
        parsed_regex = pt.parse(regex)
        normalised_regex = pt.flatten_inner_quantifiers(parsed_regex)
        pattern = pt.to_string(normalised_regex)
        pca.PositionCountingAutomaton.create(pattern)
    except:
        return False

    return True

"""
    Perform actual benchmark of Polyglot Corpus.
"""
def runtime_benchmark(input: str, output: str):
    
    results = {"Results": {}}
    for method in bench.methods + [bench.flatten]:
 
        print(f"Method: {method}")

        with open(f"{input}.json", "r") as file:
            data = json.load(file)

            total = 0
            for entry in data:
                
                regex = entry["regex"]
                word = entry["positive"]

                runtime = bench.runtime(method, word, regex, timeout=120)
                if runtime is not None:
                    total += float(runtime * 1000)  # Convert to milliseconds
                else:
                    total = -1000
                    print("Method did not finish!")
                    break

        if method != bench.flatten:
            results["Results"][method.value] = total
        else:
            results["Results"][method] = total

    # Write runtimes to JSON
    with open(f"{output}.json", "w") as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":

    if args.category == 1:
        print(f"Starting Polyglot Corpus benchmark: 0 < k <= 50")
        runtime_benchmark(f"{ambiguous_dir}/test_samples_0_50", f"{ambiguous_dir}/test_results_0_50")
        runtime_benchmark(f"{unambiguous_dir}/test_samples_0_50", f"{unambiguous_dir}/test_results_0_50")
    elif args.category == 2:
        print(f"Starting Polyglot Corpus benchmark: 50 < k <= 100")
        runtime_benchmark(f"{ambiguous_dir}/test_samples_50_100", f"{ambiguous_dir}/test_results_50_100")
        runtime_benchmark(f"{unambiguous_dir}/test_samples_50_100", f"{unambiguous_dir}/test_results_50_100")
    elif args.category == 3:
        print(f"Starting Polyglot Corpus benchmark: 100 < k <= 200")
        runtime_benchmark(f"{ambiguous_dir}/test_samples_100_200", f"{ambiguous_dir}/test_results_100_200")
        runtime_benchmark(f"{unambiguous_dir}/test_samples_100_200", f"{unambiguous_dir}/test_results_100_200")
    else:
        print(f"Invalid category given: {args.category}")
    