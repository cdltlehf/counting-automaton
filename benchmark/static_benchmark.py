import benchmark_utils as bench

import matplotlib.pyplot as plt

import numpy as np
import argparse
import json

"""
    Benchmarking for different regexes with large quantifier bounds.
"""

# Parsing of arguments
parser = argparse.ArgumentParser(description="Benchmark.")
parser.add_argument("--regex", type=int, default=1, help="Select predefined regular expression to perform benchmark on.")
args = parser.parse_args()

regexes = [
        ("a{2000}", r"a\{2000\}", "Pattern1", 2000, 100),
        (".*a.{2000}", r".*a.\{2000\}", "Pattern2", 2000, 100),
        (".*a.{1000,2000}",  r".*a.\{1000,2000\}", "Pattern3", 2000, 100),
        (".*a.{1000,}", r".*a.\{1000,\}", "Pattern4", 2000, 100),
        ("a*a{1998,2000}", r"a*a\{1998,2000\}", "Pattern5", 2000, 100),
        ("(aa|a){1000,2000}", r"(aa$\vert$a)\{1000,2000\}","Pattern6", 2000, 100),
]

TIMEOUT = 4000  # ms

def runtime_benchmark(config):
    regex, title, experiment_ID, upper_limit, interval = config
    x = np.concatenate((np.array([1], dtype=int), np.arange(interval, upper_limit + interval, interval, dtype=int)))

    results = {"Regex": title, "Range": x.tolist(), "Results": {}}
    for method in bench.methods + [bench.flatten]:
        print(f"Method: {method}")

        y = np.zeros_like(x, dtype=float)
        for i, k in enumerate(x):

            word = "a" * k
            runtime = bench.runtime(method, word, regex)

            if runtime is None:
                print("Method has reached a timeout!")
                break
            else:
                y[i] = float(runtime * 1000)  # Convert to milliseconds

        if method != bench.flatten:
            results["Results"][method.value] = y.tolist()
        else:
            results["Results"][method] = y.tolist()

    # Write runtimes to JSON
    filename = f"SBR_{experiment_ID}.json"
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

def density_benchmark(config):

    regex, title, experiment_ID, upper_limit, interval = config
    x = np.concatenate((np.array([1], dtype=int), np.arange(interval, upper_limit + interval, interval, dtype=int)))

    results = {"Regex": title, "Range": x.tolist(), "Results": {}}
    for method in bench.density_methods:

        print(f"Method: {method}")

        y = np.zeros_like(x, dtype=int)
        for i, k in enumerate(x):
            word = "a" * k
            data_collection = bench.data_experiment(regex, word, method)
            y[i] = int(bench.extract_max_density(data_collection))

        if method.value != "Sparse Counting-set":
            results["Results"]["Dense Counters"] = y.tolist()
        else:
            results["Results"][method.value] = y.tolist()

    # Write runtimes to JSON
    filename = f"SBD_{experiment_ID}.json"
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

# The following two functions are not intended to be used in the main benchmarking suite,
# but rather for ad-hoc benchmarking of single regexes. These allow the user to easily generate
# benchmarks that possibly have interesting matching time behaviour. 
colours = ["#1f77b4", "#ff7f0e"]  # Add more colors if needed
linestyles = ["--", "-", "--", ":"]
markers = ["o", "s", "D", "x"]
markersizes = [4, 4, 4, 6]

def multi_runtime_benchmark(regex, upper_limit: int, interval: int, experiment_ID: str):
    x = np.concatenate((np.array([1], dtype=int), np.arange(interval, upper_limit + interval, interval, dtype=int)))

    results = {"Regex": regex, "Range": x.tolist(), "Results": {}}
    for id, method in enumerate(bench.methods + [bench.flatten]):

        print(f"Method: {method}")

        y = np.zeros_like(x, dtype=float)
        for i, k in enumerate(x):
            word = "a" * k

            runtime = bench.runtime(method, word, regex)
            if runtime * 1000 > TIMEOUT:
                print("Method has reached a timeout!")
                break
            else:
                y[i] = float(runtime * 1000)  # Convert to milliseconds

        method_str = method.value if method != bench.flatten else method

        plt.plot(
            x[:i], y[:i],
            label=method_str,
            color=bench.method_colour[method],
            linewidth=2,
            linestyle=linestyles[id % len(linestyles)],
            marker=markers[id % len(markers)],
            markersize=markersizes[id % len(markersizes)]
        )

        results["Results"][method_str] = y.tolist()

    # Write runtimes to JSON
    filename = f"AdHoc_{experiment_ID}.json"
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

    plt.title(f"`{regex}'", fontsize=13, fontweight='bold')
    plt.xlabel("Input Length", fontsize=11)
    plt.ylabel("CPU Time (ms)", fontsize=11)
    plt.legend()
    plt.savefig(f"AdHoc_{experiment_ID}.pdf")
    plt.close() 

def single_runtime_benchmark(regex, upper_limit: int, interval: int, method, experiment_ID: str):
    x = np.concatenate((np.array([1], dtype=int), np.arange(interval, upper_limit + interval, interval, dtype=int)))

    results = {"Regex": regex, "Range": x.tolist(), "Results": {}}
 
    y = np.zeros_like(x, dtype=float)
    for i, k in enumerate(x):
        word = "a" * k

        runtime = bench.runtime(method, word, regex)
        if runtime * 1000 > TIMEOUT:
            print("Method has reached a timeout!")
            break
        else:
            y[i] = float(runtime * 1000)  # Convert to milliseconds

        results["Results"][method] = y.tolist()

    # Write runtimes to JSON
    filename = f"AdHoc_{experiment_ID}.json"
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)   

    plt.plot(
        x[:len(y)], y,
        label=method,
        linewidth=2,
    )
    plt.title(f"`{regex}'", fontsize=13, fontweight='bold')
    plt.xlabel("Input Length", fontsize=11)
    plt.ylabel("CPU Time (ms)", fontsize=11)
    plt.savefig(f"AdHoc_{experiment_ID}.pdf")
    plt.close() 

if __name__ == "__main__":

    if args.regex in range(1,7):
        regex, _, _, _, _ = regexes[args.regex - 1]

        print(f"Running benchmark on {regex}")
        runtime_benchmark(regexes[args.regex - 1])
        density_benchmark(regexes[args.regex - 1])
    else:
        print(f"Unknown regex selected: {args.regex}")
        exit()
