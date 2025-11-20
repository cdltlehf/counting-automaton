import numpy as np
import json

import benchmark_utils as bench

"""
    Benchmarking for different quantifier bounds. Ended up not being used.
"""

regexes = [
        (lambda k: f"a{{{k}}}", r"a\{k\}", "Pattern1"),
        (lambda k: f".*a.{{{k}}}", r".*a.\{k\}", "Pattern2"),
        (lambda k: f".*a.{{{k//2},{k}}}", r".*a.\{k/2,k\}", "Pattern3"),
        (lambda k: f".*a.{{{k},}}", r".*a.\{k,\}", "Pattern4"),
        (lambda k: f"a*a{{{(98*k)//100},{k}}}", r"a*a\{98k/100,k\}", "Pattern5"),
        (lambda k: f"(aa|a){{{k//2},{k}}}", r"(aa$\vert$a)\{k/2,k\}", "Pattern6"),
    ]

TIMEOUT = 4000  # ms

def runtime_benchmark(config, upper_limit: int, interval: int):
    x = np.concatenate((np.array([1], dtype=int), np.arange(interval, upper_limit + interval, interval, dtype=int)))

    fn, title, experiment_ID = config
    results = {"Regex": title, "Range": x.tolist(), "Results": {}}
    for method in bench.methods + [bench.flatten]:  # Different methods
        print(f"Method: {method}")

        y = np.zeros_like(x, dtype=float)
        for i, k in enumerate(x):   # values of k
            word = "a" * k

            runtime = bench.runtime(method, word, fn(k))
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
    filename = f"DBR/{experiment_ID}.json"
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

def density_benchmark(config, upper_limit: int, interval: int):
    x = np.concatenate((np.array([1], dtype=int), np.arange(interval, upper_limit + interval, interval, dtype=int)))

    fn, title, experiment_ID = config

    results = {"Regex": title, "Range": x.tolist(), "Results": {}}
    for method in bench.density_methods:    # Only counting-set and sparse

        print(f"Method: {method}")

        y = np.zeros_like(x, dtype=int)
        for i, k in enumerate(x):
            word = "a" * k
            data_collection = bench.data_experiment(fn(k), word, method)
            y[i] = int(bench.extract_max_density(data_collection))

        if method.value != "Sparse Counting-set":
            results["Results"]["Dense Counters"] = y.tolist()
        else:
            results["Results"][method.value] = y.tolist()

    # Write runtimes to JSON
    filename = f"DBD/{experiment_ID}.json"
    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    
    runtime_benchmark(regexes[0], 2000, 100)
    density_benchmark(regexes[0], 2000, 100)
