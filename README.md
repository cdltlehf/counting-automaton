# Counting Automaton

## Set up

Create a virtual environment and install the packages in `src/` using the commands in the code block below.
After the `pip install`, you should be able to import `cai4py` from Python scripts in arbitrary locations.

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --editable . # install all packages in `src/` in editable mode (no need to reinstall after making changes)
```

The following sections assume that the virtual environment `.ca-venv` is active.


## Code style

```bash
python -m black .
```

## Linting

```bash
python -m pylint src tests
```

## Preprocessing

When `raw-data/polyglot/all_regexes.jsonl` exists:

```bash
python scripts/preprocessing/preprocess_polyglot.py
make computation-comparison
```

## Dataset

```bash
raw-data/polyglot/all_regexes.jsonl
raw-data/polyglot/sl_regexes.jsonl

./data/patterns/examples.txt

$ scripts/preprocessing/preprocess_polyglot.py

./data/patterns/all_regexes.txt
./data/patterns/sl_regexes.txt

$ make filter

./filtered-patterns/*

$ make test-cases

./test-cases/*
```

## Case-study for example test cases

You can find example test cases in `./data/test-cases/example.txt`.

You can get the analysis results and plots for the example test cases by
running the following command.

```bash
make PATTERN_BASENAMES=example.txt computation-comparison
```

It runs the following commands.

### Computation info

```bash
python scripts/analysis/computation_info.py \
    --method {super_config,bounded_super_config,...} \
    < ./data/test-cases/example.txt
    > ./data/analysis/dynamic/computation_info/example-{method}.jsonl
    2> dev/null
```

You can find the results in
`./data/analysis/dynamic/computation_info/example-{method}.jsonl`.

You can run the following to print the results into `stdout` and logs into
`stderr`.

```bash
python scripts/figures/computation_info.py --method super_config
```

### Draw plots

```bash
python scripts/figures/plot_computation_comparison.py \
    --x-label <method-1> \
    --y-label <method-2> \
    data/analysis/dynamic/computation-info/example-<method-1>.jsonl \
    data/analysis/dynamic/computation-info/example-<method-2>jsonl \
    data/figures/computations-comparison/example-<method-1>-<method-2>.pdf
```

You can find the plots in
`data/figures/computations-comparison/example-<method-1>-<method-2>.pdf`.
