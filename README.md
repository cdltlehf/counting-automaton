# Counting Automaton

## Additions

Non-exhaustive list of changes/additions:

- `custom_counters` contains implemented counters along with base class and `counter_type` for easy switching between counters
- `super_config.py` has changes in `update`, `match` and `__init__`. This class also previously had a base class which is removed.
- `position_counting_automaton.py` has changes in `get_initial_config`, `check_final` and `get_next_configs`. Most of these changes are there to allow for the switching of counter types.
- `counter_vector.py` has changes in all of its classes. This is again to allow for the switching of classes.
- `__init__.py` of `parser_tools` has a counter expansion methods in `flatten_inner_quantifers` and `flatten_quantifiers`. These use `quantifer_fold` and `flatten` to perform the counter expansion. These methods are modelled of the original `normalize` and `fold` methods. Feel free to rename the methods from `flatten` to `expansion`.

Please look at `demo/demo.py` for an example of how the automaton and matchers are used. These are very similar to the original implementation with the addition of counter selection. In the counters and the return of the `SuperConfig` matcher are places where data collection is performed. This is still being developed for my project so feel free to remove it.

## Set up

Create a virtual environment and install the packages in `src/` using the commands in the code block below.
After the `pip install`, you should be able to import `cai4py` from Python scripts in arbitrary locations.

```bash
# Create a virtual environment
python3 -m venv .ca-venv
source .ca-venv/bin/activate
pip install --editable . # install all packages in `src/` in editable mode (no need to reinstall after making changes)
```

The following sections assume that the virtual environment `.ca-venv` is active.

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
