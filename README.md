# Counting Automaton

## Preprocessing

When `raw-data/polyglot/all_regexes.jsonl` exists:

```bash
$ source ./env
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ python scripts/preprocessing/preprocess_polyglot.py
$ make computation-comparison
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
## Cost of Super Config Computation

### Backtracking

- Guard check
- Action check
- Symbol check
