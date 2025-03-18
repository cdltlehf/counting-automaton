# Counting Automaton

## Test

```bash
$ source ./env
$ python3 test.py
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
