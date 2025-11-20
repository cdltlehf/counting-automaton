VENV := .venv

PYTHON := $(VENV)/bin/python3
PYFLAGS := -O

REQ := ./requirements.txt

METHOD ?= 1
REGEX ?= 1
CATEGORY ?= 1

.PHONY: help
help:
	@echo 'Makefile for Counting Automaton'
	@echo 'make setup:                  Set up the Python virtual environment and install dependencies.'
	@echo 'make run {method}:           Run the main application.'
	@echo '    METHOD={specify counter method, default is 1}.'
	@echo '    Methods: 1 - Bit Vector; 2 - Naive Counter; 3 - Counting-set; 4 - Sparse Counting-set; 5 - Counter Expansion'
	@echo 'make unittest:               Run the unit-test suite.'
	@echo 'make corpustest {method}:   Run the Polyglot Corpus test suite.'
	@echo '    METHOD={specify counter method, default is 1}.'
	@echo '    Methods: 1 - Bit Vector; 2 - Naive Counter; 3 - Counting-set; 4 - Sparse Counting-set; 5 - Counter Expansion'
	@echo 'make benchmark {regex}:      Run the large quantifier benchmark.'
	@echo '    REGEX={specific regex, default is 1}.'
	@echo '    Regexes: 1 - a{2000}; 2 - *a.{2000}; 3 - .*a.{1000,2000}; 4 - .*a.{1000,}; 5 - a*a{1998,2000}; 6 - (aa|a){1000,2000}'
	@echo 'make corpus {category}:      Run Polyglot Corpus benchmark.'
	@echo '    CATEGORY={specific category, default is 1}.'
	@echo '    Categories: 1 - 0 < k <= 50; 2 - 50 < k <= 100; 3 - 100 < k <= 200'
	@echo 'make clean:                  Remove the virtual environment.'

.PHONY: setup
setup: 
	$(PYTHON) -m venv $(VENV)
	@echo "Created virtual environment in $(VENV)"
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r $(REQ)
	$(VENV)/bin/pip install -e .
	@echo "Installed dependencies from $(REQ)"

run:
	$(VENV)/bin/$(PYTHON) demo/demo.py --method $(METHOD)

unittest:
	$(VENV)/bin/$(PYTHON) -m unittest discover -v validation -p "*_test.py"

corpustest:
	$(VENV)/bin/$(PYTHON) validation/corpus_validation.py --method $(METHOD)

benchmark:
	$(VENV)/bin/$(PYTHON) benchmark/static_benchmark.py --regex $(REGEX)

corpus: 
	$(VENV)/bin/$(PYTHON) benchmark/polyglot_benchmark.py --category $(CATEGORY)


DATA_DIR := data
PATTERNS_DIR := $(DATA_DIR)/patterns
SCRIPT_DIR := src/cai4py/scripts
PATTERN_BASENAMES := all_regexes.txt
PATTERNS := $(addprefix $(PATTERNS_DIR)/, $(PATTERN_BASENAMES))
METHODS := \
	   super_config \
	   bounded_super_config \
	   counter_config \
	   bounded_counter_config \
	   sparse_counter_config \
	   determinized_counter_config \
	   determinized_bounded_counter_config \
	   determinized_sparse_counter_config

all: computation-comparison

include makefiles/data.mk
include makefiles/analysis.mk
include makefiles/figures.mk

clean-data: # Remove all generated
	- rm -rf $(DATA_DIR)/figures/ $(DATA_DIR)/analysis/

clean-caches: # Remove all __pycache__ directories
	- find . -name "__pycache__" -type d -exec rm -r {} +

clean-eggs: # Remove all .egg-info directories
	- find . -name "*.egg-info" -type d -exec rm -r {} +

clean-venv: # Remove the virtual environment
	- rm -rf $(VENV)
	@echo "Removed virtual environment $(VENV)"

clean-all: clean-data clean-caches clean-eggs clean-venv

file-server:
	curl -s https://ifconfig.me
	$(PYTHON) $(PYFLAGS) -m http.server 50000

.PHONY: all clean-data clean-caches clean-eggs clean-venv clean-all clean-venv run unittest corpustest benchmark corpus
