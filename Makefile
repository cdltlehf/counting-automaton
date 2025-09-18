VENV := .venv
PYTHON := $(VENV)/bin/python3
PYFLAGS := -O

.SECONDARY:
.DELETE_ON_ERROR:

DATA_DIR ?= data
PATTERNS ?= polyglot.txt snort3.txt
METHODS ?= \
counter_expansion \
bounded_super_config \
bounded_counter_config \
sparse_counter_config

# METHODS := \
# 	   super_config \
# 	   bounded_super_config \
# 	   counter_config \
# 	   bounded_counter_config \
# 	   sparse_counter_config \
# 	   determinized_counter_config \
# 	   determinized_bounded_counter_config \
# 	   determinized_sparse_counter_config
#

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

clean-all: clean-data clean-caches clean-eggs clean-venv

file-server:
	curl -s https://ifconfig.me
	$(PYTHON) $(PYFLAGS) -m http.server 50000

.PHONY: all clean-data clean-caches clean-eggs clean-venv clean-all
