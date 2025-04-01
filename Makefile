VENV := .venv
PYTHONPATH := ${PYTHONPATH}:$(PWD)/src
ifdef RELEASE
PYTHON := PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python3 -O
else
PYTHON := PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python3
endif

DATA_DIR := data
PATTERNS_DIR := $(DATA_DIR)/patterns
PATTERN_BASENAMES := all_regexes.txt
# PATTERNS := $(wildcard $(PATTERNS_DIR)/*.txt)
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

file-server:
	@echo $$(curl -s ifconfig.me)
	$(PYTHON) -m http.server 50000
