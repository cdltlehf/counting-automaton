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

include makefiles/data.mk
include makefiles/analysis.mk
include makefiles/figures.mk

SUPER_CONFIG_SEQUENCE_DIR := $(DATA_DIR)/super-config-sequence
SUPER_CONFIG_SEQUENCE := $(subst $(RAW_DATA_DIR), $(SUPER_CONFIG_SEQUENCE_DIR), $(RAW_DATA:.txt=.json))

SYNCHRONIZING_DIR := $(DATA_DIR)/synchronizing
SYNCHRONIZING := $(subst $(RAW_DATA_DIR), $(SYNCHRONIZING_DIR), $(RAW_DATA:.txt=.json))

$(SUPER_CONFIG_SEQUENCE_DIR)/%.json: $(TEST_CASE_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/compute_super_config_sequences.py < $< > $@ || rm $@

.PHONY: super-config-sequence
super-config-sequence: $(SUPER_CONFIG_SEQUENCE)


$(SYNCHRONIZING_DIR)/%.json: $(NORMALIZED_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/analysis_synchronizing.py < $< > $@ || rm $@

.PHONY: synchronizing
synchronizing: $(SYNCHRONIZING)

test:
	$(PYTHON) -m unittest tests/*.py

file-server:
	@echo $$(curl -s ifconfig.me)
	$(PYTHON) -m http.server 50000
