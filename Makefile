VENV := .venv
PYTHONPATH := ${PYTHONPATH}:$(PWD)/src
PYTHON := PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python3

DATA_DIR := data
PATTERNS_DIR := $(DATA_DIR)/patterns
# PATTERNS := $(wildcard $(PATTERNS_DIR)/*.txt)
PATTERNS := $(wildcard $(PATTERNS_DIR)/all_regexes.txt)

include makefiles/data.mk
include makefiles/computation.mk
include makefiles/analysis.mk

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
	@echo "http://165.132.106.173:10000"
	$(PYTHON) -m http.server 50000
