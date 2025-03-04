RAW_DATA_DIR := raw-data
# RAW_DATA := $(addprefix $(RAW_DATA_DIR)/, polyglot.txt regexlib_crawled.txt snort29.txt snort30.txt)
RAW_DATA := $(addprefix $(RAW_DATA_DIR)/, $(shell ls -Sr $(RAW_DATA_DIR)))
DATA_DIR := data
NORMALIZED_DIR := $(DATA_DIR)/normalized
NORMALIZED := $(subst $(RAW_DATA_DIR), $(NORMALIZED_DIR), $(RAW_DATA))

TEST_CASE_DIR := $(DATA_DIR)/test_cases
TEST_CASE := $(subst $(RAW_DATA_DIR), $(TEST_CASE_DIR), $(RAW_DATA))

SUPER_CONFIG_SEQUENCE_DIR := $(DATA_DIR)/super-config-sequence
SUPER_CONFIG_SEQUENCE := $(subst $(RAW_DATA_DIR), $(SUPER_CONFIG_SEQUENCE_DIR), $(RAW_DATA:.txt=.json))

SYNCHRONIZING_DIR := $(DATA_DIR)/synchronizing
SYNCHRONIZING := $(subst $(RAW_DATA_DIR), $(SYNCHRONIZING_DIR), $(RAW_DATA:.txt=.json))

VENV := .venv
PYTHONPATH := ${PYTHONPATH}:$(PWD)/src
PYTHON := PYTHONPATH=$(PYTHONPATH) $(VENV)/bin/python3

$(NORMALIZED_DIR)/%.txt: $(RAW_DATA_DIR)/%.txt
	echo $(PYTHONPATH)
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/normalize.py < $< > $@ || rm $@

.PHONY: normalize
normalize: $(NORMALIZED)

$(TEST_CASE_DIR)/%.txt: $(NORMALIZED_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/append_test_cases.py < $< > $@ || rm $@

.PHONY: test-cases
test-cases: $(TEST_CASE)

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
