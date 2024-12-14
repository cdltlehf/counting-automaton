include .env
export

RAW_DATA_DIR := raw-data
# RAW_DATA := $(addprefix $(RAW_DATA_DIR)/, polyglot.txt regexlib_crawled.txt snort29.txt snort30.txt)
RAW_DATA := $(addprefix $(RAW_DATA_DIR)/, $(shell ls $(RAW_DATA_DIR)))
DATA_DIR := data
NORMALIZED_DIR := $(DATA_DIR)/normalized
NORMALIZED := $(subst $(RAW_DATA_DIR), $(NORMALIZED_DIR), $(RAW_DATA))

VENV := .venv
PYTHON := $(VENV)/bin/python3

$(NORMALIZED_DIR)/%.txt: $(RAW_DATA_DIR)/%.txt
	mkdir -p $(dir $@)
	$(PYTHON) scripts/normalize.py < $< > $@

normalized: $(NORMALIZED)

tmp:
	@echo $(RAW_DATA)
	@echo $(NORMALIZED)

test:
	$(PYTHON) -m unittest tests/*.py
