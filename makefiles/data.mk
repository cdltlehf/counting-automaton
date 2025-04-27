FILTERED_PATTERNS_DIR := $(DATA_DIR)/filtered
FILTERED_PATTERNS := $(addprefix $(FILTERED_PATTERNS_DIR)/,$(PATTERN_BASENAMES))

$(info $(FILTERED_PATTERNS))

TEST_CASES_DIR := $(DATA_DIR)/test-cases
TEST_CASES := $(addprefix $(TEST_CASES_DIR)/,$(PATTERN_BASENAMES))

$(FILTERED_PATTERNS_DIR)/%.txt: $(PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	- $(PYTHON) $(PYFLAGS) scripts/data/filter.py < $< > $@

.PHONY: filter
filter: $(FILTERED_PATTERNS)

.SECONDARY: $(FILTERED_PATTERNS)

$(TEST_CASES_DIR)/%.txt: $(FILTERED_PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	# - $(PYTHON) $(PYFLAGS) scripts/data/append_test_cases.py < $< > $@
	- $(PYTHON) $(PYFLAGS) scripts/data/append_test_cases_evil_str_gen.py < $< > $@

.PHONY: test-cases
test-cases: $(TEST_CASES)
