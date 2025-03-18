FILTERED_PATTERNS_DIR := $(DATA_DIR)/filtered-patterns
FILTERED_PATTERNS := $(subst $(PATTERNS_DIR),$(FILTERED_PATTERNS_DIR),$(PATTERNS))
TEST_CASES_DIR := $(DATA_DIR)/test-cases
TEST_CASES := $(subst $(FILTERED_PATTERNS_DIR),$(TEST_CASES_DIR),$(FILTERED_PATTERNS))

$(FILTERED_PATTERNS_DIR)/%.txt: $(PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/filter.py < $< > $@ || rm $@

.PHONY: filter
filter: $(FILTERED_PATTERNS)

$(TEST_CASES_DIR)/%.txt: $(FILTERED_PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) scripts/append_test_cases.py < $< > $@ || rm $@

.PHONY: test-cases
test-cases: $(TEST_CASES)
	@echo "Generate $(TEST_CASES)"
