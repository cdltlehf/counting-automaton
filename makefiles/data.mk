PATTERNS_DIR := $(DATA_DIR)/patterns
TEST_CASES_DIR := $(DATA_DIR)/test-cases

FILTER_PREFIXES := ambiguous_filtered filtered
TESTCASE_PREFIXES := evilstrgen xeger

FILTERED_PATTERNS := \
$(addprefix $(PATTERNS_DIR)/,\
	$(foreach prefix,$(FILTER_PREFIXES),\
		$(addprefix $(prefix)_, $(PATTERNS))\
	)\
)

TEST_CASES := \
$(addprefix $(TEST_CASES_DIR)/,$(addsuffix .jsonl,\
	$(foreach prefix,$(TESTCASE_PREFIXES),\
		$(addprefix $(prefix)_,\
			$(notdir $(basename $(FILTERED_PATTERNS)))\
		)\
	)\
))

$(PATTERNS_DIR)/%.txt: raw-data/%.txt
	@mkdir -p $(dir $@)
	cp $< $@

$(PATTERNS_DIR)/normalized_%.txt: $(PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) $(PYFLAGS) scripts/data/normalize.py < $< > $@

$(PATTERNS_DIR)/filtered_%.txt: $(PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) $(PYFLAGS) scripts/data/filter.py < $< > $@

$(PATTERNS_DIR)/ambiguous_%.txt $(PATTERNS_DIR)/unambiguous_%.txt: $(PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) $(PYFLAGS) scripts/data/run_checker.py \
		--unambiguous $(PATTERNS_DIR)/unambiguous_$*.txt < $< \
		> $(PATTERNS_DIR)/ambiguous_$*.txt

$(TEST_CASES_DIR)/evilstrgen_%.jsonl: $(PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) $(PYFLAGS) scripts/data/append_test_cases_evil_str_gen.py < $< > $@

$(TEST_CASES_DIR)/xeger_%.jsonl: $(PATTERNS_DIR)/%.txt
	@mkdir -p $(dir $@)
	$(PYTHON) $(PYFLAGS) scripts/data/append_test_cases.py < $< > $@

.PHONY: test-cases
test-cases: $(TEST_CASES)
	@echo $(TEST_CASES)
